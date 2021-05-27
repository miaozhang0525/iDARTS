import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from vadam.optimizers import Vadam   
from vadam.metrics import avneg_loglik_categorical



objective = avneg_loglik_categorical



def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)        
               
        self.optimizer = Vadam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
           prec_init = 1,train_set_size = 25000, num_samples=args.num_mc)
        self.args=args

    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)


    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
        return unrolled_model


    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, temper):
        self.optimizer.zero_grad()
        #self.model.arch_parameters()         
        if unrolled:
            #self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
            unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
            
            def closure():
				#######I need to change here to get the constrained solution
                logits = unrolled_model(input_valid)
                loss = objective(logits, target_valid)
                loss.backward()
                
                dalpha = [v.grad for v in unrolled_model.arch_parameters()]
                vector = [v.grad.data for v in unrolled_model.parameters()]
                implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

                for g, ig in zip(dalpha, implicit_grads):
                    g.data.sub_(eta, ig.data)
                for v, g in zip(self.model.arch_parameters(), dalpha):
                    if v.grad is None:
                        v.grad = Variable(g.data)
                    else:
                        v.grad.data.copy_(g.data) 
                return loss            
            
            self.optimizer.step(closure)
              
        else:
            #self._backward_step(input_valid, target_valid)
            def closure():
				#######I need to change here to get the constrained solution
                logits = self.model(input_valid)
                loss = objective(logits, target_valid)                  
                loss.backward()
                return loss                       
            self.optimizer.step(closure)
        #n_vector = [v.grad.data for v in self.model.arch_parameters()]
        #norm_n,norm_c=self._hessian_norm(self, input_valid, target_valid)
        #return norm_n,norm_c

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
        
       

        
        
