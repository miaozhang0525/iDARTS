import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.utils
from copy import deepcopy
import utils
import logging
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)


    def _compute_unrolled_model(self, train_queue, eta, network_optimizer):

        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()    

        #for step, (input, target) in enumerate(train_queue):
        for step in range(5):
            input, target = next(iter(train_queue))		


            self.model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            network_optimizer.zero_grad()
            logits = self.model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 5)
            network_optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)


    def step(self, train_queue, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()

        if unrolled:
            self._backward_step_unrolled(train_queue, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)

        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        #print(loss)
        loss.backward()

    def _backward_step_unrolled(self, train_queue, input_valid, target_valid, eta, network_optimizer):
        self._compute_unrolled_model(train_queue, eta, network_optimizer)######copy a model for the L_val, since the model should nog trained by validation data

        unrolled_loss = self.model._loss(input_valid, target_valid)
        #print(unrolled_loss)
        unrolled_loss.backward()
        dalpha = [v.grad for v in self.model.arch_parameters()]
        vector = [v.grad.data for v in self.model.parameters()]


        implicit_grads = self._hessian_vector_product(vector, train_queue, eta)######this should be L_train(w*,a), so the data should be train

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(self, vector, train_queue, eta, r=1e-2):
		
        input, target = next(iter(train_queue))     
        n = input.size(0)
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)
        
        		
        p_model = deepcopy(self.model)    

        loss = p_model._loss(input, target)         
        grads = torch.autograd.grad(loss, p_model.parameters(),create_graph=True)   
        prod=sum([(g*v).sum() for g,v in zip(grads, vector)]) 
        prod.backward()    
        grads_1=[v-eta*p.grad.data for v,p in zip(vector, p_model.parameters())]         
        del p_model               
        
        p_model = deepcopy(self.model)    

        loss = p_model._loss(input, target)         
        grads = torch.autograd.grad(loss, p_model.parameters(),create_graph=True)   
        prod=sum([(g*v).sum() for g,v in zip(grads, grads_1)]) 
        prod.backward()    
        grads_2=[v-eta*p.grad.data for v,p in zip(grads_1, p_model.parameters())]         
        del p_model               
 
     
        grads_sum=[v+p+g for v,p,g in zip(vector, grads_1, grads_2)]         
                 
		
        R = r / _concat(vector).norm()
        for p, g in zip(self.model.parameters(), grads_sum):
            p.data.add_(R, g)


        loss = self.model._loss(input, target)                


        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
        #print(grads_p)

        for p, g in zip(self.model.parameters(), grads_sum):
            p.data.sub_(2*R, g)

        loss = self.model._loss(input, target)                 
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    
