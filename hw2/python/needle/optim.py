"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            ut = self.u[p] if p in self.u else 0
            ut1 = self.momentum  * ut + (1 - self.momentum) * (p.grad.detach() + self.weight_decay * p.detach())
            self.u[p] = ut1
            p.data -= ndl.Tensor(self.lr *  ut1, dtype=p.dtype)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params :
            mt = self.m[p] if p in self.m else 0
            vt = self.v[p] if p in self.v else 0
            if p is None or p.grad is None:
                print(p)
            fg = p.grad.data + self.weight_decay * p.data

            mt1 = mt * self.beta1 + (1-self.beta1) * fg.data
            vt1 = vt * self.beta2 + (1-self.beta2) * (fg.data ** 2)
            self.m[p] = mt1.data
            self.v[p] = vt1.data

            hmt1 = mt1.data / (1 - self.beta1 ** self.t)
            hvt1 = vt1.data / (1 - self.beta2 ** self.t)

            p.data -= ndl.Tensor(self.lr * hmt1.data / (hvt1.data ** 0.5 + self.eps) , dtype = p.dtype)
        ### END YOUR SOLUTION
