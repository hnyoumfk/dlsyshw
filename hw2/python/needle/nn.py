"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1), (1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ret = ops.matmul(X , self.weight) 
        if self.use_bias:
            ret = ret + ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
        return ret
        ### END YOUR SOLUTION



from functools import reduce
class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        old_shape = X.shape
        new_shape = (old_shape[0], reduce(lambda x,y: x*y , old_shape[1:]))
        return ops.reshape(X, new_shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ret = x
        for mod in self.modules:
            ret = mod(ret)
        return ret
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        oh = init.one_hot(logits.shape[1], y)
        zy = ops.summation(oh * logits, axes=(1,))
        return ops.summation(ops.logsumexp(logits, axes=(1,)) - zy) / y.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device))
        self.bias = Parameter(init.zeros(self.dim, device=device))
        self.running_mean = init.zeros(self.dim, device=device)
        self.running_var = init.ones(self.dim, device=device)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training :
            e = ops.summation(x, (0,)) / x.shape[0]
            sub_x = x - ops.broadcast_to(e, x.shape)
            var = ops.summation( ops.power_scalar(sub_x, 2), (0,) ) / x.shape[0]
            broadcast_var = ops.broadcast_to(var, x.shape)
            broadcast_weight = ops.broadcast_to(self.weight, x.shape) 
            broadcast_bias = ops.broadcast_to(self.bias, x.shape)
            ret = broadcast_weight * (sub_x / ops.power_scalar(broadcast_var+self.eps, 0.5)) + broadcast_bias

            self.running_mean = self.running_mean * (1-self.momentum) + self.momentum * e  
            self.running_var = self.running_var * (1-self.momentum) + self.momentum * var
            return ret
        else :
            y = self.weight * ( x - self.running_mean ) * (self.running_var + self.eps) ** -0.5 + self.bias
            return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device))
        self.bias = Parameter(init.zeros(1, dim, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.transpose(x)
        e = ops.broadcast_to(ops.summation(x, (0,)) / self.dim, x.shape)
        sub_x = x - e
        var = ops.broadcast_to(ops.summation( ops.power_scalar(sub_x, 2), (0,) ) / self.dim , x.shape)
        result = ops.transpose(sub_x / ops.power_scalar(var+self.eps, 0.5))
        broadcast_weight = ops.broadcast_to(self.weight, result.shape) 
        broadcast_bias = ops.broadcast_to(self.bias, result.shape)
        ret = broadcast_weight * result  + broadcast_bias
        return ret
        ### END YOUR SOLUTION



class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training :
            drop = init.randb(*(x.shape), p=1-self.p) / (1-self.p)
            return x * drop
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ret = self.fn(x)
        return ret + x
        ### END YOUR SOLUTION



