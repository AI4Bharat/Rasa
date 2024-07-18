'''https://tissue333.gitbook.io/cornell/findings/pytorch_backward'''
import torch
from torch.autograd.function import InplaceFunction

# class MyLayer(torch.nn.Module):
#     def __init__(self):
#         mytensor = torch.tensor(3,3)
#         torch.nn.init.uniform(mytensor,0,1)
#         #add parameter with nn.Parameter()
#         #by default, it set requires gradient to true
#         self.mytensor = torch.nn.Parameter(mytensor)

#     def forward(self,input):
#         #use method provided by pytorch here
#         #it supports autograd, namely, auto-gradient computation
#         return input*self.mytensor

# class RoundGradient(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         return x.round()
#     @staticmethod
#     def backward(ctx, g):
#         return g 

# class ClampGradient(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         return x.clamp(min=0,max=1)

#     @staticmethod
#     def backward(ctx, g):
#         x, = ctx.saved_tensors
#         grad_input = g.clone()
#         grad_input[x < 0] = 0
#         grad_input[x>1] = 1
#         return grad_input

'''https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152'''
class special_round(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class special_clamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.input = input
        return torch.clamp(input, min, max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


if __name__ == '__main__':
    # x = torch.nn.Parameter(torch.randn(1,6), requires_grad = True)
    # x = torch.randn(1,2)
    # x = RoundGradient.apply(x)
    # x.backward()
    # print(x)
   x = torch.randn(1,2)
   x.requires_grad_()
   x = special_round.apply(x)
   x = x.sum()
   x.backward()
   print(x, x.grad) 
   
   
   x = torch.randn(1,2)
   x.requires_grad_()
   x = special_clamp.apply(x, 0, 1e-5)
   x = x.sum()
   x.backward()
   print(x, x.grad) 
   