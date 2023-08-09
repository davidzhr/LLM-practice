import torch
import numpy
import time


def describe(x: torch.Tensor):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


describe(torch.Tensor([[1,2]]))
describe(torch.Tensor(2, 3))
describe(torch.rand(2, 3))
describe(torch.randn(2, 3))


x = torch.ones(2, 3)
describe(x)
x.fill_(4)
describe(x)

print(torch.cuda.is_available())

device = "cuda:0"
x = torch.randn((3,3), device=device)
describe(x)

# Test  GPU
while True:
    x = x.mm(x)
    print(x)
    time.sleep(5)







