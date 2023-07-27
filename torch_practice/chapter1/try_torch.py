import torch
import numpy

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





