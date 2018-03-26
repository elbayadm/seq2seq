from collections import UserDict
import torch
from torch.autograd import Variable


def to_var(input_, requires_grad=False, volatile=False):
    """
    Returns a torch Variable on GPU.
    """
    if isinstance(input_, (UserDict, dict)):
        for key in input_:
            v = Variable(input_[key],
                         requires_grad=requires_grad,
                         volatile=volatile)
            input_[key] = v.cuda()
    else:
        input_ = Variable(input_,
                          requires_grad=requires_grad,
                          volatile=volatile).cuda()
    return input_



