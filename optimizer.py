import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import torch

# class NoamLR(_LRScheduler):
#     def __init__(self, optimizer, step_size, d_model, n_warmup_steps=2500):
#         self.step_size = step_size
#         self.d_model = d_model
#         self.n_warmup_steps = n_warmup_steps
#         self.step_count = 0
#         super(NoamLR, self).__init__(optimizer)
#
#     def step(self):
#         self.step_count += 1
#         new_lr = np.power(self.d_model, -0.5) * np.min([
#             np.power(self.step_count, -0.5),
#             np.power(self.n_warmup_steps, -1.5) * self.step_count])
#
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = new_lr



class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


