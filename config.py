import os
import sys
import time
import math
import torch
import numpy as np


######### Hyperparameters #########
base_sched_iter = [1e-1, 1e-2] # LR Schedule
base_epochs_iter = [40, 20] # Number of epochs to train for
wd_base = 1e-4

######### Layer Configurations #########
cfg_dict = {'cfg_10': [64, (64, 2), 128, (128, 2), 256, (256, 2), 512, (512, 2), 512, 512],
            'cfg_20': [64, 64, 64, (64, 2), 128, 128, 128, (128, 2), 256, 256, 256, (256, 2), 256, 256, 256, (256, 2), 512, 512, 512, 512],
            'cfg_40': [64, 64, 64, 64, 64, 64, 64, (64, 2), 128, 128, 128, 128, 128, 128, 128, (128, 2), 256, 256, 256, 256, 256, 256, 256, (256, 2), 512, 512, 512, 512, 512, 512, 512, (512, 2), 512, 512, 512, 512, 512, 512, 512, 512]}

cfg_uniform = lambda n_layers: [64]*n_layers # Used in GroupNorm vs. rank experiments


######### LR Scheduler #########
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        lr_sched = np.array([base_lr] * int(decay_iter * 0.75) + [final_lr] * int(decay_iter * 0.25))        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, lr_sched))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
        
    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr


######### Progress bar #########
term_width = 150 
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
