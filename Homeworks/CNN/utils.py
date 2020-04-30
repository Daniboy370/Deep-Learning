import pickle
import torch
import math
import time


def to_cuda(x):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        x = x.cuda()
    return x


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def save_checkpoint(model, optimizer, filepath):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


def load_checkpoint(model, optimizer, filepath):
    # "lambda" allows to load the model on cpu in case it is saved on gpu
    state = torch.load(filepath, lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    return model, optimizer


def save_model(model, filepath):
    # Update the saved model file
    torch.save(model.state_dict(), filepath)
