import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

import matplotlib.pyplot as plt


class ArgsClass:
    pass


args = ArgsClass()
args.data = './data/'  # path of the corpus
args.checkpoint = ''  # checkpoint to use
args.model = 'LSTM'  # type of net (RNN_TANH, RNN_RELU, LSTM, GRU)
args.emsize = 215  # word embeddings size
args.nhid = 215  # num. of hidden units per layer
args.nlayers = 2  # num. of layers
args.lr = 20  # initial learning rate
args.clip = 0.25  # gradient clipper
args.epochs = 20  # number of epochs
args.batch_size = 20  # batch size
args.bptt = 20  # length of sequence
args.dropout = 0.2  # dropout size on layers (0 = no dropout)
args.tied = True  # tie word embedding and softmax weights
args.seed = 1000  # random seed number
args.cuda = False  # usage of CUDA
args.log_interval = 200  # interval for printing
args.save = './output/model.pt'  # final model saving path

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


eval_batch_size = 10
test_data = batchify(corpus.test, eval_batch_size)

criterion = nn.CrossEntropyLoss()
if args.cuda:
    criterion.cuda()

# model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

args.checkpoint = './output/model.pt'

# Load checkpoint
if args.checkpoint != '':
    if args.cuda:
        model = torch.load(args.checkpoint)
    else:
        # Load GPU model on CPU
        model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

if args.cuda:
    model.cuda()
else:
    model.cpu()
print(model)

print('------------------------------------------------------')
print('\t\t Total parameters in model : ', sum(param.numel() for param in model.parameters()))
print('------------------------------------------------------\n')


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    return [state.detach() for state in h]


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of testing | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
