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
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

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

criterion = nn.CrossEntropyLoss()
if args.cuda:
    criterion.cuda()

print('------------------------------------------------------')
print('\t\t Total parameters in model : ', sum(param.numel() for param in model.parameters()))
print('------------------------------------------------------\n')


###############################################################################
# Training code
###############################################################################

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
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    mean_loss_train_curEp = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    print('train ntokens=', ntokens)
    hidden = model.init_hidden(args.batch_size)
    train_seqs_vec = range(0, train_data.size(0) - 1, args.bptt)
    for batch, i in enumerate(train_seqs_vec):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
        mean_loss_train_curEp += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    mean_loss_train_curEp /= len(train_seqs_vec)
    return mean_loss_train_curEp


# Loop over epochs.
lr = args.lr
best_val_loss = None

Loss_train_Graph_Ep, Prpx_train_Graph_Ep = [], []  # for graphs by epochs
Loss_val_Graph_Ep, Prpx_val_Graph_Ep = [], []
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_mean_loss_Ep = train()
        val_loss = evaluate(val_data)

        Loss_train_Graph_Ep.append(train_mean_loss_Ep)
        Prpx_train_Graph_Ep.append(math.exp(train_mean_loss_Ep))
        Loss_val_Graph_Ep.append(val_loss)
        Prpx_val_Graph_Ep.append(math.exp(val_loss))
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# Graphs
epochs_vec = range(1, args.epochs + 1, 1)

# --------------------- Loss Graph --------------------- #
fig1, ax1 = plt.subplots()
# by epochs
ax1.plot(epochs_vec, Loss_train_Graph_Ep, '-b', marker='+', markersize=7, linewidth=3, label='Train Loss')
ax1.plot(epochs_vec, Loss_val_Graph_Ep, '-r', marker='+', markersize=7, linewidth=3, label='Validation Loss')
ax1.set_xlabel('# Epoch', fontsize=14)
ax1.xaxis.set_ticks(np.arange(0, 21, step=2.0))
ax1.set_title('Loss vs. Epoch', fontsize=17)
ax1.set_ylabel('Loss', fontsize=14)
ax1.grid()
ax1.legend()
fig1.savefig("Loss_graph.png")
plt.show()

# ------------------ Perplexity Graph ------------------ #
fig2, ax2 = plt.subplots()
# by epochs
ax2.plot(epochs_vec, Prpx_train_Graph_Ep, '-b', marker='+', markersize=7, linewidth=3,
         label='Train Perplexity')  # graph by epochs
ax2.plot(epochs_vec, Prpx_val_Graph_Ep, '-r', marker='+', markersize=7, linewidth=3,
         label='Validation Perplexity')  # by right steps
ax2.set_xlabel('# Epoch', fontsize=14)
ax2.xaxis.set_ticks(np.arange(0, 21, step=2.0))
ax2.set_title('Perplexity vs. Epoch', fontsize=17)
ax2.set_ylabel('Perplexity', fontsize=14)
ax2.grid()
ax2.legend()
# plt.ylim(0, 1000)
fig2.savefig("Prpx_graph.png")
plt.show()
