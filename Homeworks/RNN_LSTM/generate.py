###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import torch
from torch.autograd import Variable

import data
import model


class ArgsClass:
    pass


args = ArgsClass()

args.data = './data/'  # path of the corpus
args.checkpoint = ''  # checkpoint to use
args.words = 30  # how many words to generate
args.seed = 1000  # random seed number
args.cuda = False  # usage of CUDA
args.log_interval = 200  # interval for printing

args.temperature = 1.35  # temperature size
args.outf = './generated/generated_t' + str(args.temperature) + '.txt'  # generated text output file

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    raise ValueError("temperature has to be greater or equal 1e-3")

# Load checkpoint
args.checkpoint = './output/model.pt'
if args.checkpoint != '':
    if args.cuda:
        model = torch.load(args.checkpoint)
    else:
        # Load GPU model on CPU
        model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
model.eval()

print(model)

# loading corpus data and pre-trained model
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
# model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
# model.load_state_dict(torch.load('model.pt', map_location=lambda storage, loc: storage))
# model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()


# function to wrap a single word in a tensor with its corresponding corpus index
def word2var(word, corpus):
    if word not in corpus.dictionary.word2idx:
        raise ValueError('Word is not in dictionary! cannot create file')
    token = corpus.dictionary.word2idx[word]
    out = Variable(torch.ones(1, 1).mul(token).long(), volatile=True)
    if args.cuda:
        out.data = out.data.cuda()
    return out


# input line
input_line = "Buy low, sell high is the"
input_words = input_line.replace(',', '').lower().split()

# Generating
with open(args.outf, 'w') as outf:
    hidden = model.init_hidden(1)
    # pre-feeding the model with the line, word by word
    for word in input_words:
        input = word2var(word, corpus)
        output, hidden = model(input, hidden)
        outf.write(word + ' ')

    # continue generating with hidden state
    for i in range(args.words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        word = '<eos> \n' if word == "<eos>" else word

        outf.write(word + ' ')
        print(word)