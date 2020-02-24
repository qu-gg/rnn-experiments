"""
@file torch_lrnn.py
@author Ryan Missel

Simple linear RNN model in PyTorch to try and predict the next letter in a sequence
given a text file
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import string

# Use GPU is available
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Array of all ascii letters and punctuation
all_letters = string.ascii_letters + " .,;'"

# Open dataset to use
data = open("../data/test.txt", 'r').read().split('\n')


def letterToVec(l):
    """ Converts a letter to a one-hot vector """
    vec = torch.zeros(len(all_letters))
    vec[all_letters.find(l)] = 1
    return vec


def stringToMat(string):
    """ Converts a string into a n x 1 x len(all_letters) matrix """
    lst = torch.zeros([len(string), 1, len(all_letters)])
    for s in range(len(string)):
        lst[s][0] = letterToVec(string[s])

    return lst


def makeDataset(set):
    lst = []
    for string in set:
        lst.append(stringToMat(string))
    return lst


class SimpleRNN(nn.Module):
    def __init__(self, inp_size, h_size, out_size):
        super(SimpleRNN, self).__init__()
        self.h_size = h_size

        self.in2hidden = nn.Linear(inp_size + h_size, h_size)
        self.in2out = nn.Linear(inp_size + h_size, out_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.in2hidden(combined)
        output = self.softmax(self.in2out(combined))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.h_size)


def train(samples, labels):
    """
    Handles a simple training loop through all of the data for a given network
    :param samples: all letters 1:n-1
    :param labels: all letters 2:n
    :return loss of this batch
    """
    # Generate new hidden state for this sample
    hidden = net.init_hidden()

    # Set gradient to 0 for new sample
    optim.zero_grad()

    # Predict next char for each letter in sentence
    outputs = torch.zeros([len(samples), len(all_letters)])
    for letter, label, idx in zip(samples, labels, range(len(samples))):
        output, hidden = net(letter, hidden)
        outputs[idx] = output

    # Get loss and update weights
    loss = criterion(outputs, torch.max(labels, 2)[1].squeeze_())
    loss.backward()
    optim.step()
    return loss


# Initialize RNN and optimizers
net = SimpleRNN(len(all_letters), 128, len(all_letters))
optim = torch.optim.Adam(net.parameters())
criterion = nn.NLLLoss()

# Make dataset, loop through it a number of times
dataset = makeDataset(data)
losses = []
for _ in range(50):
    templ = 0
    for string in dataset:
        samples = string[:-1]
        labels = string[1:]

        # Train network, get loss for sample
        loss = train(samples, labels)
        templ += loss
    losses.append(templ / len(dataset))
    print(templ / len(dataset))


plt.plot(losses)
plt.show()