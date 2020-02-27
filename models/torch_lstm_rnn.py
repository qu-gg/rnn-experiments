"""
@file torch_lstm_rnn.py
@author Ryan Missel

Simple linear RNN model in PyTorch to try and predict the next letter in a sequence
given a text file
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils.parameters import all_letters, check_gpu
import random


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

        # LSTM w/ Dropout layers
        self.lstm = nn.LSTM(inp_size, h_size, num_layers=3, dropout=0.2)

        # Inner layer to the next char pred
        self.in2out = nn.Linear(h_size, out_size)

        # Final softmax layer
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Go through LSTM layers
        inner, _ = self.lstm(x)

        # Get outpu and next hidden state
        inner = inner.view(x.shape[0], -1)
        output = self.softmax(self.in2out(inner))
        return output


def train(samples, labels):
    """
    Handles a simple training loop through all of the data for a given network
    :param samples: all letters 1:n-1
    :param labels: all letters 2:n
    :return loss of this batch
    """
    # Set gradient to 0 for new sample
    optim.zero_grad()

    # Predict next chars given the sample sequence
    outputs = net(samples)

    # Get loss and update weights
    loss = criterion(outputs, torch.max(labels, 2)[1].squeeze_())
    loss.backward()
    optim.step()
    return loss


def get_sample(model):
    """
    Handles getting a sample of text from the network at a given training step, sampling the start
    from a random letter
    :param net: network class
    """
    letter = all_letters[random.randint(0, len(all_letters) - 1)]
    start = stringToMat(letter)[0]
    print(letter, end='')

    # For a number of samples, get output and choose random letter within top 3 preds
    for _ in range(150):
        output = model(start.view(1, 1, -1))
        output = output.detach().numpy()[0]

        prob = np.exp(output) / np.sum(np.exp(output))
        letter = all_letters[np.random.choice(range(len(all_letters)), p=prob.ravel())]

        start = stringToMat(letter)[0]
        print(letter, end='')
    print("")


if __name__ == '__main__':
    # Check for GPU to use
    check_gpu()

    # Open dataset to use
    path = input("Enter data path: ")
    try:
        data = open(path, 'r').read().replace('\n', ' ')
    except UnicodeDecodeError:
        data = open(path, 'r', encoding='utf8').read().replace('\n', ' ')

    # Initialize RNN and optimizers
    net = SimpleRNN(len(all_letters), 512, len(all_letters))
    optim = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Make dataset, loop through it a number of times
    dataset = stringToMat(data)
    print(dataset.shape)
    samples = dataset[:-1]
    labels = dataset[1:]

    # Loop over all data for x epochs
    batch_size = 100
    losses = []
    for ep in range(1000):
        templ = 0

        # Train network in batches of characters
        for idx in range(0, len(dataset), batch_size):
            sample = samples[idx: idx+batch_size]
            label = labels[idx: idx+batch_size]

            # Train network, get loss for sample
            loss = train(sample, label)
            templ += loss

            # Print loss at epoch, test a sample
            if idx % 100000 == 0:
                print("Loss at epoch {}, idx {}: {}".format(ep, idx, templ / (idx / batch_size)))
                get_sample(net)

        losses.append(templ / len(dataset))

        # Print loss at epoch, test a sample
        print("Loss at epoch {}: {}".format(ep, templ / (len(dataset) / 250)))
        get_sample(net)

        # Save model's params
        if ep % 50 == 0:
            print("Saving model...", end='')
            torch.save(net, "ckpts/simpleRNNModel{}.pt".format(ep))
            print("...saved!")

    # Plot losses over epochs
    plt.plot(losses)
    plt.show()

    # Save model's params
    torch.save(net, "simpleRNNModelFinal.pt")