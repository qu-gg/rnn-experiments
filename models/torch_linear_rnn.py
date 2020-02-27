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

        # Inner layer of the network
        self.inner = nn.Linear(inp_size + h_size, 128)

        # Inner layer to the hidden output
        self.in2hidden = nn.Linear(128, h_size)

        # Inner layer to the next char pred
        self.in2out = nn.Linear(128, out_size)

        # Final softmax layer
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # Combine x/hidden into one vector and input to inner layer
        combined = torch.cat((x, hidden), 1)
        inner = torch.relu(self.inner(combined))

        # Get output and next hidden state
        hidden = self.in2hidden(inner)
        output = self.softmax(self.in2out(inner))
        return output, hidden

    def init_hidden(self):
        return torch.zeros([1, self.h_size])


def train(samples, labels):
    """
    Handles a simple training loop through all of the data for a given network
    :param samples: all letters 1:n-1
    :param labels: all letters 2:n
    :return loss of this batch
    """
    # Get new hidden state
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


def get_sample(model):
    """
    Handles getting a sample of text from the network at a given training step, sampling the start
    from a random letter
    :param net: network class
    """
    letter = all_letters[random.randint(0, len(all_letters) - 1)]
    start = stringToMat(letter)[0]
    hidden = model.init_hidden()

    print(letter, end='')

    # For a number of samples, get output and choose random letter within top 3 preds
    for _ in range(150):
        output, hidden = model(start, hidden)
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
    losses = []
    for ep in range(1000):
        templ = 0

        # Train network in batches of 250 characters
        for idx in range(0, len(dataset), 250):
            sample = samples[idx:idx+250]
            label = labels[idx:idx+250]

            # Train network, get loss for sample
            loss = train(sample, label)
            templ += loss
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
    torch.save(net, "simpleRNNModel.pt")