"""
@file run.py
@author Ryan Missel

Handles taking in a dataset and model and running the training/output on it
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from utils.parameters import check_gpu, all_letters

# Models to swap between
# from models.torch_linear_rnn import SimpleRNN
from models.torch_lstm_rnn import lstmRNN


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
    """ Loops through the set of strings given and converts all to Pytorch Tensor """
    lst = []
    for string in set:
        lst.append(stringToMat(string))
    return lst


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


def train(samples, labels):
    """
    Handles a simple training loop through all of the data for a given network
    :param samples: all letters 1:n-1
    :param labels: all letters 2:n
    :return loss of this batch
    """
    # Set gradient to 0 for new sample
    optim.zero_grad()

    # Get outputs of model on given sequence
    outputs = net(samples)

    # Get loss and update weights
    loss = criterion(outputs, torch.max(labels, 2)[1].squeeze_())
    loss.backward()
    optim.step()
    return loss


if __name__ == '__main__':
    # Check for GPU to use
    check_gpu()

    # Open dataset to use
    path = input("Enter data path: ")
    try:
        data = open(path, 'r').read().split('\n')
    except UnicodeDecodeError:
        data = open(path, 'r', encoding='utf8').read().split('\n')

    data = [x for x in data if x != '']

    # Initialize RNN
    load = input("Load a model? (y/n): ")
    if load == "y":
        file = input("Which model to load?: ")
        net = torch.load(file)
        net.eval()
    else:
        model = lstmRNN
        net = model(len(all_letters), 512, len(all_letters))

    # Optimizer and loss functions
    optim = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Make dataset, loop through it a number of times
    dataset = makeDataset(data)
    samples = dataset[:-1]
    labels = dataset[1:]

    # Loop over all data for x epochs
    batch_size = 100
    losses = []
    for ep in range(1000):
        templ = 0
        random.shuffle(dataset)

        # Train network in batches of characters
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx][:-1]
            label = dataset[idx][1:]

            # Train network, get loss for sample
            loss = train(sample, label)
            templ += loss

            # Print loss at epoch, test a sample
            if idx % (len(dataset) // 4) == 0:
                print("Loss at epoch {}, idx {}: {}".format(ep, idx, templ / (idx + 1)))
                get_sample(net)

        losses.append(templ / len(dataset))

        # Print loss at epoch, test a sample
        print("Loss at epoch {}: {}".format(ep, templ / len(dataset)))
        get_sample(net)

        # Save model's params
        if ep % 5 == 0:
            print("Saving model...", end='')
            torch.save(net, "models/ckpts/simpleRNNModel{}.pt".format(ep))
            print("...saved!")

    # Plot losses over epochs
    plt.plot(losses)
    plt.show()

    # Save model's params
    torch.save(net, "simpleRNNModelFinal.pt")