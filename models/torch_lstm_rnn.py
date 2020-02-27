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


class lstmRNN(nn.Module):
    def __init__(self, inp_size, h_size, out_size):
        super(lstmRNN, self).__init__()
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

    def init_hidden(self):
        return torch.zeros([1, self.h_size])


