"""
@file simple_rnn.py
@author Ryan Missel

Class for a simple single layer RNN that does character by character predictions
"""
import numpy as np


class SimpleRNN:
    def __init__(self, xsize, hsize, ysize):
        """
        Holds the weight matrices and internal layer for the simple 1-layer RNN
        :param xsize: size of the input
        :param hsize: # nodes in the hidden layer
        :param ysize: size of the output, should match xsize for character matching
        """
        self.weights_hh = np.random.random([hsize, hsize])
        self.weights_xh = np.random.random([xsize, hsize])
        self.weights_hy = np.random.random([hsize, ysize])
        self.h = np.zeros([hsize, 1])

    def step(self, x):
        """ Handles running a step for the network given an input """
        self.h = np.tanh(((self.h.T @ self.weights_hh) + (x @ self.weights_xh))).T
        y = self.weights_hy.T @ self.h
        return y


rnn = SimpleRNN(4, 3, 4)

x = np.array([[0, 1, 0, 0]])
print(rnn.step(x))

x1 = np.array([[1, 0, 0, 0]])
print(rnn.step(x1))
