"""
@file sample_model.py
@author Ryan Missel

File to import a PyTorch model and sample it for generated text
"""
import torch
from models.torch_lstm_rnn import SimpleRNN, stringToMat, letterToVec
from utils.parameters import all_letters


def load_model(PATH):
    """ Loads the torch model """
    model = torch.load(PATH)
    model.eval()
    return model


# path = input("Enter path to model: ")
path = "models/simpleRNNModel.pt"
model = load_model(path)

start = stringToMat("L")[0]
hidden = model.init_hidden()

# output, hidden = model(start, hidden)
# print(all_letters[torch.argmax(output)])

for _ in range(400):
    output, hidden = model(start, hidden)
    letter = all_letters[torch.argmax(output)]

    start = stringToMat(letter)[0]
    print(letter, end='')
