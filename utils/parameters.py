"""

"""
import torch
import string


def check_gpu():
    # Use GPU is available
    if torch.cuda.is_available():
        print("Using GPU.")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("Not using GPU.")
        torch.set_default_tensor_type('torch.FloatTensor')


# Array of all ascii letters and punctuation
all_letters = string.ascii_letters + " .,;'"

