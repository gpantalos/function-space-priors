"""
From https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4

Imagine that we have a data set of all even numbers between 0 and 128. 
The generator is going to take in random noise as an integer 
in that same range and learn to produce only even numbers.

# todo: check https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
for training tips
""" 

import torch 
import math 
import numpy as np 
from tqdm import trange 
from IPython import embed 
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float)


def generate_true_data(max_int, batch_size=16):
    # Get the number of binary places needed to represent the maximum number
    max_length = math.log(max_int, 2)

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, max_int // 2, batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1.] * batch_size

    # Generate a list of binary numbers for training.
    def create_binary_list_from_int(number):
        return [int(x) for x in bin(number)[2:]]
    data = [create_binary_list_from_int(2 * x) for x in sampled_integers]
    data = [([0.] * int(max_length - len(x))) + x for x in data]

    # Tensors
    labels = torch.tensor(labels).reshape(-1, 1)
    data = torch.tensor(data)
    return labels, data


class Generator(torch.nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.linear = torch.nn.Linear(input_length, input_length)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))


class Discriminator(torch.nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.linear = torch.nn.Linear(input_length, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))


def train(max_int=128, batch_size=16, training_steps=500):
    input_length = int(math.log(max_int, 2))

    # Models
    g = Generator(input_length)
    d = Discriminator(input_length)

    # Optimizers
    g_optimizer = torch.optim.Adam(g.parameters(), lr=0.001)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=0.001)

    # loss
    loss = torch.nn.BCELoss()
    g_losses = []
    d_losses = []

    for i in trange(training_steps):
        g_optimizer.zero_grad()

        # Create noisy input for g
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        g_out = g(noise)

        # Generate examples of real data
        true_labels, true_data = generate_true_data(max_int, batch_size=batch_size)

        # Train g. We invert the labels here and 
        # don't train d because we want g
        # to make things d classifies as true.
        g_d_out = d(g_out)
        g_loss = loss(g_d_out, true_labels)
        g_loss.backward()
        g_optimizer.step()

        # Train d on the true/generated data
        d_optimizer.zero_grad()
        true_d_out = d(true_data)
        true_d_loss = loss(true_d_out, true_labels)

        # .detach() here: we are not training g we are just focused on d
        g_d_out = d(g_out.detach())
        g_d_loss = loss(g_d_out, torch.zeros(batch_size, 1))
        d_loss = (true_d_loss + g_d_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    plt.plot(g_losses, label='g loss')
    plt.plot(d_losses, label='d loss')
    plt.legend()
    plt.show()

train()