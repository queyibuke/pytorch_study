import torch
import torch.nn as nn

rnn = nn.RNN(input_size=100, hidden_size=10, num_layers=2)

print(rnn._parameters.keys())
print(rnn.weight_hh_l0.shape)
print(rnn.weight_ih_l0.shape)
print(rnn.weight_hh_l1.shape)
print(rnn.weight_ih_l1.shape)