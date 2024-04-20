import torch
from torch import nn
import torch.nn.functional as F

import random
def make_move(board):
    valid_moves =[ move.uci() for move in list(board.legal_moves)]
    return random.choice(valid_moves)

class ChessModule(nn.Module):
    def __init__(self, hidden_size) -> None:
        super(ChessModule, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, input):
        x_input = torch.clone(input)
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.activation1(input)
        input = self.conv2(input)
        input = self.bn2(input)
        input = input + x_input
        input = self.activation2(input)
        return input

class ChessCNN(nn.Module):
    def __init__(self, hidden_layers = 64, hidden_size = 200) -> None:
        super(ChessCNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([ChessModule(hidden_size) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, input):
        input = self.input_layer(input)
        input = F.relu(input)

        for i in range(self.hidden_layers):
            input = self.module_list[i](input)

        input = self.output_layer(input)

        return input
