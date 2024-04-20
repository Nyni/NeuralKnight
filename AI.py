from chess import Board, Move
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from preprocess import board_2_np_repr, num_2_letter

import random
def make_move_random(board):
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

def np_move_2_uci(move):
    move = np.where(move == 1)
    from_row, from_col = move[1]
    from_col = num_2_letter[from_col]
    from_row = 8 - from_row

    to_row, to_col = move[2]
    to_col = num_2_letter[to_col]
    to_row = 8 - to_row

    return "".join([from_col, str(from_row), to_col, str(to_row)])

def make_move(ai: ChessCNN, board: Board):
    legal = False
    uci = ""
    while not legal:
        np_board = board_2_np_repr(board)
        prediction = ai(np_board)
        uci = np_move_2_uci(prediction)
        legal = board.is_legal(Move.from_uci(uci))
    
    return uci
