# The AI were built while following the tutorial "How to Build a 2000 ELO Chess AI with Deep Learning" by Moran Reznik at https://www.youtube.com/watch?v=aOwvRvTPQrs

from chess import Board, BLACK
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from preprocess import board_2_np_repr, num_2_letter, letter_2_num

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

def distribute_moves(vals):
    probs = np.array(vals, dtype=np.float64)
    probs = np.exp(probs)
    probs = probs / probs.sum()
    probs = probs ** 3
    probs = probs / probs.sum()
    return probs

def make_move(ai: ChessCNN, board: Board):
    np_board = torch.from_numpy(board_2_np_repr(board))
    if board.turn == BLACK:
        np_board *= -1
    np_board = np_board.unsqueeze(0)

    move = ai(np_board)[0]
    legal_moves = [i for i in board.legal_moves]
    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))
    for fro in froms:
        val = move[0,:,:][8 - int(fro[1]), letter_2_num[fro[0]]]
        vals.append(val)

    probs = distribute_moves(vals)
    choosen = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

    vals = []
    for legal_move in legal_moves:
        fro = str(legal_move)[:2]
        if fro == choosen:
            to = str(legal_move)[2:]
            val = move[1,:,:][8 - int(to[1]), letter_2_num[to[0]]]
            vals.append(val)
        else:
            vals.append(0)

    choosen = legal_moves[np.argmax(vals)]

    return choosen
