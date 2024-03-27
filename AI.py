import random
def make_move(board):
    valid_moves =[ move.uci() for move in list(board.legal_moves)]
    return random.choice(valid_moves)