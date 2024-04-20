import torch
import preprocess
from random import randint
from AI import *

MIN_ELO = 2000
MAX_TRAINING_GAMES = 10
games = preprocess.parse_pgn('libchess_db.pgn', MIN_ELO, MAX_TRAINING_GAMES)

model = ChessCNN()
model.load_state_dict(torch.load('chess_model.pt'))
model.eval()

with torch.no_grad():
    game = games[randint(0, len(games) - 1)]
    game_state = game
    for _ in range(randint(1, (len([i for i in game.mainline_moves()]) -1))):
        game_state = game_state.next() # type: ignore
    board = game_state.board() # type: ignore
    print(board)
    print("\n")
    board.push(make_move(model, board)) # type: ignore
    print(board)
