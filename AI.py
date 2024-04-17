import random

import chess

def create_san_string(san_list):
    string = ' '.join(f"{i // 2 + 1}. {value}" if i % 2 == 0 else value for i, value in enumerate(san_list))
    return string



def uci_to_san(board,uci_move):

    move = chess.Move.from_uci(uci_move)
    return board.san(move)
#
def make_move(board,san_list):
    san_string = create_san_string(san_list)
    print(san_string)
    valid_moves =[ move.uci() for move in list(board.legal_moves)]
    #for move in board.move_stack:
    #    #print(move)
     #  ;print(uci_to_san(board,move))

    return random.choice(valid_moves)


    #with open("in_progress_game.pgn", "w") as pgn_file:
    #    pgn_file.write(str(game))
    #    pgn_file.write("\n\n[Result \"*\"]\n[SetUp \"1\"]\n[FEN \"" + board.fen() + "\"]\n\n*")
    #game.add_variation(chess.pgn.GameNode(board))
