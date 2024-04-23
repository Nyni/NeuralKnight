# pyright: reportOptionalMemberAccess=false
from chess import Board, Move
import numpy as np
import chess.pgn
from torch.utils.data.dataset import Dataset

letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 
                'f': 5, 'g': 6, 'h': 7}
num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 
                5: 'f', 6: 'g', 7: 'h'}

def extract_pgn_movement(pgn_path):
    with open(pgn_path, 'r') as pgn:
        with open('pgn_movements.txt', 'a') as of:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                string = str(game.mainline_moves())
                result = game.headers.get("Result", None)
                if ('{' not in string) and result is not None:
                    of.writelines([string, ' ', result,'\n'])

def filter_elo(pgn, elo, max_match = 5000):
    elos_offset = []
    match_num = 0
    while True:
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)

        if match_num > max_match or headers is None:
            break

        if (int(headers.get("WhiteElo", 0)) > elo) and (int(headers.get("BlackElo", 0)) > elo) and (headers.get("Termination", "?") == "Normal"):
            match_num += 1
            elos_offset.append(offset)

    return elos_offset


def parse_pgn(file, min_elo = 2000, matches = 5000) -> list[chess.pgn.Game]:
    games = []
    with open(file) as pgn:
        elos = filter_elo(pgn, min_elo, matches)
        for offset in elos:
            pgn.seek(offset)
            games.append(chess.pgn.read_game(pgn))

    return games

def move_2_np_repr(move: Move):
    move_uci = move.uci()
    from_output_layer = np.zeros((8, 8), dtype=np.float32)
    from_row = 8 - int(move_uci[1])
    from_col = letter_2_num[move_uci[0]]
    from_output_layer[from_row, from_col] = 1

    to_output_layer = np.zeros((8, 8), dtype=np.float32)
    to_row = 8 - int(move_uci[3])
    to_col = letter_2_num[move_uci[2]]
    to_output_layer[to_row, to_col] = 1

    return np.stack([from_output_layer, to_output_layer])

def board_2_np_repr(board: Board):
    peices = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for p in peices:
        layers.append(create_board_layer(board, p))
    return np.stack(layers, dtype=np.float32)

def create_board_layer(board, peice):
    b = [[]]
    for char in str(board):
        if char == peice:
            b[-1].append(-1)
        elif char == peice.upper():
            b[-1].append(1)
        elif char == '\n':
            b.append([])
        elif char != ' ':
            b[-1].append(0)

    return np.array(b)


class ChessDataset(Dataset):
    def __init__(self, games: list[chess.pgn.Game]) -> None:
        super(ChessDataset, self).__init__()
        self.games  = games
        self.length = len(games)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        random_int = np.random.randint(self.length - 1)
        random_game = self.games[random_int]
        all_moves = [i for i in random_game.mainline_moves()]
        game_state_idx = np.random.randint(len(all_moves) - 1)
        game_state = random_game
        for _ in range(game_state_idx):
            game_state = game_state.next()

        input = board_2_np_repr(game_state.board())
        output = move_2_np_repr(game_state.next().move)
        if game_state_idx % 2 == 1: # if its black turn, multiply by -1
            input *= -1 # so the cnn know which move belong to which side

        return input, output

#extract_pgn_movement('libchess_db.pgn')
