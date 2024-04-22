# Elizabeth
# Training data: historical game data
# Input: Current chess board data
# Output: The next chess move in pgn format

import numpy as np
import chess
import chess.pgn
import pygame
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

pygame.init()

def encode_board(board):
    """Encode the board state to a 3D array."""
    piece_to_layer = {
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
    }
    board_array = np.zeros((8, 8, 12), dtype=np.int8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row, col = divmod(square, 8)
            layer = piece_to_layer[piece.symbol()]
            board_array[row, col, layer] = 1
    return board_array.flatten()

def load_data(filename, num_samples=1000):
    """Load and process PGN file to training data."""
    X, y = [], []
    pgn = open(filename)
    game_count = 0
    while game_count < num_samples:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            X.append(encode_board(board))
            y.append(np.random.randint(4096))
        game_count += 1
    pgn.close()
    return np.array(X), np.array(y)

def build_model():
    """Build and compile a CNN model."""
    model = Sequential([
        Input(shape=(8, 8, 12)),
        Conv2D(32, kernel_size=3, activation='relu'),
        Conv2D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y = load_data('path_to_your_games.pgn', num_samples=1000)
    y = to_categorical(y, num_classes=4096)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    model.save('chess_model.h5')

if __name__ == '__main__':
    main()
