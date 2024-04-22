# Elizabeth
# Training data: historical game data
# Input: Current chess board data
# Output: The next chess move in pgn format

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import chess
import chess.pgn

def encode_board(board):
 
    encoded = np.zeros((8, 8, 12), dtype=int)
    piece_to_layer = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color = 0 if piece.color == chess.WHITE else 6
            layer = piece_to_layer[piece.piece_type]
            encoded[square // 8][square % 8][color + layer] = 1
    return encoded

def load_data(filename, num_games=1000):
    X, y = [], []
    pgn = open(filename)
    for _ in range(num_games):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            encoded_board = encode_board(board)
            X.append(encoded_board)
            y.append(np.random.randint(0, 4096))  
    return np.array(X), np.array(y)

def build_model():
    model = Sequential([
        Input(shape=(8, 8, 12)),
        Conv2D(32, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y = load_data('games.pgn')
    y = tf.keras.utils.to_categorical(y, num_classes=4096) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    model.save('chess_model.h5')

if __name__ == "__main__":
    main()
