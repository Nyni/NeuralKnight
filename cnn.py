# Elizabeth
# Training data: historical game data
# Input: Current chess board data
# Output: The next chess move in pgn format

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import chess

def board_to_input(board):
    labels = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
              'P': -1, 'N': -2, 'B': -3, 'R': -4, 'Q': -5, 'K': -6}
    board_matrix = np.zeros((8, 8, 6))
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            index = labels[piece.symbol()]
            layer_index = abs(index) - 1
            board_matrix[i // 8][i % 8][layer_index] = np.sign(index)
    return board_matrix.reshape((8, 8, 6))

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 6)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(4096, activation='softmax')  # Output layer for 64 squares * 64 squares moves
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, epochs=5):
    model.fit(X_train, y_train, epochs=epochs)


def predict_move(board, model):
    input_tensor = np.expand_dims(board_to_input(board), axis=0)
    predictions = model.predict(input_tensor)
    move_index = np.argmax(predictions)
  
    return index_to_uci(move_index, board)

def index_to_uci(index, board):
    from_square = chess.SQUARE_NAMES[index // 64]
    to_square = chess.SQUARE_NAMES[index % 64]
    return from_square + to_square

# Example usage
model = create_model()
# X_train, y_train = load_data()  # Load your training data here
# train_model(model, X_train, y_train)

# Later in your game loop
# if it's AI's turn:
#     move = predict_move(board, model)
#     if move in [move.uci() for move in board.legal_moves]:
#         board.push_uci(move)
