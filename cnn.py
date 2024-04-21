# Elizabeth
# Training data: historical game data
# Input: Current chess board data
# Output: The next chess move in pgn format
import chess
import chess.pgn
import pygame
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random

# Initialize pygame
pygame.init()

# Game display settings
width = 480
height = 540
boxSize = width // 8
surface = pygame.display.set_mode((width, height))

# Load chess game
board = chess.Board()
print(board)

def load_pieces():
    """Loads piece images for the game."""
    pieces = {}
    for name in ['P', 'R', 'N', 'B', 'K', 'Q']:
        pieces[name] = pygame.image.load(f'Images/W{name}.png')
        pieces[name.lower()] = pygame.image.load(f'Images/B{name}.png')
    return pieces

piece_images = load_pieces()

def board_to_input(board):
    """Encodes the board state to a one-hot format for CNN processing."""
    piece_to_value = {'p': [1,0,0,0,0,0,0,0,0,0,0,0], 'n': [0,1,0,0,0,0,0,0,0,0,0,0], ...}  # Complete for all pieces
    board_matrix = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            board_matrix[square // 8][square % 8] = piece_to_value[piece.symbol()]
        else:
            board_matrix[square // 8][square % 8] = np.zeros(12)
    return board_matrix

def create_model():
    """Creates and compiles the CNN model."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 12)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(4096, activation='softmax')  # Output layer for each possible move
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
model = create_model()
model.summary()  # Remove this line in actual game to avoid console clutter

# Game main loop
playing = True
while playing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing = False

    # Additional game logic goes here
    # Use model.predict on the current board state to decide AI moves

    # Dummy rendering block
    surface.fill((0, 0, 0))  # Clear screen with black
    for i in range(8):
        for j in range(8):
            rect = pygame.Rect(i * boxSize, j * boxSize, boxSize, boxSize)
            pygame.draw.rect(surface, (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0), rect)
            piece = board.piece_at(i + j * 8)
            if piece:
                surface.blit(piece_images[piece.symbol()], rect)

    pygame.display.flip()

# Assume `X_train` and `y_train` are available
# model.fit(X_train, y_train, epochs=10, validation_split=0.1)

pygame.quit()
