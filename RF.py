import chess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler


def parse_chess_data(file_path):
    def calculate_piece_values(board_state):
        piece_values = {
            1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0,  # White pieces
            7: 1, 8: 3, 9: 3, 10: 5, 11: 9, 12: 0  # Black pieces
        }
        white_value, black_value = 0, 0
        for piece in board_state:
            if 1 <= piece <= 6:  # White pieces
                white_value += piece_values[piece]
            elif 7 <= piece <= 12:  # Black pieces
                black_value += piece_values[piece]
        material_diff = white_value - black_value
        return white_value, black_value, material_diff

    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n')

    data = []
    current_game_moves = []
    i = 0

    while i < len(lines):
        if lines[i].startswith('Result'):
            try:
                result = float(lines[i].split()[-1])
                valid_game = True
            except ValueError:
                print(f"Invalid result format at line {i}, skipping game.")
                valid_game = False

            if valid_game and all(len(move) == 64 for move in current_game_moves):
                for move in current_game_moves:
                    white_value, black_value, material_diff = calculate_piece_values(move)
                    move.extend([result, white_value, black_value, material_diff])
                data.extend(current_game_moves)
            else:
                print(f"Skipping game due to invalid data before line {i}.")
            current_game_moves = []
            i += 1  # Move to the next line after result
        else:
            move = []
            try:
                for _ in range(8):
                    row_data = list(map(int, lines[i].strip().split()))
                    if len(row_data) != 8:
                        raise ValueError("Incorrect number of integers in row")
                    move.extend(row_data)
                    i += 1
            except (ValueError, IndexError):
                print(f"Invalid or incomplete data starting at line {i}, skipping to next game.")
                current_game_moves = []
                i = skip_to_next_result(lines, i)  # Skip to next 'Result' line
                continue

            if len(move) == 64:
                current_game_moves.append(move)

    columns = [f'position{i+1}' for i in range(64)] + ['Result', 'White_Material', 'Black_Material', 'Material_Diff']
    df = pd.DataFrame(data, columns=columns)
    return df

def skip_to_next_result(lines, current_index):
    """Moves the index to the next 'Result' line or to the end of the list."""
    while current_index < len(lines) and not lines[current_index].startswith('Result'):
        current_index += 1
    return current_index

#testing validity of feature structure
df = parse_chess_data('newChess_results.txt')
print(df.head(90))

# drop dataframe down to material features
reduced_features = df[['White_Material', 'Black_Material', 'Material_Diff']]
target = df['Result'].astype(int)  # Ensure the Result column is of integer type

# Initialize the scaler
scaler = MinMaxScaler()

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(reduced_features, target, test_size=0.2, random_state=42)

# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Test accuracy
#predictions = model.predict(X_test_scaled)
#print("Accuracy:", accuracy_score(y_test, predictions))
#print("Classification Report:\n", classification_report(y_test, predictions))




#############
def encode_board(board):
    piece_to_number = {
        'p': 7, 'r': 10, 'n': 8, 'b': 9, 'q': 11, 'k': 12,
        'P': 1, 'R': 4, 'N': 2, 'B': 3, 'Q': 5, 'K': 6
    }
    features = []
    for rank in reversed(range(8)):
        for file in range(8):
            square_index = chess.square(file, rank)
            piece = board.piece_at(square_index)
            number = piece_to_number.get(piece.symbol(), 0) if piece else 0
            features.append(number)
    return features

def calculate_material_values(features):
    piece_values = {
        1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0,  # White pieces
        7: 1, 8: 3, 9: 3, 10: 5, 11: 9, 12: 0  # Black pieces
    }
    white_value, black_value = 0, 0
    for index, piece in enumerate(features):
        value = piece_values.get(piece, 0)
        if piece <= 6 and piece != 0:
            white_value += value
        elif piece > 6:
            black_value += value
    material_diff = white_value - black_value
    return [white_value, black_value, material_diff]

def predict_win_probability(board):
    features = encode_board(board)
    material_features = calculate_material_values(features)
    scaled_features = scaler.transform([material_features])
    probability = model.predict_proba(scaled_features)[0][1]  # Index 1 is for White winning
    return probability