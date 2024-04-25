import chess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import pickle



def parse_chess_data(file_path):
    def calculate_piece_values(board_state):
        piece_values = {
            1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0,  # White pieces
            7: 1, 8: 3, 9: 3, 10: 5, 11: 9, 12: 0  # Black pieces
        }
        white_value, black_value = 0, 0
        for piece in board_state:
            if piece <= 6 and piece > 0:  # White pieces
                white_value += piece_values[piece]
            elif piece > 6:  # Black pieces
                black_value += piece_values[piece]
        material_diff = white_value - black_value  # Calculate differential material value
        return white_value, black_value, material_diff

    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n')

    data = []  # to store all moves and their  result
    current_game_moves = []  # temporarily store moves for the current game

    i = 0  # Initialize a counter for lines
    while i < len(lines):
        if lines[i].startswith('Result'):
            result = float(lines[i].split()[-1])
            for move in current_game_moves:
                white_value, black_value, material_diff = calculate_piece_values(move[:64])
                move.extend([result, white_value, black_value, material_diff])
            data.extend(current_game_moves)
            current_game_moves = []
            i += 1
        else:
            if i + 8 > len(lines) or "Result" in lines[i+7]:
                print(f"Warning: Incomplete board state starting at line {i}.")
                break
            move = []
            for _ in range(8):
                if lines[i].startswith('Result'):
                    print(f"Error processing line {i}: {lines[i]}")
                    i += 1
                    break
                move.extend(list(map(int, lines[i].strip().split())))
                i += 1
            if len(move) == 64:
                current_game_moves.append(move)

    # Include the differential material value in the column names
    columns = [f'position{i+1}' for i in range(64)] + ['Result', 'White_Material', 'Black_Material', 'Material_Diff']
    df = pd.DataFrame(data, columns=columns)
    return df

#testing validity of feature structure
df = parse_chess_data('chess_results.txt')
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

