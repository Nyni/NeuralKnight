import chess
import chess.pgn
import pygame
from RF import predict_win_probability
import AI






pygame.init()

width = 480  # Width of the window
height = 600  # Height of the Window matches the width + an extra row for score
boxSize = width // 8 # 1/8 of the window widtch.
game = chess.pgn.Game()
board = chess.Board()
san_moves = []
current_probability = predict_win_probability(board)
#print(board)
surface = pygame.display.set_mode((width, height))


#Update the probability display after each move
#current_probability = predict_win_probability(board)
#print(f"Probability of White winning: {current_probability:.2%}")
#############################################################


#ADDED FUNCTION- encodes board for predict_win_probability function
def encode_and_print_board(board):
    piece_to_number = {
        'p': 7, 'r': 10, 'n': 8, 'b': 9, 'q': 11, 'k': 12,
        'P': 1, 'R': 4, 'N': 2, 'B': 3, 'Q': 5, 'K': 6
    }
    rows = []
    for rank in reversed(range(8)):  # Start from '8' down to '1'
        encoded_row = []
        for file in range(8):  # Start from 'a' to 'h'
            square_index = chess.square(file, rank)
            piece = board.piece_at(square_index)
            number = piece_to_number.get(piece.symbol(), 0) if piece else 0
            encoded_row.append(number)
        rows.append(' '.join(f'{num:2}' for num in encoded_row))
    for row in rows:
        print(row)
    print()  # Add an extra newline for separation between board states



playing = True

def Load_Pieces():
    pieces = {}
    pieces['P'] =pygame.image.load('Images/WP.png')
    pieces['R'] = pygame.image.load('Images/WR.png')
    pieces['N'] = pygame.image.load('Images/WN.png')
    pieces['B'] = pygame.image.load('Images/WB.png')
    pieces['K'] = pygame.image.load('Images/WK.png')
    pieces['Q'] = pygame.image.load('Images/WQ.png')

    pieces['p'] =pygame.image.load('Images/BP.png')
    pieces['r'] = pygame.image.load('Images/BR.png')
    pieces['n'] = pygame.image.load('Images/BN.png')
    pieces['b'] = pygame.image.load('Images/BB.png')
    pieces['k'] = pygame.image.load('Images/BK.png')
    pieces['q'] = pygame.image.load('Images/BQ.png')

    return pieces

def Draw(move_list,begin_square,board,prob):
    surface.fill((0,0,0))

    Draw_Board()
    Draw_Current_Selection(begin_square)
    Draw_Legal_Moves(move_list)
    DrawPieces()
    Draw_Status(board,prob)
def Draw_Board():
    colors = [(210,180,140), (100, 50, 35)]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(surface, color, (col * boxSize, row * boxSize, boxSize, boxSize))

def Draw_Status(board,curr_prob):

    text = ""
    myfont = pygame.font.SysFont("monospace", 34,bold=True)
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            text = "Checkmate! Black Wins"
        else:
            text = "Checkmate! White Wins"
    elif board.is_stalemate():
        text = "Stalemate"
    elif board.turn == chess.WHITE:
        text = "White to play"
        if board.is_check():
            text += " Check!"
    elif board.turn == chess.BLACK:
        text = "Black to play"
        if board.is_check():
            text += " Check!"

    rounded_prob = round(curr_prob, 4)
    text2 = "White to win: {:.4%}".format(rounded_prob)
    label = myfont.render(text, 1, (255, 255, 0))
    surface.blit(label, label.get_rect(center=(width / 2, 510)))
    if not board.is_checkmate() and not board.is_stalemate():
        label2 = myfont.render(text2, 1, (255, 255, 0))
        surface.blit(label2, label2.get_rect(center=(width / 2, 560)))

def Draw_Current_Selection(begin_square):
    if begin_square != '00':
        x = chess.square_file(chess.parse_square(begin_square)) * boxSize
        y = (7 - chess.square_rank(chess.parse_square(begin_square))) * boxSize
        # Draw a rectangle at the selected square with pale red color
        pygame.draw.rect(surface, (255, 200, 200), (x, y, boxSize, boxSize))
def Draw_Legal_Moves(move_list):
    for move in move_list:

        if(len(move)==5):
            dest_square = move[2:4]
        else:
            dest_square = move[2:]
        x = chess.square_file(chess.parse_square(dest_square)) * boxSize
        y = (7 - chess.square_rank(chess.parse_square(dest_square))) * boxSize
        # Draw a rectangle at the selected square with pale green color
        pygame.draw.rect(surface, (200, 255, 200), (x, y, boxSize, boxSize))
def DrawPieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            x = chess.square_file(square) * boxSize
            y = (7 - chess.square_rank(square)) * boxSize
            surface.blit(piece_images[piece.symbol()], (x, y))

piece_images = Load_Pieces()
rowNames={0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
colNames={0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}

begin = '00'
dest = '00'
move_list = []
turn = 1
time_to_play=300
while playing:
    for event in pygame.event.get():
        # quit game
        if event.type == pygame.QUIT:
            playing = False
        if turn==1:

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check position of the mouse
                x, y = event.pos
                x = x // boxSize
                y = y // boxSize
                # If the mouse is in the scorebox, do nothing
                if y == 8:
                    pass
                else:
                    if begin == '00':
                        begin = str(colNames[x] + str(rowNames[y]))
                        #print(begin)
                        move_list = [move.uci() for move in list(board.legal_moves) if
                                     move.from_square == chess.parse_square(begin)]
                        #print([move for move in move_list])
                    else:
                        dest = str(colNames[x] + str(rowNames[y]))
                        move_to_make = str(begin + dest)
                        if move_to_make in [move.uci() for move in list(board.legal_moves)]:
                            new_move = chess.Move.from_uci(move_to_make)
                            san_moves.append(board.san(new_move))
                            board.push_uci(move_to_make)
                            current_probability = predict_win_probability(board)

                            #print(san_moves)
                            turn = 2
                        elif move_to_make + "q" in [move.uci() for move in list(board.legal_moves)]:
                            move_to_make = move_to_make + "q"

                            new_move = chess.Move.from_uci(move_to_make)
                            san_moves.append(board.san(new_move))

                            board.push_uci(move_to_make)
                            current_probability = predict_win_probability(board)

                            #print(san_moves)
                            turn = 2
                        begin = '00'
                        dest = '00'
                        move_list = []

        elif turn == 2:


            move_to_make= AI.make_move(board,san_moves)
            san_moves.append(board.san(move_to_make))
            board.push(move_to_make)
            #new_move = chess.Move.from_uci(move_to_make)
            #san_moves.append(board.san(new_move))
            #board.push_uci(move_to_make)
            turn = 1
            current_probability = predict_win_probability(board)


        else:
            if event.type == pygame.MOUSEBUTTONDOWN:
                board.reset()
                san_moves=[]
                turn = 1
                current_probability = predict_win_probability(board)
        if board.is_checkmate() or board.is_stalemate():
            turn = 3




    Draw(move_list,begin,board,current_probability)
    pygame.display.flip()