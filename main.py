import chess
import chess.pgn
import pygame

import AI

pygame.init()

width = 480  # Width of the window
height = 540  # Height of the Window matches the width + an extra row for score
boxSize = width // 8 # 1/8 of the window widtch.
game = chess.pgn.Game()
board = chess.Board()
san_moves = []
print(board)
surface = pygame.display.set_mode((width, height))

"""
board.push_san("d4")
board.push_san("e5")
board.push_san("d4e5")
print(board)
"""

legal_moves = list(board.legal_moves)
for move in legal_moves:
    print(move.uci())

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

def Draw(move_list,begin_square,board):
    surface.fill((0,0,0))

    Draw_Board()
    Draw_Current_Selection(begin_square)
    Draw_Legal_Moves(move_list)
    DrawPieces()
    Draw_Status(board)
def Draw_Board():
    colors = [(210,180,140), (100, 50, 35)]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(surface, color, (col * boxSize, row * boxSize, boxSize, boxSize))

def Draw_Status(board):

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

    label = myfont.render(text, 1, (255, 255, 0))
    surface.blit(label, label.get_rect(center=(width / 2, 510)))
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
            """
            if event.type == pygame.MOUSEBUTTONDOWN:
                #check position of the mouse
                x, y = event.pos
                x = x//boxSize
                y = y//boxSize
                # if the mouse is in the scorebox, do nothing
                if y == 8:
                    pass
                else:
                    if begin=='00':
                        begin =str(colNames[x]+str(rowNames[y]))
                        print(begin)
                        move_list = [move.uci() for move in list(board.legal_moves) if move.from_square == chess.parse_square(begin)]
                        print([move for move in move_list])
                    else:
                        dest = str(colNames[x] + str(rowNames[y]))
                        moves = [move.uci() for move in list(board.legal_moves)]
                        move_to_make = str(begin + dest)
                        if move_to_make in moves:
                            board.push_san(move_to_make)
                            turn = 2
                        elif str(move_to_make + "q") in moves:
                            board.push_san(move_to_make + "q")
                            turn = 2
                        begin = '00'
                        dest = '00'
                        move_list=[]



        elif turn == 2:

            #game.add_variation(str(board))
            #board_pgn = board.epd()

            board.push_san(AI.make_move(board))
            turn = 1
            """
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
                        print(begin)
                        move_list = [move.uci() for move in list(board.legal_moves) if
                                     move.from_square == chess.parse_square(begin)]
                        print([move for move in move_list])
                    else:
                        dest = str(colNames[x] + str(rowNames[y]))
                        move_to_make = str(begin + dest)
                        if move_to_make in [move.uci() for move in list(board.legal_moves)]:
                            new_move = chess.Move.from_uci(move_to_make)
                            san_moves.append(board.san(new_move))
                            board.push_uci(move_to_make)
                            print(san_moves)
                            turn = 2
                        elif move_to_make + "q" in [move.uci() for move in list(board.legal_moves)]:
                            move_to_make = move_to_make + "q"

                            new_move = chess.Move.from_uci(move_to_make)
                            san_moves.append(board.san(new_move))

                            board.push_uci(move_to_make)
                            print(san_moves)
                            turn = 2
                        begin = '00'
                        dest = '00'
                        move_list = []

        elif turn == 2:
            move_to_make= AI.make_move(board,san_moves)
            new_move = chess.Move.from_uci(move_to_make)
            san_moves.append(board.san(new_move))
            board.push_uci(move_to_make)
            turn = 1

        else:
            if event.type == pygame.MOUSEBUTTONDOWN:
                board.reset()
                san_moves=[]
                turn = 1
        if board.is_checkmate() or board.is_stalemate():
            turn = 3




    Draw(move_list,begin,board)
    pygame.display.flip()