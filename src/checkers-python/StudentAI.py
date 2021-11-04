from random import randint
from BoardClasses import Move
from BoardClasses import Board
import copy

"""
How to run the code locally:
From <project_dir>/Tools
python3 AI_Runner.py 7 7 2 l Sample_AIs/Random_AI/main.py ../src/checkers-python/main.py
"""

# Describes a node in a search tree.
class GameStateNode:
    def __init__(self, parent, board, color):
        self.board = board
        self.opponent = {1: 2, 2: 1}
        self.color = color

        self.parent = parent
        self.children = []

    # Expand the node by filling the children with a list of all boards for all moves that result from this board
    def expand(self):
        moves = self.board.get_all_possible_moves(self.color)

        # Iterate over all moves in the possible moves
        for checker_moves in moves:
            # Iterate over each move that this checker can make
            for move in checker_moves:
                # Get a copy of the board that results from this move
                resulting_board = self.resulting_board(move, self.color)

                # Create a tree node to hold the new board
                child = GameStateNode(self, resulting_board, self.opponent[self.color])
                self.children.append(child)

    def resulting_board(self, move, turn):
        new_board = copy.deepcopy(self.board)
        new_board.make_move(move, turn)
        return new_board

    # Return a heuristic value for the board in this node.
    # Smaller values are good for player 1
    # Larger values are good for player 2
    def heuristic(self):
        return self.board.white_count - self.board.black_count

    # Determine if this board is in the win state for a given player
    def is_win(self):
        return self.board.is_win(self.color)

# StudentAI class

# The following part should be completed by students.
# Students can modify anything except the class name and existing functions and variables.
class StudentAI:

    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        # Why does this start as an empty string then get set to a number later?
        self.color = ''
        self.opponent = {1: 2, 2: 1}
        self.color = 2

    # Get the next move that the AI wants to make
    # The move passed in is the move that the opponent just made,
    # or an invalid move if we get to move first
    def get_move(self, move):
        # If there are no elements in the move passed in, then do... something
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
        else:
            self.color = 1

        # Get all possible moves. Each element in the list is itself a list of all the moves
        # that one particular checker on the board can make
        moves = self.board.get_all_possible_moves(self.color)

        # Check if the possible moves exist
        if len(moves) <= 0:
            raise RuntimeError("StudentAI: tried to get a move, but no possible moves could be found. "
                               "Are you sure the game hasn't already ended?")

        # A list of moves with the best heuristic
        bestMoves = []
        bestHeuristic = 1000000000

        # Go through all moves in the list of all possible moves
        for checkers_moves in moves:
            for move in checkers_moves:
                # get the heuristic of the current move
                currentHeuristic = self.move_heuristic(move)

                # If the current move is better than the best so far,
                # clear out the list and add this move to it
                if currentHeuristic < bestHeuristic:
                    bestMoves.clear()
                    bestMoves.append(move)
                    bestHeuristic = currentHeuristic

                # If the current move has the same heuristic as the best so far,
                # add this move to the list of best moves
                elif currentHeuristic == bestHeuristic:
                    bestMoves.append(move)

        # If there was only one move then set it to that move
        if len(bestMoves) == 1:
            move = bestMoves[0]
        # If multiple moves had the same heuristic use the tiebreaker to choose one from the list
        else:
            move = self.heuristic_tiebreaker(bestMoves)

        # Make a new move using the randomly selected element of the randomly selected move
        # This modifies our copy of the board so that it matches the game's copy
        self.board.make_move(move, self.color)
        return move

    # Get the heuristic value of a move
    # The SMALLER the heuristic value, the BETTER the move
    def move_heuristic(self, move):
        heuristic = 0
        for index in range(len(move) - 1):
            # Decrease heuristic by horizontal distance between this and next move
            heuristic -= abs(move[index][0] - move[index + 1][0])
            # Decrease heuristic by vertical distance between this and next move
            heuristic -= abs(move[index][1] - move[index + 1][1])
        return heuristic

    # Given a list of moves with the same heuristic,
    # use some tie breaking method to choose one of the moves
    def heuristic_tiebreaker(self, moves):
        return moves[randint(0, len(moves) - 1)]

