from random import randint
from BoardClasses import Move
from BoardClasses import Board


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
        # If there are no elements in the move passed in, then
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
        else:
            self.color = 1

        # Get all possible moves
        moves = self.board.get_all_possible_moves(self.color)

        # Choose a random checker move
        index = randint(0, len(moves) - 1)
        # Choose a random element in the list of items in the move
        # (meaning you might randomly not do a double jump even if it is available)
        inner_index = randint(0, len(moves[index]) - 1)
        move = moves[index][inner_index]

        # Make a new move using the randomly selected element of the randomly selected move
        self.board.make_move(move, self.color)
        return move
