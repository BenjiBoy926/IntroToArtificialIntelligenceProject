from random import randint
from BoardClasses import Move
from BoardClasses import Board

# How to run the code locally:
# From <project_dir>/Tools
# python3 AI_Runner.py 7 7 2 Sample_AIs/Random_AI/main.py ../src/checkers-python/main.py

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

        # Get all possible moves
        moves = self.board.get_all_possible_moves(self.color)

        # Check if the possible moves exist
        if len(moves) <= 0:
            raise RuntimeError("StudentAI: tried to get a move, but no possible moves could be found. "
                               "Are you sure the game hasn't already ended?")

        # A list of moves with the best heuristic
        bestMoves = []
        bestHeuristic = 1000000000

        # Go through all moves in the list of all possible moves
        for m in moves:
            # get the heuristic of the current move
            currentHeuristic = self.move_heuristic(m)

            # If the current move is better than the best so far,
            # clear out the list and add this move to it
            if currentHeuristic < bestHeuristic:
                bestMoves.clear()
                bestMoves.append(m)
                bestHeuristic = currentHeuristic

            # If the current move has the same heuristic as the best so far,
            # add this move to the list of best moves
            elif currentHeuristic == bestHeuristic:
                bestMoves.append(m)

        # If there was only one move then set it to that move
        if len(bestMoves) == 1:
            move = bestMoves[0]
        # If multiple moves had the same heuristic use the tiebreaker to choose one from the list
        else:
            move = self.heuristic_tiebreaker(bestMoves)

        # Set the move equal to just the furthest move in the list
        move = move[len(move) - 1]

        # Make a new move using the randomly selected element of the randomly selected move
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
        return 0

    # Given a list of moves with the same heuristic,
    # use some tie breaking method to choose one of the moves
    def heuristic_tiebreaker(self, moves):
        return moves[randint(0, len(moves) - 1)]

