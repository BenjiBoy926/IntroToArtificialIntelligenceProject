from random import randint
from BoardClasses import Move
from BoardClasses import Board
import functools
import copy

"""
How to run the code locally:
From <project_dir>/Tools
python3 AI_Runner.py 7 7 2 l Sample_AIs/Random_AI/main.py ../src/checkers-python/main.py
"""


# Static functions

# Get the board that results from the given move on the given player's turn
def resulting_board(board, move, player_number):
    new_board = copy.deepcopy(board)
    new_board.make_move(move, player_number)
    return new_board


# Return a heuristic value for the board in this node.
# Smaller values are good for player 1 (black)
# Larger values are good for player 2 (white)
# NOTE: the comments in the board class we're using says that player 1 is white and player 2 is black,
# but the code seems to suggest otherwise.
# Comments: BoardClasses.py @ line 99
# Code that implies otherwise: BoardClasses.py @ line 104-110
def board_heuristic(board):
    return board.white_count - board.black_count


# Given two boards, return the board that would be best for the given player
def better_board(board1, board2, player_num):
    # Player 1 prefers smaller heuristics
    if player_num == 1:
        if board_heuristic(board1) < board_heuristic(board2):
            return board1
        else:
            return board2
    # Player 2 prefers bigger heuristics
    elif player_num == 2:
        if board_heuristic(board1) > board_heuristic(board2):
            return board1
        else:
            return board2
    else:
        raise ValueError("Player number must be either a 1 or a 2")


# Given two states for the same player, return the state that would be best for that player
def better_state(state1, state2):
    if state1.player_number == state2.player_number:
        board = better_board(state1.board, state2.board, state1.player_number)
        if board == state1.board:
            return state1
        else:
            return state2
    else:
        raise ValueError("These states cannot be compared because they do not occur on the same player's turn")


# Get the heuristic value of a move
# The SMALLER the heuristic value, the BETTER the move
def move_heuristic(move):
    heuristic = 0
    for index in range(len(move) - 1):
        # Decrease heuristic by horizontal distance between this and next move
        heuristic -= abs(move[index][0] - move[index + 1][0])
        # Decrease heuristic by vertical distance between this and next move
        heuristic -= abs(move[index][1] - move[index + 1][1])
    return heuristic


# Given a list of moves with the same heuristic,
# use some tie breaking method to choose one of the moves
def heuristic_tiebreaker(moves):
    return moves[randint(0, len(moves) - 1)]


# Describes a state in the game. It has the current board, the current player's turn, and the move (if any)
# that caused this state
class GameState:
    def __init__(self, board, player_number, inciting_move=None):
        self.board = board
        self.opponent = {1: 2, 2: 1}
        self.player_number = player_number
        self.inciting_move = inciting_move

    # Get a list of all the states that can result from all the possible moves from this state
    def get_all_resulting_states(self):
        moves = self.board.get_all_possible_moves(self.player_number)
        resulting_states = []

        # Iterate over all moves in the possible moves
        for checker_moves in moves:
            # Iterate over each move that this checker can make
            for move in checker_moves:
                # Get a copy of the board that results from this move
                new_board = resulting_board(self.board, move, self.player_number)

                # Create a game state with the new board and next opponent's turn
                child = GameState(new_board, self.opponent[self.player_number], move)
                resulting_states.append(child)

        # Return the list of resulting states
        return resulting_states


class GameStateNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []

    # Fill this node's list of children with all resulting states
    def expand(self):
        # Get a list of all states that result from this state
        resulting_states = self.state.get_all_resulting_states()
        self.children = []

        # Add a node for each state to this list of children
        for state in resulting_states:
            self.children.append(GameStateNode(state, self))

    # Use the minimax algorithm to decide which state is the best state to go to
    # for the current player at this node
    # Return the state object that is best for this player
    def minimax_choice(self):
        # If this node has no children, the states to reduce is the list of all states
        # that directly result from this one
        if len(self.children) <= 0:
            states_to_reduce = self.state.get_all_resulting_states()
        # If this node has children, the states to reduce is the list of all states
        # chosen by child nodes of this node
        else:
            states_to_reduce = [child.minimax_choice() for child in self.children]

        # Reduce the list down to the best state in the list for this player
        return functools.reduce(better_state, states_to_reduce)


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
        self.opponent = {1: 2, 2: 1}

        # Start by assuming we are player 2
        self.color = 2

        # Depth is in PLIES, not PLAYS, meaning it's the number of my move - their move pairs
        self.search_depth = 4

    # Get the next move that the AI wants to make
    # The move passed in is the move that the opponent just made,
    # or an invalid move if we get to move first
    def get_move(self, move):
        # If the opponent previously made a move, update our board to express it
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
        # If our opponent did not previously make a move, that means we are player 1!
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
                currentHeuristic = move_heuristic(move)

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
            move = heuristic_tiebreaker(bestMoves)

        # Make a new move using the randomly selected element of the randomly selected move
        # This modifies our copy of the board so that it matches the game's copy
        self.board.make_move(move, self.color)
        return move

    def build_search_tree(self, inciting_move):
        # Make inciting move None instead of an invalid zero-length move
        if len(inciting_move) <= 0:
            inciting_move = None

        # Create the root node from the current game state
        root = GameStateNode(GameState(self.board, self.color, inciting_move))

        for i in range(self.search_depth):
            root.expand()

        return root
