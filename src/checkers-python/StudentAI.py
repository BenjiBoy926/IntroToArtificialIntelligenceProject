from random import randint
from BoardClasses import Move
from BoardClasses import Board
import functools
import copy
import math

"""
How to run the code locally:
From <project_dir>/Tools
python3 AI_Runner.py 7 7 2 l Sample_AIs/Random_AI/main.py ../src/checkers-python/main.py

Manual play:
python3 main.py 7 7 2 m start_player 0
"""

# Static functions

# Given the current player's number, get the number of their opponent
def opponent(player_number):
    if player_number == 1:
        return 2
    elif player_number == 2:
        return 1
    else:
        raise ValueError(f"Invalid player number '{player_number}'")

# Get the board that results from the given move on the given player's turn
# NOTE: maybe later for memory efficiency we should use the same board and make moves / undo moves
# as we move down / up the tree. Just a thought
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


class GameStateTree:
    def __init__(self, root, board, exploration_constant):
        self.root = root
        self.board = board
        self.exploration_constant = exploration_constant

    def choose_best_move(self):
        # Select a leaf to simulate moves from
        current = self.select()

        # Expand this node...?
        current.expand()

        # Then: simulate a game
        # Finally: back propagate the results of the game

    # The selection step of the Monte Carlo Tree search
    # Compute the upper confidence bound and use it to select the child to go to
    def select(self):
        current = self.root

        while not current.is_leaf():
            # Reduce to the child with the best confidence
            current = functools.reduce(self.better_confidence, current.children)

            # Make that move on the board, preparing to simulate
            self.board.make_move(current.inciting_move, current.player_number)

        return current

    # Given two nodes, choose the one with the better Monte Carlo algorithm confidence
    def better_confidence(self, node1, node2):
        if node1.confidence(self.exploration_constant) > node2.confidence(self.exploration_constant):
            return node1
        else:
            return node2


class GameStateNode:
    def __init__(self, player_number, inciting_move, parent=None):
        self.parent = parent
        self.children = []

        # Current player in this state and the move that resulted in this state
        self.player_number = player_number
        self.inciting_move = inciting_move

        # Monte Carlo state
        self.simulations = 0
        self.wins = 0

    # Expand this node by filling its list of children with all possible moves that can be made on the given board
    def expand(self, board):
        # Get a list of all possible moves on this board
        moves = board.get_all_possible_moves(self.player_number)
        self.children = []

        # Add a child for each move in the list
        for checker_moves in moves:
            for move in checker_moves:
                self.children.append(GameStateNode(opponent(self.player_number), move))

    # Return the depth of this node, 0 if it has no parent
    def depth(self):
        current_depth = 0
        current_node = self.parent

        # Loop until the current node is none
        while current_node is not None:
            current_depth += 1

            # Update current to its own parent
            current_node = current_node.parent

        return current_depth

    # True if this node is a leaf and has no children
    def is_leaf(self):
        return len(self.children) <= 0

    # Return the confidence that Monte Carlo has that it should pick this node for the next simulation
    def confidence(self, exploration_constant):
        first_term = self.wins / self.simulations
        square_root_term = math.sqrt(math.log(self.parent.simulations) / self.simulations)
        return first_term + exploration_constant * square_root_term


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
        self.search_depth = 1

        # output file used by student ai
        self.output = open("output.txt", "w")

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

        # Build the search tree
        # print("Build the search tree...")
        tree_root = self.build_search_tree(move)

        # Get the minimax choice of the search tree
        # print(f"Tree constructed, getting minimax choice...")
        move = tree_root.minimax_choice().inciting_move

        # Modify the board using the selected move
        # print(f"Minimax decision made: {move}")
        self.board.make_move(move, self.color)

        # Return the selected move back to the caller
        return move

    def build_search_tree(self, inciting_move):
        # Make inciting move None instead of an invalid zero-length move
        if len(inciting_move) <= 0:
            inciting_move = None

        # Create the root node from the current game state
        root = GameStateNode(self.color, inciting_move)
        queue = [root]

        # Loop until no nodes remain in the queue for expanding
        while len(queue) > 0:
            # Pop the current node out of the front of the queue
            current = queue.pop(0)

            # If the current node's depth is less than the target search depth,
            # then expand it and add all it's children to the queue
            if current.depth() < self.search_depth * 2:
                current.expand()

                for child in current.children:
                    queue.append(child)

        # Return the root of the search tree
        return root
