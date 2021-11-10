from BoardClasses import Move
from BoardClasses import Board
import functools
import copy
import math
import random

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


# Given two nodes, return the node with the better win ratio
def better_win_ratio(node1, node2):
    if node1.win_ratio() > node2.win_ratio():
        return node1
    else:
        return node2


class GameStateTree:
    def __init__(self, board, player_number, exploration_constant):
        self.root = GameStateNode(player_number)
        self.board = board
        self.player_number = player_number
        self.exploration_constant = exploration_constant

    def choose_best_move(self):
        return functools.reduce(better_win_ratio, self.root.children).inciting_move

    def run_simulations(self, iterations):
        for i in range(iterations):
            # Select a leaf to simulate moves from
            current = self.select()

            # Expand this node and get the result
            current = self.expand(current)

            # Simulate a game and determine if we win
            we_won = self.simulate()

            # Back propagate the results of the game
            self.propagate(current, we_won)

    # The selection step of the Monte Carlo Tree search
    # Compute the upper confidence bound and use it to select the child to go to
    def select(self):
        print("Selecting a node")
        current = self.root

        while not current.is_leaf() and self.board.is_win(self.player_number) == 0:
            # Reduce to the child with the best confidence
            current = functools.reduce(self.better_confidence, current.children)

            # Make that move on the board, preparing to simulate
            self.board.make_move(current.inciting_move, current.player_number)

        return current

    # The expansion step of the Monte Carlo
    # Expand the given node and choose a child to start the simulation from
    def expand(self, selection):
        # If this board is not a win for anyone, expand the given node and choose a node to simulate a game from
        if self.board.is_win(self.player_number) == 0:
            selection.expand(self.board)

            # If some nodes were added after the expansion, select the first child
            if not selection.is_leaf():
                selection = selection.children[0]

                # Update the board to reflect the state at the returned node
                self.board.make_move(selection.inciting_move, self.player_number)
                return selection
            # If no nodes were added in the expansion, return the same node
            # This would be kind of weird, because it means we didn't win but the board has not possible moves
            # If anything, the board should detect a tie at this point
            else:
                return selection
        # If this board is a win for someone, give the same node back for this expansion step
        else:
            return selection

    # The simulation step of the Monte Carlo
    # Run a random game from the current board and return true if we won and false if not
    def simulate(self):
        total_moves = 0

        # Make random moves on the board until a win state is found
        while self.board.is_win(self, self.player_number) == 0:
            moves = self.board.get_all_possible_moves(self.player_number)

            # Get a random checker and a random move for the checker
            random_checker = random.randrange(0, len(moves))
            random_move = random.randrange(0, len(moves[random_checker]))

            # Make the random move on the current board
            move = moves[random_checker][random_move]
            self.board.make_move(move, self.player_number)

            # Increment total moves
            total_moves += 1

        # Get the final result of the simulation
        result = self.board.is_win(self.player_number)

        # Undo all the moves you just did so we have the correct board state
        for i in range(total_moves):
            self.board.undo()

        # Return true if we won
        return result == self.player_number

    # Back propagate the result of a simulation from the given leaf node
    def propagate(self, current, we_won):
        while current is not None:
            # Undo the move that got us to this board
            self.board.undo()

            # Increase number of simulations for this node
            current.simulations += 1

            # If we won, increase the wins for this node too
            if we_won:
                current.wins += 1

            # Update the current node to back-propagate
            current = current.parent

    # Given two nodes, choose the one with the better Monte Carlo algorithm confidence
    def better_confidence(self, node1, node2):
        if node1.confidence(self.exploration_constant) > node2.confidence(self.exploration_constant):
            return node1
        else:
            return node2

    # Change the root of the tree to the child with the same inciting move
    def update_root(self, move):
        # Get a node in the children of the root with the same move as the one passed in
        match = filter(lambda n: n.inciting_move == move, self.root.children)
        # Get the next node in the iterator
        node = next(match, None)

        # If node is not none then update the root and the board
        if node is not None:
            self.root = node
            self.board.make_move(move, self.player_number)
        # If no child node is found that results from the given move, raise a value error
        else:
            raise ValueError(f"Current search tree root has no child node that results from move '{move}'")


class GameStateNode:
    def __init__(self, player_number, inciting_move=None, parent=None):
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
        if self.simulations > 0:
            square_root_term = math.sqrt(math.log(self.parent.simulations) / self.simulations)
            return self.win_ratio() + exploration_constant * square_root_term
        # If this node is not simulated at all, we should definitely select it next!
        else:
            return math.inf

    def win_ratio(self):
        if self.simulations > 0:
            return self.wins / self.simulations
        else:
            return 0


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

        # Build a tree for ourselves to use
        self.tree = GameStateTree(self.board, self.color, 2)

    # Get the next move that the AI wants to make
    # The move passed in is the move that the opponent just made,
    # or an invalid move if we get to move first
    def get_move(self, move):
        # If the opponent previously made a move, update our board to express it
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
            self.tree.update_root(move)
        # If our opponent did not previously make a move, that means we are player 1!
        else:
            self.color = 1
            self.tree.player_number = self.color

        # Run simulations on the tree
        print("Running simulations...")
        self.tree.run_simulations(100)

        # Get the minimax choice of the search tree
        print(f"Simulations complete, getting best move...")
        move = self.tree.choose_best_move()

        # Modify the board using the selected move
        print(f"Monte Carlo decision made: {move}")
        self.board.make_move(move, self.color)

        # Update the root of the tree so it is in the correct position the next time it is our turn
        print("Update root for the tree")
        self.tree.update_root(move)

        # Return the selected move back to the caller
        return move
