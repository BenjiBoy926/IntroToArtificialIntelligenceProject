from BoardClasses import Move
from BoardClasses import Board
import functools
import copy
import math
import random
import collections
import threading

"""
How to run the code locally:
From <project_dir>/Tools
python3 AI_Runner.py 7 7 2 l Sample_AIs/Poor_AI_368/main.py ../src/checkers-python/main.py

Manual play:
From <project_dir>/src/checkers-python
python3 main.py 7 7 2 m start_player 0
"""


# TODO: get a better board heuristic. Should be accurate and discriminative.
#  Slide 13 of the MCTS ideas slides gives some vague hint: "Implement simple MiniMax (a few moves
#  lookahead) and run your H against Random/Poor/Average. What is your win rate?"
#  (What does this even MEAN?!)

# Static functions

# Given the current player's number, get the number of their opponent
def opponent(player_number):
    if player_number == 1:
        return 2
    elif player_number == 2:
        return 1
    else:
        raise ValueError(f"Invalid player number '{player_number}'")


# Return a heuristic value for the board in this node.
# Smaller values are good for player 1 (black)
# Larger values are good for player 2 (white)
# NOTE: the comments in the board class we're using says that player 1 is white and player 2 is black,
# but the code seems to suggest otherwise.
# Comments: BoardClasses.py @ line 99
# Code that implies otherwise: BoardClasses.py @ line 104-110
def board_heuristic(board):
    return board.white_count - board.black_count


# We have to check if the strings are equal because checking if the objects are equal
# results in bad behaviour where moves that SHOULD be equal are not equal
def moves_equal(move1, move2):
    return str(move1) == str(move2)


# Randomly select a move that can be made on the given board by the given player
def random_move(board, player_number):
    moves = board.get_all_possible_moves(player_number)

    # Get a random checker and a random move for the checker
    random_checker_index = random.randrange(0, len(moves))
    random_move_index = random.randrange(0, len(moves[random_checker_index]))

    # Return the randomly selected move
    return moves[random_checker_index][random_move_index]


class GameStateTree:
    def __init__(self, col, row, p, player_number, exploration_constant, as_first_standard_blend_parameter):
        self.root = GameStateNode(player_number)
        self.board = Board(col, row, p)
        self.exploration_constant = exploration_constant
        self.as_first_standard_blend_parameter = as_first_standard_blend_parameter

        # Chain of moves that the tree used to get to the current propagate step of the simulations
        self.move_chain = []
        # Index the siblings of the nodes in the move chain by their inciting move
        # so that AMAF computations are faster
        self.sibling_index = collections.defaultdict(lambda: [])
        self.async_simulation_thread = None
        self.async_simulation_thread_running = False

        self.board.initialize_game()

    # Return true if we have many moves that we can make from the current root
    def has_multiple_moves(self):
        # Just in case the root has no children, expand it
        self.root.expand(self.board)
        return len(self.root.children) > 1

    # Choose the node with the best win ratio and return the move used to get to that node
    def choose_best_move(self):
        # Just in case the root has no children, expand it
        self.root.expand(self.board)
        return functools.reduce(self.better_win_ratio, self.root.children).inciting_move

    # Choose a random move on the current board
    def choose_random_move(self):
        return random_move(self.board, self.root.player_number)

    # Start an async thread of simulations on the tree
    def start_async_simulations(self):
        if self.async_simulation_thread is not None:
            self.stop_async_simulations()

        # Set the simulation thread to start running
        self.async_simulation_thread_running = True
        self.async_simulation_thread = threading.Thread(target=self.__async_simulations)

    # Stop async thread of simulations
    def stop_async_simulations(self):
        # Set thread running to false and join it to this thread
        if self.async_simulation_thread is not None:
            self.async_simulation_thread_running = False

            # If the thread is not alive yet, then start it
            if not self.async_simulation_thread.is_alive():
                self.async_simulation_thread.start()

            # Immediately join the thread and set it to none
            self.async_simulation_thread.join()
            self.async_simulation_thread = None

    # Run the number of simulations listed
    def run_simulations(self, iterations):
        for i in range(iterations):
            self.simulation_step()

    # Run a single step in the monte carlo simulation
    def simulation_step(self):
        # Clear out the move chain
        self.move_chain.clear()
        # Clear out the sibling index
        self.sibling_index.clear()

        # Select a leaf to simulate moves from
        current = self.__select()

        # Expand this node and get the result
        current = self.__expand(current)

        # Simulate a game and determine the result we got from the simulation
        result = self.__simulate(current.player_number)

        # Back propagate the results of the game
        self.__propagate(current, result)

    # Change the root of the tree to the child with the same inciting move
    def update_root(self, move):
        # Expand the root. This should not overwrite existing children
        self.root.expand(self.board)
        # Get a node in the children of the root with the same move as the one passed in
        match = filter(lambda n: moves_equal(n.inciting_move, move), self.root.children)
        # Get the next node in the iterator
        node = next(match, None)

        # If node is not none then update the root and the board
        if node is not None:
            node.make_move(self.board)

            # Update the root, and erase its parent
            self.root = node
            self.root.parent = None
        # If no child node is found that results from the given move, raise a value error
        else:
            raise ValueError(f"Current search tree root has no child node that results from move '{move}'")

    # Given two nodes, choose the one with the better Monte Carlo selection confidence
    def larger_selection_term(self, node1, node2):
        term1 = node1.selection_term(self.root.player_number, self.exploration_constant,
                                     self.as_first_standard_blend_parameter)
        term2 = node2.selection_term(self.root.player_number, self.exploration_constant,
                                     self.as_first_standard_blend_parameter)

        # Return the node with higher selection confidence
        if term1 > term2:
            return node1
        else:
            return node2

    # Given two nodes, return the node with the better win ratio, based on the player number of the root
    def better_win_ratio(self, node1, node2):
        ratio1 = node1.result_ratio(self.root.player_number, self.as_first_standard_blend_parameter)
        ratio2 = node2.result_ratio(self.root.player_number, self.as_first_standard_blend_parameter)

        if ratio1 > ratio2:
            return node1
        else:
            return node2

    def string(self, max_depth):
        stack = [self.root]
        string = ""

        while len(stack) > 0:
            current = stack.pop()
            depth = current.depth()

            # Output the vertical bars for nodes with a certain depth
            for i in range(depth):
                string += "|"

            # Add the string for this node to the total string
            string += "->"
            string += current.string(self.root.player_number, self.exploration_constant,
                                     self.as_first_standard_blend_parameter)
            string += "\n"

            if depth < max_depth:
                # Extend the stack with each child paired with its depth
                stack.extend([child for child in current.children])
            else:
                for i in range(depth + 1):
                    string += "|"

                # Output info that the results were truncated
                string += "->"
                string += "* results truncated"
                string += "\n"

        return string

    # The selection step of the Monte Carlo Tree search
    # Compute the upper confidence bound and use it to select the child to go to
    def __select(self):
        current = self.root

        while not current.is_leaf() and self.board.is_win(self.root.player_number) == 0:

            if current is self.root:
                print("SELECT:")

                for child in current.children:
                    print("\t" + child.string(self.root.player_number, self.exploration_constant,
                                              self.as_first_standard_blend_parameter))

            # Reduce to the child with the best confidence
            current = functools.reduce(self.larger_selection_term, current.children)

            if current.parent is self.root:
                print("\t\tChild chosen: " + current.string(self.root.player_number, self.exploration_constant,
                                                            self.as_first_standard_blend_parameter))

            # Make that move on the board, preparing to simulate
            self.__go_to_node(current)

        return current

    # The expansion step of the Monte Carlo
    # Expand the given node and choose a child to start the simulation from
    def __expand(self, selection):
        # If this board is not a win for anyone, expand the given node and choose a node to simulate a game from
        if self.board.is_win(self.root.player_number) == 0:
            selection.expand(self.board)

            # If some nodes were added after the expansion, select the first child
            if not selection.is_leaf():
                selection = selection.children[0]

                # Update the board to reflect the state at the returned node
                self.__go_to_node(selection)
                return selection
            # If no nodes were added in the expansion, return the same node
            # This would be kind of weird, because it means we didn't win but the board has no possible moves
            # If anything, the board should detect a tie at this point
            else:
                return selection
        # If this board is a win for someone, give the same node back for this expansion step
        else:
            return selection

    # The simulation step of the Monte Carlo
    # Run a random game from the current board and return true if we won and false if not
    def __simulate(self, player_number):
        total_moves = 0

        # Make random moves on the board until a win state is found
        while self.board.is_win(player_number) == 0:
            # Make a random move on the current board
            move = random_move(self.board, player_number)
            self.board.make_move(move, player_number)

            # Add this move to the move chain
            self.move_chain.append(move)

            # Increment total moves
            total_moves += 1
            # Update current player's move to their opponent
            player_number = opponent(player_number)

        # Get the final result of the simulation
        result = self.board.is_win(player_number)

        # Undo all the moves you just did so we have the correct board state
        for i in range(total_moves):
            self.board.undo()

        # Return the result of the board
        return result

    # Back propagate the result of a simulation from the given leaf node
    def __propagate(self, current, result):
        while current is not None:
            # If this node's parent is not none, then undo the move that got us to this node
            # If it IS none, we know that this is the root node, so no move got us here
            if current.parent is not None:
                self.board.undo()

            # Update the standard simulation data for the current node
            current.standard_simulation_data.update(result)

            # Update the current node to back-propagate
            current = current.parent

        # Go through all moves in the move chain
        for move in self.move_chain:
            # Go through each sibling with this move in the index
            for sibling in self.sibling_index[move]:
                sibling.as_first_simulation_data.update(result)

    # Runs simulations while the async thread is marked as "running"
    def __async_simulations(self):
        while self.async_simulation_thread_running:
            self.simulation_step()

    # Go to a node in the tree by making the node's move on the board
    # updating the move chain and adding to the sibling index
    def __go_to_node(self, node):
        node.make_move(self.board)
        # Append the inciting move to the move chain
        self.move_chain.append(node.inciting_move)
        # Append each sibling to the index, accessed by its move
        for sibling in node.siblings(False):
            self.sibling_index[sibling.inciting_move].append(sibling)


class GameStateNode:
    def __init__(self, player_number, inciting_move=None, parent=None):
        self.parent = parent
        self.children = []

        # Current player in this state and the move that resulted in this state
        self.player_number = player_number
        self.inciting_move = inciting_move

        # Initialize simulation data for "as first" and "standard"
        self.as_first_simulation_data = GameStateSimulationData()
        self.standard_simulation_data = GameStateSimulationData()

    # Expand this node by filling its list of children with all possible moves that can be made on the given board
    def expand(self, board):
        # Get a list of all possible moves on this board
        moves = board.get_all_possible_moves(self.player_number)

        # Add a child for each move in the list
        for checker_moves in moves:
            for move in checker_moves:
                # If no child exists with this move yet, then add it
                if all([child.inciting_move != move for child in self.children]):
                    self.children.append(GameStateNode(opponent(self.player_number), move, self))

    # Get the siblings of this node
    def siblings(self, include_self):
        if self.parent is not None:
            return [sibling for sibling in self.parent.children if sibling is not self or include_self]
        elif include_self:
            return [self]
        else:
            return []

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

    # Make the inciting move of this node using the player number of its parent
    def make_move(self, board):
        if self.inciting_move is not None:
            if self.parent is not None:
                board.make_move(self.inciting_move, self.parent.player_number)
            else:
                raise RuntimeError("This node has no parent who can make the inciting move on the board")
        else:
            raise RuntimeError("This node has no inciting move to make on the board")

    # True if this node is a leaf and has no children
    def is_leaf(self):
        return len(self.children) <= 0

    # Determine the blend between the "AMAF" simulations and standard simulations
    def as_first_standard_blend(self, param):
        if param > 0:
            return max(0, (param - self.standard_simulation_data.result_count()) / param)
        else:
            return 0

    # Return the confidence that Monte Carlo has that it should pick this node for the next simulation
    def selection_term(self, result, exploration_constant, param):
        if self.parent is not None:
            blend = self.as_first_standard_blend(param)

            parent_simulations = self.parent.as_first_simulation_data.result_count()
            as_first_term = self.as_first_simulation_data.selection_term(result, exploration_constant,
                                                                         parent_simulations)

            parent_simulations = self.parent.standard_simulation_data.result_count()
            standard_term = self.standard_simulation_data.selection_term(result, exploration_constant,
                                                                         parent_simulations)

            # If the standard term is non-zero then return the blend
            if standard_term != 0:
                return blend * as_first_term + (1 - blend) * standard_term
            # If the standard term is zero then return a large number to guarantee selection
            else:
                return 1000000
        else:
            raise RuntimeError("Cannot obtain the selection term of a node without a parent")

    # Ratio of times this node has gotten the given result
    # over the number of times this node has been simulated
    def result_ratio(self, result, param):
        blend = self.as_first_standard_blend(param)
        as_first_result_ratio = self.as_first_simulation_data.result_ratio(result)
        standard_result_ratio = self.standard_simulation_data.result_ratio(result)
        return blend * as_first_result_ratio + (1 - blend) * standard_result_ratio

    def string(self, result, exploration_constant, param):
        if self.inciting_move is not None:
            string = f"Node {self.inciting_move}"
        else:
            string = "Node (root)"

        string += f" - Standard: {self.standard_simulation_data.string(result)}, "
        string += f"AMAF: {self.as_first_simulation_data.string(result)}, "
        string += f"Blend: {self.as_first_standard_blend(param)}"

        # If this has a parent then add the selection term
        if self.parent is not None:
            string += f", Selection: {self.selection_term(result, exploration_constant, param)}"

        return string


class GameStateSimulationData:
    def __init__(self):
        self.results = collections.defaultdict(lambda: 0)

    # Count up all the results in the data
    def result_count(self):
        if len(self.results) > 0:
            def add(num1, num2):
                return num1 + num2

            # Add up all the results to get the result count
            return functools.reduce(add, self.results.values())
        # If it has no results to reduce then return 0
        else:
            return 0

    # Get the ratio for a particular result
    def result_ratio(self, result):
        count = self.result_count()

        # Double check if there are no results so we do not divide by zero
        if count > 0:
            return self.results[result] / self.result_count()
        else:
            return 0

    # Update the simulation data with a new result
    def update(self, result):
        self.results[result] += 1

    # Compute the selection term for this simulation data
    def selection_term(self, result, exploration_constant, parent_result_count):
        result_count = self.result_count()

        # If there are results then run the computation
        if result_count > 0:
            square_root_term = math.sqrt(math.log(parent_result_count) / result_count)
            return self.result_ratio(result) + exploration_constant * square_root_term
        else:
            return 0

    def string(self, result):
        return f"{self.results[result]}/{self.result_count()}"


# StudentAI class

# The following part should be completed by students.
# Students can modify anything except the class name and existing functions and variables.
class StudentAI:
    def __init__(self, col, row, p):
        # Build a tree for ourselves to use
        # The tree always starts as player 1
        self.tree = GameStateTree(col, row, p, 1, 2, 1000)

        # Start simulations immediately
        # This will be stopped really soon if our turn is first, but if their turn is first we may have time
        # to run some simulations before our first turn begins
        self.tree.start_async_simulations()

    # Get the next move that the AI wants to make
    # The move passed in is the move that the opponent just made,
    # or an invalid move if we get to move first
    def get_move(self, move):
        # Stop async simulations
        self.tree.stop_async_simulations()

        # If the opponent previously made a move, update our board to express it
        if len(move) != 0:
            self.tree.update_root(move)

        # If the tree has multiple moves it could make, run more simulations to improve the tree state
        if self.tree.has_multiple_moves():
            print("Running simulations...")
            self.tree.run_simulations(50)

        # Get the best move of the search tree
        print(f"Simulations complete, getting best move...")
        move = self.tree.choose_best_move()

        # Modify the board using the selected move
        print(f"Monte Carlo decision made: {move}")
        print(f"State of the tree:")
        print(self.tree.string(1))

        # Update the root of the tree so it is in the correct position the next time it is our turn
        print("Updating root for the tree")
        self.tree.update_root(move)

        # Now that we have made our move, start async simulations that can run while the other player
        # decides what they will do
        self.tree.start_async_simulations()

        # Return the selected move back to the caller
        return move
