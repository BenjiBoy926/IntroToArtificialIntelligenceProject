from random import randint
from BoardClasses import Move
from BoardClasses import Board
import functools
import copy

"""
How to run the code locally:
From <project_dir>/Tools
python3 AI_Runner.py 7 7 2 l Sample_AIs/Random_AI/main.py ../src/checkers-python/main.py

Manual play:
python3 main.py 7 7 2 m start_player 0
"""

# Static functions

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


# Given two states, return the state that would be best for the given player
def better_state(state1, state2, player_number):
    board = better_board(state1.board, state2.board, player_number)
    if board == state1.board:
        return state1
    else:
        return state2


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

    def is_win(self):
        return self.board.is_win(self.player_number) == self.player_number


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

        # Get the result of reducing all states based on the best state
        print(f"States to reduce: {[state.inciting_move for state in states_to_reduce]}")
        reduction = functools.reduce(self.my_better_state, states_to_reduce)
        print(f"State chosen: {reduction.inciting_move}")

        # Reduce the list down to the best state in the list for this player
        return reduction

    def my_better_state(self, state1, state2):
        return better_state(state1, state2, self.state.player_number)


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
        print("Build the search tree...")
        tree_root = self.build_search_tree(move)

        # Get the minimax choice of the search tree
        print(f"Tree constructed, getting minimax choice...")
        move = tree_root.minimax_choice().inciting_move

        # Modify the board using the selected move
        print(f"Minimax decision made: {move}")
        self.board.make_move(move, self.color)

        # Return the selected move back to the caller
        return move

    def build_search_tree(self, inciting_move):
        # Make inciting move None instead of an invalid zero-length move
        if len(inciting_move) <= 0:
            inciting_move = None

        # Create the root node from the current game state
        root = GameStateNode(GameState(self.board, self.color, inciting_move))
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
