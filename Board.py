from collections import namedtuple

import numpy as np

WinState = namedtuple('WinState', ['is_ended', 'winner'])


def display_board(state):
    print(" -----------------------")
    print(' '.join(map(str, range(len(state[0])))))
    print(state)
    print(" -----------------------")


def get_valid_moves(state):
    # zero value in top row denotes that it could be a valid move for any player.
    return state[0] == 0


class Board:
    def __init__(self, rows=6, cols=7, win_length=4, state=None):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        if state is None:
            self.state = np.zeros([self.rows, self.cols], dtype=np.int)
        else:
            self.state = state

    def get_new_board(self, state):
        if state is None:
            state = self.state
        return Board(self.rows, self.cols, self.win_length, state)

    def add_move(self, action, player):
        # Get the position in the column with zero
        zeros, = np.where(self.state[:, action] == 0)
        if len(zeros) == 0:
            raise ValueError("No space left in the column to add the move. Col: " + str(action))
        else:
            self.state[zeros[-1]][action] = player

    def get_win_state(self):
        for player in [-1, 1]:
            player_moves = self.state == -player

            if (self.is_straight_winner(player_moves) or  # checks horizontally
                    self.is_straight_winner(player_moves.transpose()) or  # Checks Vertically
                    self.is_diagonal_winner(player_moves)):  # Checks diagonally
                return WinState(True, -player)

        # draw has very little value.
        if not get_valid_moves(self.state).any():
            return WinState(True, None)

        # not ended.
        return WinState(False, None)

    # Checks if any specific moves contains a diagonal win.
    def is_diagonal_winner(self, player_moves):
        for i in range(len(player_moves) - self.win_length + 1):
            for j in range(len(player_moves[0]) - self.win_length + 1):
                if all(player_moves[i + x][j + x] for x in range(self.win_length)):
                    return True
            for j in range(self.win_length - 1, len(player_moves[0])):
                if all(player_moves[i + x][j - x] for x in range(self.win_length)):
                    return True
        return False

    # Checks if any specific player moves contains a vertical or horizontal win.
    def is_straight_winner(self, player_moves):
        straight_lengths = [player_moves[:, i:i + self.win_length].sum(axis=1)
                            for i in range(len(player_moves) - self.win_length + 2)]
        return max([x.max() for x in straight_lengths]) >= self.win_length
