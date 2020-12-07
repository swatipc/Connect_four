import numpy as np

from Board import Board


class Connect4Game:
    def __init__(self):
        self.board = Board()
        self.board_size = (self.board.rows, self.board.cols)
        self.actions_size = self.board.cols

    # creates new board from given board and  update it with action
    def get_next_board(self, state, player, action):
        new_board = self.board.get_new_board(np.copy(state))
        new_board.add_move(action, player)
        return new_board, -player

    def get_game_ended_result(self, state, player):
        new_board = self.board.get_new_board(np.copy(state))
        win_state = new_board.get_win_state()
        if win_state.is_ended:
            if win_state.winner is None:
                # draw has very little value.
                return 1e-4
            elif win_state.winner == player:
                return +1
            elif win_state.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected win state found: ', win_state)
        else:
            # 0 used to represent unfinished game.
            return 0

    @staticmethod
    def get_canonical_form(board, player):
        # Flip the board from player perspective
        return board.state * player

    # Board can be flipped right/left based for more training augmented data
    @staticmethod
    def get_symmetries(state, pi):
        return [(state, pi), (state[:, ::-1], pi[::-1])]

    @staticmethod
    def encode(board):
        return board.tostring()


