import logging

from tqdm import tqdm

from Board import display_board, get_valid_moves

log = logging.getLogger(__name__)


class Tournament:
    def __init__(self, player1, player2, game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def play_game(self, display=False):
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.board
        iter = 1
        while self.game.get_game_ended_result(board.state, current_player) == 0:
            iter += 1
            if display:
                display_board(board)
                log.info("Iteration: %d :: Player: %d", iter, current_player)

            action = players[current_player + 1](self.game.get_canonical_form(board, current_player))
            valids = get_valid_moves(self.game.get_canonical_form(board, current_player))

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, current_player = self.game.get_next_board(board.state, current_player, action)

        winner = current_player * self.game.get_game_ended_result(board.state, current_player)
        if display:
            print("Game Over :: Iteration: %d :: Winner: %d", iter, winner)
            display_board(board)
        return winner

    def play_games(self, num, display=False):
        num = int(num / 2)
        player1_won = 0
        player2_won = 0
        draws = 0
        for _ in tqdm(range(num), desc="Tournament (Player 1 head start)"):
            winner = self.play_game(display=display)
            if winner == -1:
                player2_won += 1
            elif winner == 1:
                player1_won += 1
            else:
                draws += 1

        # Switch player roles
        self.player1, self.player2 = self.player2, self.player1

        for i in tqdm(range(num), desc="Tournament (Player 2 head start)"):
            winner = self.play_game(display=display)
            if winner == 1:
                player2_won += 1
            elif winner == -1:
                player1_won += 1
            else:
                draws += 1

        return player1_won, player2_won, draws
