from Connect4Game import Connect4Game
from MCTS import MCTS
from tf_neural_net import TfNeuralNet
from utils import Config

config = Config({
    'num_mcts_simulations': 25,  # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    "logs_dir": 'logs/',
    'checkpoint': '../models/',
    'model_name': 'best_model.tar',
})


class Evaluator:

    def __init__(self):
        self.game = Connect4Game()
        self.neural_network = TfNeuralNet(self.game, 'logs')
        self.neural_network.load_checkpoint(config.checkpoint, config.model_name)
        self.mcts = MCTS(self.game, self.neural_network, config)

    def compute_action(self, state, player):
        winner = self.mcts.game.get_game_ended_result(state, player)
        if winner == -1:  # Human won
            return ['1', None]
        else:
            action = self.mcts.get_action_with_high_prob(state)
            next_board, _ = self.mcts.game.get_next_board(state, player, action)
            next_winner = self.mcts.game.get_game_ended_result(next_board.state, player)
            if next_winner == 0:
                return [None, int(action)]
            elif next_winner == 1:  # Network won
                return ['0', int(action)]
            else:  # Draw
                return ['-1', None]


