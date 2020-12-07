from datetime import datetime

import coloredlogs as coloredlogs
import tensorflow as tf

from Connect4Game import Connect4Game
from Trainer import Trainer
from tf_neural_net import TfNeuralNet
from utils import Config

action = "train"  # "test_human" for Neural Vs Human and "test_nn" for Neural Vs Neural
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

config = Config({
    'num_iters': 5,
    'num_self_play_games': 100,  # Number of self-play games to simulate.
    'temp_threshold': 15,  # Threshold after which exploration stops
    'model_accept_threshold': 0.6,  # Required threshold num of wins required to accept model.
    'queue_max_len': 200000,  # Number of game examples to train the neural networks.
    'num_mcts_simulations': 25,  # Number of games moves for MCTS to simulate.
    'num_tournament_games': 50,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    "logs_dir": 'logs/',
    'checkpoint': 'checkpoints/',
    'num_iters_for_train_examples_history': 25,
})


def main():
    if action == "train":
        print("")
        # Define Connect4 Game Board
        game = Connect4Game()
        # Define Neural Network
        # Uncomment Keras network if you want to, but it so slow when compared to tensorflow.
        # nn = KerasNeuralNet(game)
        # Tensorboard log writer
        tensorboard_log_dir = config.logs_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
        nn = TfNeuralNet(game, tensorboard_log_dir)
        # Trainer
        c = Trainer(game, nn, config, tensorboard_log_dir)
        c.train()


if __name__ == '__main__':
    main()
