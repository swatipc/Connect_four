import logging
import warnings
from collections import deque
from datetime import datetime
from random import shuffle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from Tournament import Tournament
from Board import Board, display_board
from MCTS import MCTS
from tf_neural_net import TfNeuralNet

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, game, nnet, config, tensor_logs_dir):
        self.config = config
        self.game = game
        self.nnet = nnet
        self.pnet = TfNeuralNet(self.game, tensor_logs_dir)  # Creating another network for competitor
        self.mcts = MCTS(self.game, self.nnet, self.config)
        self.train_examples_history = []

        self.log_writer = tf.summary.FileWriter(tensor_logs_dir + "/metrics")

        self.iter_var = tf.Variable(0, dtype=tf.int8)
        self.n_wins_var = tf.Variable(0, dtype=tf.int8)
        self.p_wins_var = tf.Variable(0, dtype=tf.int8)
        self.draws_var = tf.Variable(0, dtype=tf.int8)
        self.session = tf.Session()

    def start_game(self):
        train_examples = []
        self.current_player = 1
        games_count = 0
        board = Board()

        while True:
            games_count += 1
            canonical_state = self.game.get_canonical_form(board, self.current_player)
            temp = int(games_count < self.config.temp_threshold)

            pi = self.mcts.get_action_prob(canonical_state, temp=temp)
            symmetries = self.game.get_symmetries(canonical_state, pi)

            for s, p in symmetries:
                train_examples.append([s, self.current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            #
            # print("===========================")
            # display_board(canonical_state)
            # print(pi)
            # print(action)
            # print("===========================")

            board, self.current_player = self.game.get_next_board(board.state, self.current_player, action)

            r = self.game.get_game_ended_result(board.state, self.current_player)

            result = []
            if r != 0:
                for x in train_examples:
                    a = r * ((-1) ** (x[1] != self.current_player))
                    result.append((x[0], x[2], a))
                return result

    def train(self):

        for i in range(1, self.config.num_iters + 1):
            log.info(f'Iteration #{i} ...')
            iteration_train_samples = deque([], maxlen=self.config.queue_max_len)

            # Start self-play to generate training samples
            for i in tqdm(range(self.config.num_self_play_games), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.config)
                iteration_train_samples += self.start_game()

            # save the iteration examples to the history
            self.train_examples_history.append(iteration_train_samples)

            if len(self.train_examples_history) > self.config.num_iters_for_train_examples_history:
                self.train_examples_history.pop(0)

            train_examples = []
            for train_example in self.train_examples_history:
                train_examples.extend(train_example)

            self.nnet.save_checkpoint(folder=self.config.checkpoint, filename='temp_model.tar')
            self.pnet.load_checkpoint(folder=self.config.checkpoint, filename='temp_model.tar')
            p_mcts = MCTS(self.game, self.pnet, self.config)

            # shuffle examples before training
            shuffle(train_examples)

            self.nnet.train(train_examples)
            n_mcts = MCTS(self.game, self.nnet, self.config)

            log.info('Playing tournament against last iteration network')
            tournament = Tournament(p_mcts.get_action_with_high_prob, n_mcts.get_action_with_high_prob, self.game)
            p_wins, n_wins, draws = tournament.play_games(self.config.num_tournament_games)
            self.monitor_metrics(i, n_wins, p_wins, draws)

            # If all games are draw or if n_wins doesn't exceed threshold,
            if (p_wins + n_wins == 0) or (n_wins / float(p_wins + n_wins) < self.config.model_accept_threshold):
                log.info('Model not accepted')
                self.nnet.load_checkpoint(folder=self.config.checkpoint, filename='temp_model.tar')
            else:
                log.info('Model accepted')
                self.nnet.save_checkpoint(folder=self.config.checkpoint, filename='best_model.tar')

    def monitor_metrics(self, iter, n_wins, p_wins, draws):
        log.info('%d, N_wins/P_Wins: [%d/%d], Draws: [%d]' % (iter, n_wins, p_wins, draws))

        self.session.run(self.n_wins_var.assign(n_wins))
        self.log_writer.add_summary(self.session.run(tf.summary.scalar('n_wins', self.n_wins_var)), iter)

        self.session.run(self.p_wins_var.assign(p_wins))
        self.log_writer.add_summary(self.session.run(tf.summary.scalar('p_wins', self.p_wins_var)), iter)

        self.session.run(self.draws_var.assign(draws))
        self.log_writer.add_summary(self.session.run(tf.summary.scalar('draws', self.draws_var)), iter)

        self.log_writer.flush()

