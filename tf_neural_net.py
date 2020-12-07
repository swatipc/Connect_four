import logging
import os
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.nn import relu, tanh
from tensorflow.layers import dropout, dense, conv2d, batch_normalization

from utils import Config, AvgMeter

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

config = Config({
    'lr': 0.001,
    'dropout': 0.3,
    'num_channels': 512,
    'epochs': 10,
    'batch_size': 64,
})


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def conv2d(layer, padding):
    return tf.layers.conv2d(layer, config.num_channels, kernel_size=[3, 3], padding=padding)


class TfNeuralNet:
    def __init__(self, game, tensor_logs_dir):
        self.game = game
        self.shape = game.board_size
        self.possible_moves_size = self.shape[1]

        self.log_writer = tf.summary.FileWriter(tensor_logs_dir + "/losses")

        self.checkpoint_dir = "checkpoints"
        create_dir(self.checkpoint_dir)

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.saver = None

        with tf.Session() as temp_session:
            temp_session.run(tf.global_variables_initializer())

        with self.graph.as_default():
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.game.board.rows, self.game.board.cols])
            self.dropout = tf.placeholder(tf.float32)
            self.isTraining = tf.placeholder(tf.bool, name="is_training")
            self.pi_losses_var = tf.Variable(0, dtype=tf.float32)
            self.v_losses_var = tf.Variable(0, dtype=tf.float32)

            X = tf.reshape(self.input_boards, [-1, self.shape[0], self.shape[1], 1])
            h_conv1 = relu(batch_normalization(conv2d(X, 'same'), axis=3, training=self.isTraining))
            h_conv2 = relu(batch_normalization(conv2d(h_conv1, 'same'), axis=3, training=self.isTraining))
            h_conv3 = relu(batch_normalization(conv2d(h_conv2, 'valid'), axis=3, training=self.isTraining))
            h_conv4 = relu(batch_normalization(conv2d(h_conv3, 'valid'), axis=3, training=self.isTraining))
            h_conv4_flat = tf.reshape(h_conv4, [-1, config.num_channels * (self.shape[0] - 4) * (self.shape[1] - 4)])
            s_fc1 = dropout(relu(batch_normalization(dense(h_conv4_flat, 1024), axis=1, training=self.isTraining)),
                            rate=self.dropout)
            s_fc2 = dropout(relu(batch_normalization(dense(s_fc1, 512), axis=1, training=self.isTraining)),
                            rate=self.dropout)
            self.pi = dense(s_fc2, self.possible_moves_size)
            self.prob = tf.nn.softmax(self.pi)
            self.v = tanh(dense(s_fc2, 1))

            # Place holders for Predicted (pi, v)s
            self.predicted_pis = tf.placeholder(dtype=tf.float32, shape=[None, self.possible_moves_size])
            self.predicted_vs = tf.placeholder(dtype=tf.float32, shape=[None])

            # Real Losses
            self.loss_pi = tf.losses.softmax_cross_entropy(self.predicted_pis, self.pi)
            self.loss_v = tf.losses.mean_squared_error(self.predicted_vs, tf.reshape(self.v, shape=[-1, ]))
            self.total_loss = self.loss_pi + self.loss_v

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(config.lr).minimize(self.total_loss)

            self.session.run(tf.variables_initializer(self.graph.get_collection('variables')))

    def monitor_metrics(self, epoch, avg_pi_loss, avg_v_loss):
        # print('> %d, pi_losses: pi_losses: [%.3f], v_losses: [%.3f]' % (epoch, pi_losses, v_losses))
        self.session.run(self.pi_losses_var.assign(avg_pi_loss))
        self.log_writer.add_summary(self.session.run(tf.summary.scalar('pi_loss_avg', self.pi_losses_var)), epoch)

        self.session.run(self.v_losses_var.assign(avg_v_loss))
        self.log_writer.add_summary(self.session.run(tf.summary.scalar('v_loss_avg', self.v_losses_var)), epoch)

        self.log_writer.flush()

    def train(self, dataset):
        for epoch in range(1, config.epochs + 1):
            log.info("Epoch : %d",  epoch)

            pi_losses = AvgMeter()
            v_losses = AvgMeter()
            num_batches = int(len(dataset) / config.batch_size)

            tq = tqdm(range(num_batches), desc='Network Training')
            for i in tq:
                random_indexes = np.random.randint(len(dataset), size=config.batch_size)
                input_boards, predicted_pis, predicted_vs = list(zip(*[dataset[i] for i in random_indexes]))

                # Update network variables in the created placeholders
                placeholders_dict = {
                    self.input_boards: input_boards,
                    self.predicted_pis: predicted_pis,
                    self.predicted_vs: predicted_vs,
                    self.dropout: config.dropout,
                    self.isTraining: True
                }

                # Start training
                self.session.run(self.train_step, feed_dict=placeholders_dict)
                # Calculate losses
                pi_loss, v_loss = self.session.run([self.loss_pi, self.loss_v], feed_dict=placeholders_dict)

                pi_losses.update(pi_loss, len(input_boards))
                v_losses.update(v_loss, len(input_boards))
                tq.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                self.monitor_metrics(epoch, pi_losses.avg, v_losses.avg)


    def predict(self, board):
        board = board[np.newaxis, :, :]

        # Update network variables in the created placeholders
        placeholders_dict = {
            self.input_boards: board,
            self.dropout: 0,
            self.isTraining: False
        }

        predicted_pi, predicted_v = self.session.run([self.prob, self.v], feed_dict=placeholders_dict)
        return predicted_pi[0], predicted_v[0]

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        if self.saver is None:
            self.saver = tf.train.Saver(self.graph.get_collection('variables'))
        with self.graph.as_default():
            self.saver.save(self.session, filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, filepath)
