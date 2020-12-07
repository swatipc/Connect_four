import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, Dense, Reshape
from tensorflow.python.keras.utils.vis_utils import plot_model
from tqdm import tqdm

from utils import Config, AvgMeter

config = Config({
    'lr': 0.001,
    'dropout': 0.3,
    'num_channels': 512,
    'epochs': 2,
    'batch_size': 64,
})

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def total_loss(y_true, y_pred):
    loss_pi = tf.losses.categorical_crossentropy(y_pred[0], y_true[0])
    loss_v = tf.losses.mean_squared_error(y_pred[1], tf.reshape(y_true[1], shape=[-1, ]))
    return loss_pi + loss_v


class KerasNeuralNet:
    def __init__(self, game):

        self.game = game
        self.shape = (game._base_board.height, game._base_board.width)
        self.input_board = Input(self.shape, dtype=float)
        self.possible_moves_size = self.shape[1]
        self.checkpoint_dir = "checkpoints"
        create_dir(self.checkpoint_dir)

        X = Reshape((self.shape[0], self.shape[1], 1))(self.input_board)
        h_conv1 = ReLU()(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same', use_bias=False)(X)))
        h_conv2 = ReLU()(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='same', use_bias=False)(h_conv1)))
        h_conv3 = ReLU()(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))
        h_conv4 = ReLU()(BatchNormalization(axis=3)(Conv2D(config.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))
        h_conv4_flat = Reshape((config.num_channels * (self.shape[0] - 4) * (self.shape[1] - 4), ))(h_conv4)
        s_fc1 = Dropout(config.dropout)(ReLU()(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))
        s_fc2 = Dropout(config.dropout)(ReLU()(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))

        self.pi = Dense(self.possible_moves_size, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.target_pi = Input([None, self.possible_moves_size], dtype=float)
        self.target_v = Input([None], dtype=float)

        model = Model(inputs=self.input_board, outputs=(self.pi, self.v))
        model.compile(loss=total_loss, optimizer=Adam(config.lr))
        print(model.summary())
        plot_model(model, "modelplots/model.png", show_shapes=True, show_layer_names=True)
        self.model = model

    def train(self, dataset):
        pi_losses = AvgMeter()
        v_losses = AvgMeter()

        for epoch in range(config.epochs):
            print("Epoch: " + str(epoch))
            num_batches = int(len(dataset) / config.batch_size)

            for i in tqdm(range(num_batches)):
                random_indexes = np.random.randint(len(dataset), size=config.batch_size)
                input_boards, target_pis, target_vs = list(zip(*[dataset[i] for i in random_indexes]))
                input_boards = np.asarray(input_boards, dtype=float)
                target_pis = np.asarray(target_pis, dtype=float)
                target_vs = np.asarray(target_vs, dtype=float)
                losses = self.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=config.batch_size)
                pi_losses.update(losses.history['pi_loss'], len(input_boards))
                v_losses.update(losses.history['v_loss'], len(input_boards))

    def predict(self, board):
        board = board[np.newaxis, :, :]
        predicted_pi, predicted_v = self.model(board, training=False)
        return predicted_pi[0], predicted_v[0]

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        self.model.load_weights(filepath)





