# This algorithm and implementation was heavily referenced from https://web.stanford.edu/~surag/posts/alphazero.html
import logging
import math

import numpy as np

from Board import get_valid_moves, display_board

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    def __init__(self, game, nnet, config):
        self.game = game
        self.nnet = nnet
        self.args = config

        self.Qsa = {}  # Q values for (s,a)
        self.Nsa = {}  # num times edge (s,a) was visited
        self.Ns = {}  # num times a board s was visited
        self.Ps = {}  # initial policy (returned by neural net)
        self.Es = {}  # game result
        self.Vs = {}  # valid moves

    def get_action_prob(self, canonical_board, temp=1):
        for i in range(self.args.num_mcts_simulations):
            self.search(canonical_board)

        s = self.game.encode(canonical_board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.actions_size)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        # action = np.random.choice(len(probs), p=probs)
        # display_board(canonical_board)
        # print(probs)
        # print(action)

        return probs

    def get_action_with_high_prob(self, canonical_board):
        pis = self.get_action_prob(canonical_board, temp=0)
        return np.argmax(pis)

    def search(self, canonical_board):
        state = self.game.encode(canonical_board)

        if state not in self.Es:
            self.Es[state] = self.game.get_game_ended_result(canonical_board, 1)
        if self.Es[state] != 0:
            return -self.Es[state]

        if state not in self.Ps:
            self.Ps[state], v = self.nnet.predict(canonical_board)
            valid_moves = get_valid_moves(canonical_board)
            self.Ps[state] = self.Ps[state] * valid_moves
            sum_Ps_s = np.sum(self.Ps[state])
            self.Ps[state] /= sum_Ps_s
            self.Vs[state] = valid_moves
            self.Ns[state] = 0
            return -v

        valid_moves = self.Vs[state]
        current_best = -float('inf')
        best_action = -1

        # Choose the action with highest U value 
        for action in range(self.game.actions_size):
            if valid_moves[action]:
                if (state, action) in self.Qsa:
                    u = self.Qsa[(state, action)] + self.args.cpuct * self.Ps[state][action] * math.sqrt(
                        self.Ns[state]) / (
                                1 + self.Nsa[(state, action)])
                else:
                    u = self.args.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state] + EPS)

                if u > current_best:
                    current_best = u
                    best_action = action

        action = best_action
        next_board, next_player = self.game.get_next_board(canonical_board, 1, action)
        next_state = self.game.get_canonical_form(next_board, next_player)

        v = self.search(next_state)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + v) / (
                        self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1

        else:
            self.Qsa[(state, action)] = v
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return -v
