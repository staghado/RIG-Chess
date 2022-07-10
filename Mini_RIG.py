# -*- coding: utf-8 -*-

import chess
from chess.pgn import Game

import requests
import bz2

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import time
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D
from keras.losses import mean_squared_error
from keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam

"""# Data retrieval and processing"""

def download_data_from_the_web(url='https://database.lichess.org/standard/lichess_db_standard_rated_2013-05.pgn.bz2') :
  
  r = requests.get(url, allow_redirects=True)
  decompressed_r = bz2.decompress(r.content)
  open('chess_data_sample.txt', 'wb').write(decompressed_r)
  with open('chess_data_sample.txt', 'rb') as f:
    lines = f.readlines()
  
  moves = []
  for l in lines :
    move = l.decode('utf-8')
    if move[0] != '[' and move[0] != '\n' and not '{' in move :
      moves.append(move)

  return moves

moves = download_data_from_the_web(url='https://database.lichess.org/standard/lichess_db_standard_rated_2013-05.pgn.bz2')

"""### Encoding the chessboard"""

def create_board(sequence_moves, max_moves=5) :
  
  board = chess.Board()
  moves_list = list(map(lambda x : x.strip(), re.split(r'(?:\d){1,2}\.', sequence_moves)))
  moves_list = moves_list[1:len(moves_list)-1]
  
  for t in moves_list[0:max_moves] : 
    board.push_san(t.split()[0])
    board.push_san(t.split()[1])
    
  return board

def fen_to_matrix(board):

  pieces = ['r', 'n', 'b', 'q', 'k', 'p'] + list(map(lambda x : x.upper(), ['r', 'n', 'b', 'q', 'k', 'p']))
  list_arrays = [ np.eye(12)[:, i] for i in range(12) ]
  encoding_dict = { k : v for k, v in zip(pieces, list_arrays) }
  fen_list = board.fen().split()[0].split('/')
  M = np.zeros((12, 8, 8))
  for i, row in enumerate(fen_list) :
    j = 0
    for e in row :
      if e.isdigit() :
        j+= int(e)
      else :
         M[:, i, j] = encoding_dict[e]
         j+=1
  return M

"""### Dataset preparation"""

def get_X_X(moves):
  X = [0]*len(moves)
  for i, move in enumerate(moves):
    #print(i, move)
    board = create_board(sequence_moves=move)
    M = fen_to_matrix(board)
    X[i] = M
  return X

X = get_X_X(moves)

"""# Variational Autoencoder

### Build
"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta = beta

    def init_encoder(self):
      latent_dim = 2
      encoder_inputs = keras.Input(shape=(12, 8, 8))
      x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
      x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
      x = layers.Flatten()(x)
      x = layers.Dense(16, activation="relu")(x)
      z_mean = layers.Dense(latent_dim, name="z_mean")(x)
      z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
      z = Sampling()([z_mean, z_log_var])
      encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
      return encoder

    def init_decoder(self):
      latent_dim = 2
      latent_inputs = keras.Input(shape=(latent_dim,))
      x = layers.Dense(3 * 2 * 64, activation="relu")(latent_inputs)
      x = layers.Reshape((3, 2, 64))(x)
      x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
      x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
      decoder_outputs = layers.Conv2DTranspose(8, 3, activation="sigmoid", padding="same")(x)
      decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
      return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            print(reconstruction.get_shape().as_list())
            print(data.get_shape().as_list())
            reconstruction_loss = tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction, axis=0), axis=[0, 1, 2]
                )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """Step run during validation."""
        if isinstance(data, tuple):
            data = data[0]

        
        z_mean, z_log_var, z = self.encoder(data)
        # For test we use only z_mean :
        reconstruction = self.decoder(z_mean)
        reconstruction_loss = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(data, reconstruction, axis=0), axis=[0, 1, 2]
             )
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        #print(kl_loss.get_shape().as_list())
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        #print(kl_loss.get_shape().as_list())
        kl_loss *= -0.5
        total_loss = reconstruction_loss + self.beta * kl_loss
      
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

"""### Test"""

x_data = np.array(X[:int(0.7 * len(X))])
x_val = np.array(X[int(0.7 * len(X)):])

x_data = x_data.astype('float32')
x_val = x_val.astype('float32')

vae = VAE(beta=1/1000)
vae.compile(optimizer=keras.optimizers.Adam())
history = vae.fit(x_data, epochs=100, batch_size=64, validation_data=(x_val, x_val))

plt.plot(history.history['reconstruction_loss'])
plt.plot(history.history['val_reconstruction_loss'])
plt.xlabel('Epoch')
plt.ylabel('Reconstruction loss(BCE)')
plt.legend(['Train', 'Val'])
plt.show()

plt.plot(history.history['kl_loss'])
plt.plot(history.history['val_kl_loss'])
plt.xlabel('Epoch')
plt.ylabel('KL loss')
plt.legend(['Train', 'Val'])
plt.show()

def plot_latent_space(vae, n=10, figsize=15):
    size = 8
    scale = 1.0
    figure = np.zeros((size * n, size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of chess configurations in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            board = x_decoded[0][11].reshape(size, size)
            figure[
                i * size : (i + 1) * size,
                j * size : (j + 1) * size,
            ] = board

    plt.figure(figsize=(figsize, figsize))
    start_range = size // 2
    end_range = n * size + start_range
    pixel_range = np.arange(start_range, end_range, size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z_mean")
    plt.ylabel("z_log_var")
    plt.imshow(figure)
    plt.colorbar()
    plt.show()


plot_latent_space(vae)

"""---

# Reinforcement Learning

### Environment
"""

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5

class Board(object):
  def __init__(self, encoder, decoder, opposing_agent, FEN=None, capture_reward_factor=0.01):
    """
    Chess Board Environment
    Args:
      FEN: str
      Starting FEN notation, if None then start in the default chess position
      capture_reward_factor: float [0,inf]
      reward for capturing a piece. Multiply material gain by this number. 0 for normal chess.
    """
    self.encoder = encoder
    self.decoder = decoder
    self.FEN = FEN
    self.opposing_agent = opposing_agent
    self.capture_reward_factor = capture_reward_factor
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.goal = self.generate_goal()
    self.layer_board = self.init_layer_board()

  def init_layer_board(self):
    """
    Initalize the numerical representation of the environment
    Returns:
    """
    M = np.zeros(shape=(8, 8, 8))
    for i in range(64):
      row = i // 8
      col = i % 8
      piece = self.board.piece_at(i)
      if piece == None:
        continue
      elif piece.symbol().isupper():
        sign = 1
      else:
        sign = -1
      layer = mapper[piece.symbol()]
      M[layer, row, col] = sign
      M[6, :, :] = 1 / self.board.fullmove_number
    if self.board.turn:
      M[6, 0, :] = 1
    else:
      M[6, 0, :] = -1
    M[7, :, :] = 1
    return np.concatenate((M, self.represent_goal()), axis=0)

  def update_layer_board(self, move=None):
      self._prev_layer_board = self.layer_board.copy()
      self.init_layer_board()

  def pop_layer_board(self):
      self.layer_board = self._prev_layer_board.copy()
      self._prev_layer_board = None

  def generate_goal(self):
    z_sample = np.random.normal(loc=0.0, scale=1.0, size=2)
    return z_sample

  def process_board(self):
    """
    Process a board
    """
    pieces = ['r', 'n', 'b', 'q', 'k', 'p'] + list(map(lambda x : x.upper(), ['r', 'n', 'b', 'q', 'k', 'p']))
    list_arrays = [ np.eye(12)[:, i] for i in range(12) ]
    encoding_dict = { k : v for k, v in zip(pieces, list_arrays) }
    fen_list = self.board.fen().split()[0].split('/')
    M = np.zeros((12, 8, 8))
    for i, row in enumerate(fen_list) :
      j = 0
      for e in row :
        if e.isdigit() :
          j+= int(e)
        else :
          M[:, i, j] = encoding_dict[e]
          j+=1
    return  M

  def step(self, action, test=True):
    """
    Run a step
    Args:
      action: python chess move
    Returns:
      epsiode end: Boolean Whether the episode has ended
      reward: float Difference in material value after the move
    """
    # Play a move
    self.board.push(action)
    self.update_layer_board(action)
    # Did the game end (checkmate or stalemate) or not yet
    result = self.board.result()
    if (result == '1-0') or (result == '0-1') or (result == '1/2-1/2'):
      episode_end = True
    else:
      episode_end = False
    # Compute the reward
    processed_board = np.expand_dims(self.process_board(), 0) # .astype("float32")
    encoded_result = self.encoder.predict(processed_board)[0]
    reward = -np.sqrt(np.sum((encoded_result-self.goal)**2))
    return episode_end, reward

  def get_random_action(self):
    """
    Sample a random action
    Returns: move A legal python chess move.
    """
    legal_moves = [x for x in self.board.generate_legal_moves()]
    legal_moves = np.random.choice(legal_moves)
    return legal_moves

  def project_legal_moves(self):
    """
    Create a mask of legal actions
    Returns: np.ndarray with shape (64,64)
    """
    self.action_space = np.zeros(shape=(64, 64))
    moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
    for move in moves:
      self.action_space[move[0], move[1]] = 1
    return self.action_space

  def get_material_value(self):
    """
    Sums up the material balance using Reinfield values
    Returns: The material balance on the board
    """
    pawns = 1 * np.sum(self.layer_board[0, :, :])
    rooks = 5 * np.sum(self.layer_board[1, :, :])
    minor = 3 * np.sum(self.layer_board[2:4, :, :])
    queen = 9 * np.sum(self.layer_board[4, :, :])
    return pawns + rooks + minor + queen

  def reset(self):
    """
    Reset the environment
    """
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.layer_board = self.init_layer_board()

  def sampled_board_threshold(self, threshold=0.5):
    """
    Sample a board using the decoder of the VAE
    """
    x_decoded = (self.decoder.predict(np.expand_dims(self.goal, 0))[0] >= threshold).astype('int64')
    board = [['.']*8 for i in range(8)]
    pieces = ['r', 'n', 'b', 'q', 'k', 'p'] + list(map(lambda x : x.upper(), ['r', 'n', 'b', 'q', 'k', 'p']))
    dict_pieces = {k:v for k, v in enumerate(pieces)}
    for p in range(x_decoded.shape[0]):
      for row in range(8):
        for col in range(8):
          if x_decoded[p, row, col] == 1:
            board[row][col] = dict_pieces[p]
    return board

  def print_goal(self):
    '''
    Print the goal as a multi line string
    '''
    board = self.sampled_board_threshold()
    for i in range(8):
      biggus_stringus = ''
      for j in range(8):
        biggus_stringus += board[i][j] + '  '
      print(biggus_stringus, end='\n')

  def represent_goal(self, threshold=0.5):
    '''
    Represent a latent goal as a (12, 8, 8) board
    '''
    return (self.decoder.predict(np.expand_dims(self.goal, 0))[0] >= threshold).astype('int64')

  def set_goal(self):
    '''
    Set a goal for the enironment
    '''
    self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
    self.goal = self.generate_goal()
    self.layer_board = self.init_layer_board()

"""### Agent"""

class GreedyAgent(object):

    def __init__(self, color=-1):
        self.color = color

    def predict(self, layer_board, noise=True):
        layer_board1 = layer_board[0, :, :, :]
        pawns = 1 * np.sum(layer_board1[0, :, :])
        rooks = 5 * np.sum(layer_board1[1, :, :])
        minor = 3 * np.sum(layer_board1[2:4, :, :])
        queen = 9 * np.sum(layer_board1[4, :, :])

        maxscore = 40
        material = pawns + rooks + minor + queen
        board_value = self.color * material / maxscore
        if noise:
            added_noise = np.random.randn() / 1e3
        return board_value + added_noise


class Agent(object):

    def __init__(self, lr=0.001, network='big'):
        self.optimizer = Adam(learning_rate=lr)
        self.model = Model()
        self.proportional_error = False
        if network == 'simple':
            self.init_simple_network()
        if network == 'big':
            self.init_bignet()

    def fix_model(self):
        """
        The fixed model is the model used for bootstrapping
        Returns:
        """

        self.fixed_model = clone_model(self.model)
        self.fixed_model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
        self.fixed_model.set_weights(self.model.get_weights())

    def init_simple_network(self):

        layer_state = Input(shape=(20, 8, 8), name='state')
        conv1 = Conv2D(8, (3, 3), activation='sigmoid')(layer_state)
        conv2 = Conv2D(6, (3, 3), activation='sigmoid')(conv1)
        conv3 = Conv2D(4, (3, 3), activation='sigmoid')(conv2)
        flat4 = Flatten()(conv3)
        dense5 = Dense(24, activation='sigmoid')(flat4)
        dense6 = Dense(8, activation='sigmoid')(dense5)
        value_head = Dense(1)(dense6)

        self.model = Model(inputs=layer_state,
                           outputs=value_head)
        self.model.compile(optimizer=self.optimizer,
                           loss=mean_squared_error
                           )

    def init_bignet(self):
        layer_state = Input(shape=(20, 8, 8), name='state')
        conv_xs = Conv2D(4, (1, 1), activation='relu')(layer_state)
        conv_s = Conv2D(8, (2, 2), strides=(1, 1), activation='relu')(layer_state)
        conv_m = Conv2D(12, (3, 3), strides=(2, 2), activation='relu')(layer_state)
        conv_l = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(layer_state)
        conv_xl = Conv2D(20, (8, 8), activation='relu')(layer_state)
        conv_rank = Conv2D(3, (1, 8), activation='relu')(layer_state)
        conv_file = Conv2D(3, (8, 1), activation='relu')(layer_state)

        f_xs = Flatten()(conv_xs)
        f_s = Flatten()(conv_s)
        f_m = Flatten()(conv_m)
        f_l = Flatten()(conv_l)
        f_xl = Flatten()(conv_xl)
        f_r = Flatten()(conv_rank)
        f_f = Flatten()(conv_file)

        dense1 = Concatenate(name='dense_bass')([f_xs, f_s, f_m, f_l, f_xl, f_r, f_f])
        dense2 = Dense(256, activation='sigmoid')(dense1)
        dense3 = Dense(128, activation='sigmoid')(dense2)
        dense4 = Dense(56, activation='sigmoid')(dense3)
        dense5 = Dense(64, activation='sigmoid')(dense4)
        dense6 = Dense(32, activation='sigmoid')(dense5)

        value_head = Dense(1)(dense6)

        self.model = Model(inputs=layer_state,
                           outputs=value_head)
        self.model.compile(optimizer=self.optimizer,
                           loss=mean_squared_error
                           )

    def predict(self, board_layer):
        return self.model.predict(board_layer)

    def TD_update(self, states, rewards, sucstates, episode_active, gamma=0.9):
        """
        Update the SARSA-network using samples from the minibatch
        Args:
            minibatch: list
                The minibatch contains the states, moves, rewards and new states.
        Returns:
            td_errors: np.array
                array of temporal difference errors
        """
        suc_state_values = self.fixed_model.predict(sucstates)
        V_target = np.array(rewards) + np.array(episode_active) * gamma * np.squeeze(suc_state_values)
        # Perform a step of minibatch Gradient Descent.
        self.model.fit(x=states, y=V_target, epochs=1, verbose=0)
        V_state = self.model.predict(states)  # the expected future returns
        td_errors = V_target - np.squeeze(V_state)

        return td_errors

    def MC_update(self, states, returns):
        """
        Update network using a monte carlo playout
        Args:
            states: starting states
            returns: discounted future rewards
        Returns:
            td_errors: np.array
                array of temporal difference errors
        """
        self.model.fit(x=states, y=returns, epochs=0, verbose=0)
        V_state = np.squeeze(self.model.predict(states))
        td_errors = returns - V_state

        return td_errors

"""### Tree search"""

def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.9):
        """
        Game Node for Monte Carlo Tree Search
        Args:
            board: the chess board
            parent: the parent node
            gamma: the discount factor
        """
        self.children = {}  # Child nodes
        self.board = board  # Chess board
        self.parent = parent
        self.values = []  # reward + Returns
        self.gamma = gamma
        self.starting_value = 0

    def update_child(self, move, Returns):
        """
        Update a child with a simulation result
        Args:
            move: The move that leads to the child
            Returns: the reward of the move and subsequent returns
        Returns:
        """
        child = self.children[move]
        child.values.append(Returns)

    def update(self, Returns=None):
        """
        Update a node with observed Returns
        Args:
            Returns: Future returns
        Returns:
        """
        if Returns:
            self.values.append(Returns)

    def select(self, color=1):
        """
        Use Thompson sampling to select the best child node
        Args:
            color: Whether to select for white or black
        Returns:
            (node, move)
            node: the selected node
            move: the selected move
        """
        assert color == 1 or color == -1, "color has to be white (1) or black (-1)"
        if self.children:
            max_sample = np.random.choice(color * np.array(self.values))
            max_move = None
            for move, child in self.children.items():
                child_sample = np.random.choice(color * np.array(child.values))
                if child_sample > max_sample:
                    max_sample = child_sample
                    max_move = move
            if max_move:
                return self.children[max_move], max_move
            else:
                return self, None
        else:
            return self, None

    def simulate(self, model, env, depth=0, max_depth=2, random=False, temperature=1):
        """
        Recursive Monte Carlo Playout
        Args:
            model: The model used for bootstrap estimation
            env: the chess environment
            depth: The recursion depth
            max_depth: How deep to search
            temperature: softmax temperature
        Returns:
            Playout result.
        """
        board_in = env.board.fen()
        if env.board.turn and random:
            move = np.random.choice([x for x in env.board.generate_legal_moves()])
        else:
            successor_values = []
            for move in env.board.generate_legal_moves():
                episode_end, reward = env.step(move)
                result = env.board.result()

                if (result == "1-0" and env.board.turn) or (
                        result == "0-1" and not env.board.turn):
                    env.board.pop()
                    env.init_layer_board()
                    break
                else:
                    if env.board.turn:
                        sucval = reward + self.gamma * np.squeeze(
                            model.predict(np.expand_dims(env.layer_board, axis=0)))
                    else:
                        sucval = np.squeeze(env.opposing_agent.predict(np.expand_dims(env.layer_board, axis=0)))
                    successor_values.append(sucval)
                    env.board.pop()
                    env.init_layer_board()

            if not episode_end:
                if env.board.turn:
                    move_probas = softmax(np.array(successor_values), temperature=temperature)
                    moves = [x for x in env.board.generate_legal_moves()]
                else:
                    move_probas = np.zeros(len(successor_values))
                    move_probas[np.argmax(successor_values)] = 1
                    moves = [x for x in env.board.generate_legal_moves()]
                if len(moves) == 1:
                    move = moves[0]
                else:
                    move = np.random.choice(moves, p=np.squeeze(move_probas))

        episode_end, reward = env.step(move)

        if episode_end:
            Returns = reward
        elif depth >= max_depth:  # Bootstrap the Monte Carlo Playout
            Returns = reward + self.gamma * np.squeeze(model.predict(np.expand_dims(env.layer_board, axis=0)))
        else:  # Recursively continue
            Returns = reward + self.gamma * self.simulate(model, env, depth=depth + 1,temperature=temperature)

        env.board.pop()
        env.init_layer_board()

        board_out = env.board.fen()
        assert board_in == board_out

        if depth == 0:
            return Returns, move
        else:
            noise = np.random.randn() / 1e6
            return Returns + noise

"""### TD search"""

class TD_search(object):

    def __init__(self, env, agent, gamma=0.9, search_time=1, memsize=2000, batch_size=256, temperature=1):
        """
        Chess algorithm that combines bootstrapped monte carlo tree search with Q Learning
        Args:
            env: RLC chess environment
            agent: RLC chess agent
            gamma: discount factor
            search_time: maximum time spent doing tree search
            memsize: Amount of training samples to keep in-memory
            batch_size: Size of the training batches
            temperature: softmax temperature for mcts
        """
        self.env = env
        self.agent = agent
        self.tree = Node(self.env)
        self.gamma = gamma
        self.memsize = memsize
        self.batch_size = batch_size
        self.temperature = temperature
        self.reward_trace = []  # Keeps track of the rewards
        self.piece_balance_trace = []  # Keep track of the material value on the board
        self.ready = False  # Whether to start training
        self.search_time = search_time
        self.min_sim_count = 10

        self.mem_state = np.zeros(shape=(1, 20, 8, 8))
        self.mem_sucstate = np.zeros(shape=(1, 20, 8, 8))
        self.mem_reward = np.zeros(shape=(1))
        self.mem_error = np.zeros(shape=(1))
        self.mem_episode_active = np.ones(shape=(1))

    def learn(self, iters=40, c=5, timelimit_seconds=3600, maxiter=80):
        """
        Start Reinforcement Learning Algorithm
        Args:
            iters: maximum amount of iterations to train
            c: model update rate (once every C games)
            timelimit_seconds: maximum training time
            maxiter: Maximum duration of a game, in halfmoves
        Returns:
        """
        starttime = time.time()
        for k in range(iters):
            self.env.reset()
            print("Goal:")
            self.env.print_goal()
            if k % c == 0:
                self.agent.fix_model()
                print("iter", k)
            if k > c:
                self.ready = True
            self.play_game(k, maxiter=maxiter)
            if starttime + timelimit_seconds < time.time():
                break
        return self.env.board

    def play_game(self, k, maxiter=80):
        """
        Play a chess game and learn from it
        Args:
            k: the play iteration number
            maxiter: maximum duration of the game (halfmoves)
        Returns:
            board: Chess environment on terminal state
        """
        episode_end = False
        turncount = 0
        tree = Node(self.env.board, gamma=self.gamma)  # Initialize the game tree

        # Play a game of chess
        while not episode_end:
            state = np.expand_dims(self.env.layer_board.copy(), axis=0)
            state_value = self.agent.predict(state)

            # White's turn involves tree-search
            if self.env.board.turn:

                # Do a Monte Carlo Tree Search after game iteration k
                start_mcts_after = -1
                if k > start_mcts_after:
                    tree = self.mcts(tree)
                    # Step the best move
                    max_move = None
                    max_value = np.NINF
                    for move, child in tree.children.items():
                        sampled_value = np.mean(child.values)
                        if sampled_value > max_value:
                            max_value = sampled_value
                            max_move = move
                else:
                    max_move = np.random.choice([move for move in self.env.board.generate_legal_moves()])
                
                print("White's turn:")
                print(self.env.board)

            # Black's turn is myopic
            else:
                max_move = None
                max_value = np.NINF
                for move in self.env.board.generate_legal_moves():
                    self.env.step(move)
                    if self.env.board.result() == "0-1":
                        max_move = move
                        self.env.board.pop()
                        self.env.init_layer_board()
                        break
                    successor_state_value_opponent = self.env.opposing_agent.predict(
                        np.expand_dims(self.env.layer_board, axis=0))
                    if successor_state_value_opponent > max_value:
                        max_move = move
                        max_value = successor_state_value_opponent

                    self.env.board.pop()
                    self.env.init_layer_board()

                print("Black's turn:")
                print(self.env.board)

            if not (self.env.board.turn and max_move not in tree.children.keys()) or not k > start_mcts_after:
                tree.children[max_move] = Node(gamma=0.9, parent=tree)

            episode_end, reward = self.env.step(max_move)
            print('White Turn : ', self.env.board.turn)
            print(f"Reward at turn {turncount} : {reward}")
            print(f"Maxmove {turncount} : {max_move}")

            tree = tree.children[max_move]
            tree.parent = None
            gc.collect()

            sucstate = np.expand_dims(self.env.layer_board, axis=0)
            new_state_value = self.agent.predict(sucstate)

            error = reward + self.gamma * new_state_value - state_value
            error = float(np.squeeze(error))

            turncount += 1
            if turncount > maxiter and not episode_end:
                episode_end = True

            episode_active = 0 if episode_end else 1

            # construct training sample state, prediction, error
            self.mem_state = np.append(self.mem_state, state, axis=0)
            self.mem_reward = np.append(self.mem_reward, reward)
            self.mem_sucstate = np.append(self.mem_sucstate, sucstate, axis=0)
            self.mem_error = np.append(self.mem_error, error)
            self.reward_trace = np.append(self.reward_trace, reward)
            self.mem_episode_active = np.append(self.mem_episode_active, episode_active)

            if self.mem_state.shape[0] > self.memsize:
                self.mem_state = self.mem_state[1:]
                self.mem_reward = self.mem_reward[1:]
                self.mem_sucstate = self.mem_sucstate[1:]
                self.mem_error = self.mem_error[1:]
                self.mem_episode_active = self.mem_episode_active[1:]
                gc.collect()

            if turncount % 10 == 0:
                self.update_agent()

        piece_balance = self.env.get_material_value()
        self.piece_balance_trace.append(piece_balance)
        print("game ended with result", reward, "and material balance", piece_balance, "in", turncount, "halfmoves")

        return self.env.board

    def update_agent(self):
        """
        Update the Agent with TD learning
        Returns:
            None
        """
        if self.ready:
            choice_indices, states, rewards, sucstates, episode_active = self.get_minibatch()
            td_errors = self.agent.TD_update(states, rewards, sucstates, episode_active, gamma=self.gamma)
            self.mem_error[choice_indices.tolist()] = td_errors

    def get_minibatch(self, prioritized=True):
        """
        Get a mini batch of experience
        Args:
            prioritized:
        Returns:
        """
        if prioritized:
            sampling_priorities = np.abs(self.mem_error) + 1e-9
        else:
            sampling_priorities = np.ones(shape=self.mem_error.shape)
        sampling_probs = sampling_priorities / np.sum(sampling_priorities)
        sample_indices = [x for x in range(self.mem_state.shape[0])]
        choice_indices = np.random.choice(sample_indices,
                                          min(self.mem_state.shape[0],
                                              self.batch_size),
                                          p=np.squeeze(sampling_probs),
                                          replace=False
                                          )
        states = self.mem_state[choice_indices]
        rewards = self.mem_reward[choice_indices]
        sucstates = self.mem_sucstate[choice_indices]
        episode_active = self.mem_episode_active[choice_indices]

        return choice_indices, states, rewards, sucstates, episode_active

    def mcts(self, node):
        """
        Run Monte Carlo Tree Search
        Args:
            node: A game state node object
        Returns:
            the node with playout sims
        """

        starttime = time.time()
        sim_count = 0
        board_in = self.env.board.fen()

        # First make a prediction for each child state
        for move in self.env.board.generate_legal_moves():
            if move not in node.children.keys():
                node.children[move] = Node(self.env.board, parent=node)

            episode_end, reward = self.env.step(move)

            if episode_end:
                successor_state_value = 0
            else:
                successor_state_value = np.squeeze(
                    self.agent.model.predict(np.expand_dims(self.env.layer_board, axis=0))
                )

            child_value = reward + self.gamma * successor_state_value

            node.update_child(move, child_value)
            self.env.board.pop()
            self.env.init_layer_board()
        if not node.values:
            node.values = [0]

        while starttime + self.search_time > time.time() or sim_count < self.min_sim_count:
            depth = 0
            color = 1
            node_rewards = []

            # Select the best node from where to start MCTS
            while node.children:
                node, move = node.select(color=color)
                if not move:
                    # No move means that the node selects itself, not a child node.
                    break
                else:
                    depth += 1
                    color = color * -1  # switch color
                    episode_end, reward = self.env.step(move)  # Update the environment to reflect the node
                    node_rewards.append(reward)
                    # Check best node is terminal

                    if self.env.board.result() == "1-0" and depth == 1:  # -> Direct win for white, no need for mcts.
                        self.env.board.pop()
                        self.env.init_layer_board()
                        node.update(1)
                        node = node.parent
                        return node
                    elif episode_end:  # -> if the explored tree leads to a terminal state, simulate from root.
                        while node.parent:
                            self.env.board.pop()
                            self.env.init_layer_board()
                            node = node.parent
                        break
                    else:
                        continue

            # Expand the game tree with a simulation
            Returns, move = node.simulate(self.agent.fixed_model,
                                          self.env,
                                          temperature=self.temperature,
                                          depth=0)
            self.env.init_layer_board()

            if move not in node.children.keys():
                node.children[move] = Node(self.env.board, parent=node)

            node.update_child(move, Returns)

            # Return to root node and backpropagate Returns
            while node.parent:
                latest_reward = node_rewards.pop(-1)
                Returns = latest_reward + self.gamma * Returns
                node.update(Returns)
                node = node.parent

                self.env.board.pop()
                self.env.init_layer_board()
            sim_count += 1

        board_out = self.env.board.fen()
        assert board_in == board_out

        return node

"""# RIG"""

class RIG_Algorithm(object):

  def __init__(self, env, agent, TD_search, Node, vae, data, epochs_vae = 100, N_episodes = 100, N_steps = 100,  gamma=0.9, search_time=1, memsize=2000, batch_size=64):
    """
    RIG algorithm
    Args:
    """
    self.env = env
    self.agent = agent
    self.TD_search = TD_search
    self.Node = Node
    self.vae = vae
    self.data = data
    self.epochs_vae = epochs_vae
    self.N_episodes = N_episodes
    self.N_steps = N_steps
    self.gamma = gamma
    self.search_time = search_time
    self.memsize = memsize
    self.batch_size = batch_size


  def run(self):
    # 1 : collect D = {s(i)} using exploration policy : we have data 
    # 2 : Train VAE on D

    self.vae.compile(optimizer=keras.optimizers.Adam())
    self.vae.fit(self.data, epochs=self.epochs_vae, batch_size=self.batch_size) 
    # 3 : Fit prior p(z) to latent encodings : done in step 2
    # 4 : 

    for n in range(self.N_episodes):
      # Generate a new goal
      #generated_goal = np.random.normal(loc=0.0, scale=1.0, size=2)
      #self.TD_search.env.goal = generated_goal
      #self.env.goal = generated_goal
      self.TD_search.env.set_goal()

      self.TD_search.learn(iters=self.N_steps, c=5, timelimit_seconds=180, maxiter=10)

"""### Execution"""

vae = VAE(beta=1/10000)

opponent = GreedyAgent()
player = Agent(lr=0.01, network='simple')

env = Board(vae.encoder, vae.decoder, opponent, FEN=None)

learner = TD_search(env, player, gamma=0.99, search_time=2)
node = Node(learner.env.board, gamma=learner.gamma)

data = x_data

rig = RIG_Algorithm(env, player, learner, node, vae, data, N_episodes = 10, N_steps = 10)

rig.run()

"""### Results"""

plt.plot(rig.TD_search.reward_trace)

reward_smooth = pd.DataFrame(rig.TD_search.reward_trace)
reward_smooth.rolling(window=100, min_periods=0).mean().plot(figsize=(16, 9),
                                                             title='average performance')
plt.show()

pgn = Game.from_board(learner.env.board)

with open("rlc_pgn", "w") as log:
    log.write(str(pgn))