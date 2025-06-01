# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:54:49 2023

@author: Attila

 The code is based on Lukas Janasek's work (logic, functions, methodology).
 The main difference is the Actor function. This case it is continious. The output is mu and log sigma,
 which is transformed later on with exp. Before the output layer, the input is flattened.


"""
import logging
import os

import tensorflow as tf

import tensorflow_probability as tfp
import tf_keras
import numpy as np

EPSILON = 0.0000000001


def update_target(model_target, model_ref, rho=0.001):
    model_target.set_weights([rho * ref_weight + (1 - rho) * target_weight
                              for (target_weight, ref_weight) in
                              list(zip(model_target.get_weights(), model_ref.get_weights()))])


class Actor:
    '''
    Actor network for Continious case
    Defining a neural network that models a multivariate normal distribution
    Outputs are mu and log(sigma)--> not to loose information with softplus
    Stock_dimension: Number of stocks
    Flattened : We need to flatten cause the output is a matrix.

    '''

    def __init__(self, state_shape, stock_dimension,dropout_rate=0.1, args=None):
        # stock_dimension: Number of stocks
        # shape: (features,stocks)
        self.args = args
        y_inputs = tf.keras.layers.Input(shape=state_shape, dtype=tf.float32)
        # normalized = LayerNormalization(axis=1)(y_inputs)
        # conv = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(normalized)

        o1 = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu,
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(
            y_inputs)  # too high L2. it should be 0.001
        # normalized = LayerNormalization(axis=1)(o1)
        #o1 = tf.keras.layers.Dropout(rate=dropout_rate)(o1)
        o1 = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu,
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(o1)  # 0.01
        last_init = tf.keras.initializers.HeNormal()
        #o1 = tf.keras.layers.Dropout(rate=dropout_rate)(o1)
        flattened_input = tf.keras.layers.Flatten()(o1)
        alphas = tf.keras.layers.Dense(stock_dimension, activation='softplus', kernel_initializer=last_init)(flattened_input)

        self.model = tf.keras.Model(inputs=y_inputs, outputs=alphas)

    def call_action(self, state):
        # TODO : actually we should multiply (with less then 1) rather than add a number. We ll get
        # nan-s if we do not add +1. Also we may experience overflow
        alphas = self.model(state)

        variance_control = 1  # the higher the value, the lower the variance, must be positive 0.7 defailt and i used 0.5 for 26

        # Assign -1000 to all elements
        # batch_size = tf.shape(mu)[0]
        # indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], 4)], axis=1)
        # updates = tf.fill([batch_size], 1000.0)
        # mu = tf.tensor_scatter_nd_update(mu, indices, updates)

        return alphas + variance_control

    def sample_action(self, state):
        '''
        Sampling actions (portfolio weights)
        '''
        alphas = self.call_action(state)
        sampled_raw_actions = tfp.distributions.Dirichlet(alphas).sample()

        return sampled_raw_actions

    def get_log_action_prob(self, state, sampled_raw_actions):
        '''
        'sampled_raw_actions': We use this as an input and not sampling it here cause we would get different result.
        The sampled_raw_actions is needed before the learn function to do step.
        This sampled action comes from the same distibution(mu,sigma are the same here and sample_action funtion) so
        officially we should not loose anything if there is no training part before.
        When we are under the Gradient Tape we should
        learn the distribution via 'sampled_raw_actions' cause it describes the distribution.
        Question(1): would it change the training if the sampling (sampled_raw_actions part) would be under the gradient tape?

        '''
        alphas = self.call_action(state)
        dirichlet_dist = tfp.distributions.Dirichlet(alphas)
        log_prob_actions = tf.math.log(dirichlet_dist.prob(sampled_raw_actions) + EPSILON)
        log_prob_actions = tf.expand_dims(log_prob_actions, axis=1)

        entropy = dirichlet_dist.entropy()
        mean_entropy = tf.reduce_mean(entropy)

        return log_prob_actions, mean_entropy

    def get_trainable_vars(self):
        vars = self.model.trainable_variables

        return vars


def get_critic_model(state_shape, tau_levels=10, dirictlet=False, args=None):
    # stock_dimension: Number of stocks

    y_inputs = tf.keras.layers.Input(shape=state_shape, dtype=tf.float32)

    # normalized = LayerNormalization(axis=1)(y_inputs)
    # conv = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(normalized)

    o = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(y_inputs)
    # normalized = LayerNormalization(axis=1)(o)
    o = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(o)

    last_init = tf.keras.initializers.GlorotNormal() #tf.keras.initializers.GlorotNormal(): should be better than HeNormal here
    output = tf.keras.layers.Dense(tau_levels, activation='linear', kernel_initializer=last_init)(o)

    model = tf.keras.Model(y_inputs, output)
    return model


def get_monte_carlo_droput_critic(state_shape, tau_levels=10, dirictlet=False,dropout_rate=0.1, args=None):
    '''
    MC dropout is nothing else just sampling from trainied weights. Consequently, we let training=True
    during interference and add droput layer.
    Shapley values package is not competible with leaky relu, I changed here to relu.
    TODO: figoure this out
    '''

    y_inputs = tf.keras.layers.Input(shape=state_shape, dtype=tf.float32)

    # normalized = LayerNormalization(axis=1)(y_inputs)
    # conv = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(normalized)

    o = tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(y_inputs) # 0.0005
    # normalized = LayerNormalization(axis=1)(o)
    #o = tf.keras.layers.BatchNormalization()(o)
    o = tf.keras.layers.Dropout(rate=dropout_rate)(o)
    o = tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(o) # 0001 default
    #o = tf.keras.layers.BatchNormalization()(o)
    o = tf.keras.layers.Dropout(rate=dropout_rate)(o)
    last_init = tf.keras.initializers.GlorotNormal()
    output = tf.keras.layers.Dense(tau_levels, activation='linear', kernel_initializer=last_init)(o)

    model = tf.keras.Model(y_inputs, output)
    return model


'''

def prior(kernel_size, bias_size, dtype=None):
    """
    Define the prior distribution as a multivariate normal with mean 0 and variance 1.
    """
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n),scale_diag=tf.ones(n) * 2.0
                )
            )
        ]
    )
    return prior_model
'''

tfd = tfp.distributions

def prior(kernel_size, bias_size=0, dtype=None):
    """
    Returns a Keras model that, when called, outputs a tfp.Distribution
    (MultivariateNormalDiag) with trainable loc and fixed scale=2.0.
    """
    n = kernel_size + bias_size
    return tf_keras.Sequential([
        # Step 1: A VariableLayer that creates a trainable vector (our 'mu')
        tfp.layers.VariableLayer(
            shape=(n,),
            dtype=dtype,
            initializer=tf.keras.initializers.GlorotNormal(),
            name="prior_mu"
        ),
        # Step 2: Convert that vector into a TFP distribution
        tfp.layers.DistributionLambda(
            lambda mu: tfd.MultivariateNormalDiag(
                loc=mu,
                scale_diag=tf.ones([n], dtype=dtype) * 2.0 #2
            ),
            name="prior_dist"
        )
    ])
def posterior(kernel_size, bias_size=0, dtype=None):
    """
    Returns a Keras model that outputs a tfp.distributions.MultivariateNormalTriL
    with fully trainable loc and lower-triangular scale.
    """
    n = kernel_size + bias_size
    return tf_keras.Sequential([
        # Step 1: A single trainable vector that parameterizes (loc + raw scale_tril)
        tfp.layers.VariableLayer(
            tfp.layers.MultivariateNormalTriL.params_size(n),
            dtype=dtype,
            initializer=tf.keras.initializers.GlorotNormal(),
            name="posterior_params"
        ),
        # Step 2: Convert that vector into a TFP distribution
        # (This layer directly creates a MultivariateNormalTriL)
        tfp.layers.MultivariateNormalTriL(n, name="posterior_dist")
    ])


def get_bayesian_critic_model(state_shape, tau_levels=10):
    if isinstance(state_shape, tuple):
        state_shape = tf.TensorShape(state_shape)

    y_inputs = tf_keras.layers.Input(shape=state_shape, dtype=tf.float32)
    # 1st DenseVariational
    x = tfp.layers.DenseVariational(
        units=32,
        make_prior_fn=prior,          # returns the prior model
        make_posterior_fn=posterior,  # returns the posterior model
        kl_weight=64/1400,
        activation='relu',
        name="dense_variational_1"
    )(y_inputs)

    # 2nd DenseVariational
    x = tfp.layers.DenseVariational(
        units=16,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=64/1400,
        activation='relu',
        name="dense_variational_2"
    )(x)

    # Final deterministic Dense
    outputs = tf_keras.layers.Dense(tau_levels, activation='linear')(x)

    model = tf_keras.Model(y_inputs, outputs)
    return model


class QACDirichletAgent:
    def __init__(self, state_shape, stock_dimension, args):
        # stock_dimension: Number of stocks
        tau_levels = [t / args.tau_levels for t in range(1, args.tau_levels)]

        self.name = 'qac_agent'

        self.N = len(tau_levels)
        self.learning_tau = args.learning_tau
        self.tau_levels = tf.convert_to_tensor(tau_levels, dtype='float32')
        self.entropy_reg = args.entropy_reg
        self.args = args
        self.learning_tau_index = tau_levels.index(args.learning_tau)
        self.rho = args.rho
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.td_error_history = []
        self.entropy_history = []
        self.diff_penalty = []
        self.cost_history = []


        # self.y=[]
        self.v = []

        self.actor_network = Actor(state_shape, stock_dimension)
        if args.critic_type == 'monte_carlo_dropout':
            self.critic_network = get_monte_carlo_droput_critic(state_shape, self.N)
            self.critic_target = get_monte_carlo_droput_critic(state_shape, self.N)
        elif args.critic_type == 'bayesian':
            self.critic_network = get_bayesian_critic_model(state_shape, self.N)
            self.critic_target = get_bayesian_critic_model(state_shape, self.N)
        else:  # Default to standard critic model
            self.critic_network = get_critic_model(state_shape, self.N)
            self.critic_target = get_critic_model(state_shape, self.N)
            
        self.critic_target.set_weights(self.critic_network.get_weights())

        self.gradient_magnitudes_critic_per_episode = []
        self.gradient_magnitudes_actor_per_episode = []

        self.gamma = tf.constant(args.gamma, dtype='float32')

        critic_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            args.critic_lr_start,
            args.episodes, # 1406/batch * episodes
            end_learning_rate=args.critic_lr_end,
            power=1.5,  # it can cause NaNs, be careful with learning rates. In a non stationary environment we should not use decay rate
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_decay)

        actor_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            args.actor_lr_start,
            args.episodes,#args.episodes,
            end_learning_rate=args.actor_lr_end,
            power = 1.5,
        )
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_decay)

    def get_mean_of_action(self, s):
        alphas = self.actor_network.call_action(s)
        mean_share = alphas / tf.reduce_sum(alphas, axis=1, keepdims=True)

        return mean_share

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,64), dtype=tf.float32), # 111. To handle it we need to do a call function for this. but it is fine now :) 71,64
            #tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 7), dtype=tf.float32)
        ])
    def get_actions_and_rewards(self, s, price_close, price_close_next):
        a = self.actor_network.sample_action(s)

        return_rate = tf.reduce_sum(((price_close_next / price_close) - 1) * a, axis=1, keepdims=True) + 1
        wealth_now = tf.math.cumprod(return_rate, axis=0, exclusive=True)
        wealth_next = tf.math.cumprod(return_rate, axis=0, exclusive=False)

        reward = wealth_next - wealth_now

        return a, reward * 84.38

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 64), dtype=tf.float32), # 64 without indicators
            tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        ])
    def update_weights(self, s, sn, a_raw, r):
        """
        Function to update weights with optimizer
        s:   state
        sn:  next state
        a_y: log prob raw
        r:   reward
        For Bayesian layers, I think we need to add the critic.losses, which 
        contains the bayesian loss part ( KL divergence)
        critic_loss = (
            tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)
            + q_order_loss
            + sum(self.critic_network.losses)  # KL divergence for Bayesian layers
                )
        """
        vn = self.critic_target(sn)
        y = r + self.gamma * vn
        increased_order_loss_weight = 2.0
        # Original = without quantile estimations
        if self.args.actor_loss !='original':
            with tf.GradientTape() as critic_tape:
                v = self.critic_network(s)
                error = y - v

                abs_error = tf.math.abs(error)
                is_negative = tf.where(tf.math.less(error, 0.0), 1.0, 0.0)
                q_order_loss = tf.reduce_mean(tf.maximum(0.0, v[:, :-1] - v[:, 1:] + EPSILON))*increased_order_loss_weight
                #q_order_loss = tf.reduce_mean(tf.nn.softplus(v[:, :-1] - v[:, 1:]) * increased_order_loss_weight)

                loss = tf.math.multiply(tf.math.abs(tf.math.subtract(self.tau_levels, is_negative)), abs_error)
                critic_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)+q_order_loss

            critic_grad = critic_tape.gradient(critic_loss, self.critic_network.trainable_variables)
            # critic_grad = [tf.clip_by_norm(grad, 10) for grad in critic_grad]

            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables)
            )   
        else:
            with tf.GradientTape() as critic_tape:
                v = self.critic_network(s)
                error = y - v


                critic_loss = error.pow(2).mean()

            critic_grad = critic_tape.gradient(critic_loss, self.critic_network.trainable_variables)

            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables)
            )   

        v = self.critic_network(s)
        error = y - v
        #error = error[:, self.learning_tau_index: self.learning_tau_index + 1]
        #is_negative = tf.where(tf.math.less(error, 0.0), 1.0, 0.0)
        scale = 1
        if self.args.actor_loss=='weighted_quantile':
            t = self.tau_levels[self.learning_tau_index]
            error = error[:, self.learning_tau_index:self.learning_tau_index + 1]
            #is_negative = tf.where(error < 0.0, 1.0, 0.0)
            #t_weight = t - is_negative
            t_weight = tf.where(error < 0.0, 1.0 - t, t)

        elif self.args.actor_loss == 'advantage':
            error = error[:, self.learning_tau_index: self.learning_tau_index + 1]
        elif self.args.actor_loss == 'is_negative':
            error = error[:, self.learning_tau_index: self.learning_tau_index + 1]
            is_negative = tf.where(tf.math.less(error, 0.0), 1.0, 0.0)
            #is_positive = tf.where(error > 0.0, 1.0, 0.0)  # reinforce hitting upside
        elif self.args.actor_loss =='expectation':
            avg_error = tf.reduce_mean(error, axis=1, keepdims=True)

        elif self.args.actor_loss == 'power':
                error = error[:, self.learning_tau_index: self.learning_tau_index + 1]
                #t = self.tau_levels[self.learning_tau_index]
                t = tf.cast(self.tau_levels[self.learning_tau_index], tf.float32)

                eta = tf.where(
                    tf.equal(t, 0.1),
                    tf.constant(-2.0, dtype=tf.float32),
                    tf.where(
                        tf.equal(t, 0.9),
                        tf.constant(2.0, dtype=tf.float32),
                        tf.constant(0.0, dtype=tf.float32)
                    )
                    )
                weight_pow = tf.where(
                        eta >= 0,
                        tf.pow(t, 1 / (1 + tf.abs(eta))),
                        1 - tf.pow(1 - t, 1 / (1 + tf.abs(eta)))
                    )





        with tf.GradientTape() as actor_tape:
            log_prob, entropy = self.actor_network.get_log_action_prob(state=s, sampled_raw_actions=a_raw)
            if self.args.actor_loss=='is_negative':   
                #actor_loss = tf.math.reduce_mean(log_prob * scale * is_negative) - self.args.entropy_reg * entropy
                actor_loss = tf.reduce_mean(tf.reduce_sum(log_prob * scale * is_negative, axis=1))- self.args.entropy_reg * entropy
            elif self.args.actor_loss=='weighted_quantile':
                #actor_loss = -tf.reduce_mean(log_prob * error * t_weight)- self.args.entropy_reg * entropy
                actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * error * t_weight, axis=1)) - self.args.entropy_reg * entropy
            elif self.args.actor_loss=='advantage':
                #actor_loss = -tf.math.reduce_mean(log_prob * error)- self.args.entropy_reg * entropy
                actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * error, axis=1)) - self.args.entropy_reg * entropy
            elif self.args.actor_loss =='expectation':
                actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * avg_error, axis=1))
            elif self.args.actor_loss =='power':
                actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * error*weight_pow, axis=1))

                

        actor_vars = self.actor_network.get_trainable_vars()
        actor_grad = actor_tape.gradient(actor_loss, actor_vars)
        # actor_grad = [tf.clip_by_norm(grad, 1) for grad in actor_grad]
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, actor_vars)
        )

        return a_raw, v, vn, error, actor_loss, actor_grad, critic_loss, critic_grad

    def learn(self, s, sn, a_raw, r):
        """
        Run update for all networks (for training)
        s: state
        sn: next period state
        p: prices
        pn: next period prices
        """
        # print(s.shape,sn.shape,a_y.shape,r.shape)

        # s = tf.convert_to_tensor(s, dtype=tf.float32)
        # sn = tf.convert_to_tensor(sn, dtype=tf.float32)
        # a = tf.convert_to_tensor(p, dtype=tf.float32)
        # r = tf.convert_to_tensor(pn, dtype=tf.float32)
        r = tf.convert_to_tensor(r, dtype=tf.float32)
        a_raw, v, vn, error, actor_loss, actor_grad, critic_loss, critic_grad = self.update_weights(s, sn, a_raw, r)

        update_target(self.critic_target, self.critic_network, self.rho)
        return a_raw, v, vn, error, actor_loss, actor_grad, critic_loss, critic_grad

    def act(self, state):
        action_probabilities = self.actor_network.sample_action(state)
        return action_probabilities

    def save_weights(self, path):
        """
        Save weights to `path`
        """
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.actor_network.model.save_weights(path + "an_y.weights.h5")
        # self.actor_network.attention_model.save_weights(path + "an_a.h5")
        self.critic_network.save_weights(path + "cn.weights.h5")
        self.critic_target.save_weights(path + "ct.weights.h5")

    def load_weights(self, path, raise_error=False):
        """
        Load weights from path
        """
        try:
            self.actor_network.model.load_weights(path + "an_y.weights.h5")
            self.critic_network.load_weights(path + "cn.weights.h5")
            self.critic_target.load_weights(path + "ct.weights.h5")
        except OSError as err:
            logging.warning("Weights files cannot be found, %s", err)
            if raise_error:
                raise err
            
    def get_merged_states(self,data):
        '''
        we will randomly sample from this, we need to get the states as a list
        we only use the critic network: only state at time t what we need
        '''
        state=[]
        for batch in data:
            for instance in batch['state']:
                state.append(instance)
        return np.array(state)


