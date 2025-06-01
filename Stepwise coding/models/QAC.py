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
import numpy as np
from tensorflow.keras.layers import Dropout,LayerNormalization

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

    def __init__(self, state_shape, stock_dimension, args=None):
        # stock_dimension: Number of stocks
        # shape: (features,stocks)

        self.log_sigma = tf.Variable(initial_value=np.log(args.sigma_start), trainable=True, dtype=tf.float32)
        self.log_sigma.overwrite_with_gradient = False

        self.args = args
        y_inputs = tf.keras.layers.Input(shape=state_shape, dtype=tf.float32)
        #normalized = LayerNormalization(axis=-1)(y_inputs)
        # conv = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(normalized)

        o1 = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu,
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(
            y_inputs)  # too high L2. it should be 0.001

        o1 = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu,
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(o1)  # 0.01
        last_init = tf.keras.initializers.GlorotNormal()
        flattened_input = tf.keras.layers.Flatten()(o1)
        mu = tf.keras.layers.Dense(stock_dimension, activation='tanh', kernel_initializer=last_init)(flattened_input)

        self.model = tf.keras.Model(inputs=y_inputs, outputs=mu)

    def call_action(self, state):
        mu = self.model(state)

        sigma = (tf.exp(self.log_sigma) + EPSILON)
        # print(mu.shape,sigma.shape)
        return mu, sigma

    def sample_action(self, state):
        '''
        Sampling actions (portfolio weights)
        we will do the normalization later in the step function at the environment to add up to 1
        actions = tf.nn.softmax(sampled_raw_actions,axis=1)
        '''
        mu, sigma = self.call_action(state)
        sampled_raw_actions = tfp.distributions.Normal(mu, sigma).sample()

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
        mu, sigma = self.call_action(state)
        # print(mu,sigma)
        normal_dist = tfp.distributions.Normal(mu, sigma)
        entropy = normal_dist.entropy()
        log_prob_actions = tf.math.log(normal_dist.prob(sampled_raw_actions) + EPSILON)
        mean_entropy = tf.reduce_mean(entropy)

        return log_prob_actions, mean_entropy

    def get_trainable_vars(self):
        vars = self.model.trainable_variables
        vars += [self.log_sigma]

        return vars


def get_critic_model(state_shape, tau_levels=10, dirictlet=False, args=None):

    y_inputs = tf.keras.layers.Input(shape=state_shape, dtype=tf.float32)
    #normalized = LayerNormalization(axis=-1)(y_inputs)  


    o = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(y_inputs)

    o = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(o)
    #last_init = tf.keras.initializers.HeNormal()
    last_init = tf.keras.initializers.GlorotNormal()
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

    o = tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(y_inputs)
    # normalized = LayerNormalization(axis=1)(o)
    #o = tf.keras.layers.BatchNormalization()(o)
    o = tf.keras.layers.Dropout(rate=dropout_rate)(o)
    o = tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal(),
                              kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001))(o)
    #o = tf.keras.layers.BatchNormalization()(o)
    o = tf.keras.layers.Dropout(rate=dropout_rate)(o)
    last_init = tf.keras.initializers.HeNormal() # should be Glorot()
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

def prior(kernel_size, bias_size=0, dtype=None):
    """
    Prior distribution with a trainable mean and fixed wide variance.
    """
    n = kernel_size + bias_size
    
    
    trainable_mu = tf.Variable(tf.zeros(n), trainable=True, dtype=dtype, name="trainable_mu")
    
    
    fixed_sigma = tf.ones(n) * 3.0 # Optimal?

    return tf.keras.Sequential([
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(
                loc=trainable_mu,         # Trainable mean
                scale_diag=fixed_sigma    # Fixed variance
            )
        )
    ])


def posterior(kernel_size, bias_size, dtype=None):
    """
    Posterior distribution as a learnable multivariate normal.
    MultivariateNormalTriL: Faster training
    """
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def get_bayesian_critic_model(state_shape, tau_levels=10, args=None):
    '''
    https://keras.io/examples/keras_recipes/bayesian_neural_networks/
    https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseVariational
    '''
    if isinstance(state_shape, tuple):
        state_shape = tf.TensorShape(state_shape)

    y_inputs = tf.keras.layers.Input(shape=state_shape, dtype=tf.float32)
    # bayesian layer
    o = tfp.layers.DenseVariational(
        units=16,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / 100,   # This matters a lot. If we do with the whole data size, we ll get very different result
        activation=tf.nn.relu,
        activity_regularizer=tf.keras.regularizers.L2(l2=0.0005)
    )(y_inputs)
    o = tfp.layers.DenseVariational(
        units=16,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / 100,
            activation=tf.nn.relu,
            activity_regularizer=tf.keras.regularizers.L2(l2=0.0005)
    )(o)

    last_init = tf.keras.initializers.HeNormal()
    output = tf.keras.layers.Dense(tau_levels, activation='linear', kernel_initializer=last_init)(o)

    model = tf.keras.Model(inputs=y_inputs, outputs=output)
    return model

class QACAgent:
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

        self.current_wealth = tf.Variable(args.initial_wealth, dtype=tf.float32, trainable=False)
        # self.y=[]

        self.actor_network = Actor(state_shape, stock_dimension, args)
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
            args.episodes,
            end_learning_rate=args.critic_lr_end,
            power=1.5,  # it can cause NaNs, be careful with learning rates.
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_decay)

        actor_decay = tf.keras.optimizers.schedules.PolynomialDecay(
            args.actor_lr_start,
            args.episodes,
            end_learning_rate=args.actor_lr_end,
            power=1.5,
        )
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_decay)

    def get_mean_of_action(self, s):
        '''
        we do not sample. we take the model output and normalize to one
        '''
        mean_raw, _ = self.actor_network.call_action(s)
        mean_share = tf.nn.softmax(mean_raw)

        return mean_share
    # For eval, cause there we just run. without transaction costs etc it is ok


    
   





        
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 12), dtype=tf.float32), # 72
            tf.TensorSpec(shape=(None, 12), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32), # 7
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        ])
    def update_weights(self, s, sn, a_raw, r):
        """
        Function to update weights with optimizer
        s:   state
        sn:  next state
        a_y: log prob raw
        r:   reward
        """
        '''
        terminal state?
        '''
        vn = self.critic_target(sn)
        y = r + self.gamma * vn
        increased_order_loss_weight = 5.0  # 5.0
        with tf.GradientTape() as critic_tape:
            v = self.critic_network(s)  # if monte carlo dropout then set traing = True
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



        v = self.critic_network(s)
        error = y - v
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

        

        scale = 1
        # t_weight*10: Scale up!!
        with tf.GradientTape() as actor_tape:
            log_prob, entropy = self.actor_network.get_log_action_prob(state=s, sampled_raw_actions=a_raw)
            if self.args.actor_loss=='is_negative':   
                actor_loss = tf.reduce_sum(log_prob * scale * is_negative, axis=1)#- self.args.entropy_reg * entropy
            elif self.args.actor_loss=='weighted_quantile':
                actor_loss = -tf.reduce_sum(log_prob * error * t_weight*10, axis=1) #- self.args.entropy_reg * entropy
            elif self.args.actor_loss=='advantage':
                actor_loss = -tf.reduce_sum(log_prob * error, axis=1) #- self.args.entropy_reg * entropy

            actor_loss = tf.reduce_mean(actor_loss)- self.args.entropy_reg * entropy

        actor_vars = self.actor_network.get_trainable_vars()
        actor_grad = actor_tape.gradient(actor_loss, actor_vars)
        # actor_grad = [tf.clip_by_norm(grad, 1) for grad in actor_grad]

        self.actor_optimizer.apply_gradients(
            zip(actor_grad, actor_vars)
        )

        return a_raw, v, vn, error, actor_loss, actor_grad, critic_loss, critic_grad

    def learn(self, transitions): # transitions
        """
        Run update for all networks (for training)
        s: state
        sn: next period state
        p: prices
        pn: next period pricesss
        """
        # print(s.shape,sn.shape,a_y.shape,r.shape)
        #s = tf.expand_dims(tf.convert_to_tensor(s, dtype=tf.float32), axis=0)
        #sn = tf.expand_dims(tf.converst_to_tensor(sn, dtype=tf.float32), axis=0)
        #a_raw = tf.expand_dims(tf.convert_to_tensor(a_raw, dtype=tf.float32), axis=0)
     
        states, next_states, a_raws, rewards = zip(*transitions)

        # Convert to tensors
        s = tf.convert_to_tensor(states, dtype=tf.float32)
        sn = tf.convert_to_tensor(next_states, dtype=tf.float32)

        a_raw = tf.convert_to_tensor(a_raws, dtype=tf.float32)
        r = tf.convert_to_tensor(rewards, dtype=tf.float32)
     
        r = tf.reshape(r, (-1, 1))        


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
        self.actor_network.model.save_weights(path + "an_y.weights.h5")
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


