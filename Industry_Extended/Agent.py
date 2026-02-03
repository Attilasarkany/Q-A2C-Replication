from models.QAC import QACAgent
from PortfolioEnvironment import Portfolio_engine
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import os
import tkinter
import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use('module://ipykernel.pylab.backend_inline')


import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
from monitoring import Monitoring
import argparse
import os
import logging
from tqdm.auto import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random
from scipy.stats import skew
from scipy.stats import kurtosis
LOG = logging.getLogger()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")
parser.add_argument("--evaluate", default=True, type=str2bool, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")

# General args for training
parser.add_argument("--batch_size", default=2048, type=int, help="Batch size.")
parser.add_argument("--episodes", default=30, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.5, type=float, help="Discounting factor.")

parser.add_argument("--learning_tau", type=float, default=0.2)
parser.add_argument('--tau_levels', type=int, nargs='?', default=10)

parser.add_argument('--critic_lr_start', type=float, nargs='?', default=0.01)
parser.add_argument('--critic_lr_end', type=float, nargs='?', default=0.001)
parser.add_argument('--actor_lr_start', type=float, nargs='?', default=0.001)
parser.add_argument('--actor_lr_end', type=float, nargs='?', default=0.0001)
parser.add_argument('--rho', type=float, nargs='?', default=0.1)
parser.add_argument('--entropy_reg', type=float, nargs='?', default=0.01)
parser.add_argument('--sigma_start', type=float, nargs='?', default=0.5) # 0.1

# Args for saving
parser.add_argument('--checkpoints_dir', type=str, nargs='?', default='./training_outcome')

parser.add_argument('--load_weights', type=str2bool, nargs='?', default=False)
parser.add_argument('--load_from_file', type=str, nargs='?', default='test')

parser.add_argument('--save_weights', type=str2bool, nargs='?', default=True)
parser.add_argument('--save_as_file', type=str, nargs='?', default='test')
parser.add_argument('--mode', type=str, nargs='?', default='train')
parser.add_argument('--add_tic_date', type=str2bool, nargs='?', default=False)
parser.add_argument('--reward_scaling', type=float, nargs='?', default=1221) # 1221

# Validation 
parser.add_argument('--min_epochs', type=int, default=4, 
                    help='Minimum number of epochs to train before early stopping is applied.')

parser.add_argument('--patience', type=int, default=2, 
                    help='Number of epochs to wait without improvement in validation loss before early stopping.')

parser.add_argument('--mc', type=str2bool, nargs='?', default=True)

parser.add_argument( '--critic_type', type=str, nargs='?', default='standard', 
    choices=['standard', 'monte_carlo_dropout', 'bayesian'],
    help="Type of critic model to use: 'standard', 'monte_carlo_dropout', 'bayesian'"
)


# Capital / fees / levarage 
#parser.add_argument('--initial_wealth', type=int, nargs='?', default=10000)

parser.add_argument( '--actor_loss', type=str, nargs='?', default='weighted_quantile', 
    choices=['advantage', 'is_negative', 'weighted_quantile','original'],
    help="Type of actor loss to use: 'standard', 'monte_carlo_dropout', 'bayesian'"
)
parser.add_argument('--action_interpret', type=str, nargs='?', default='transaction')
parser.add_argument('--transaction_cost', type=float, nargs='?', default=0.001)

# 25 portfolio : we dont need rf but ok
parser.add_argument('--r_f', type=float, nargs='?', default=1.0001)
parser.add_argument('--cost_fraction', type=float, nargs='?', default=0.0001)

parser.add_argument('--initial_wealth', type=float, nargs='?', default=1.0)


# Instantiate Args
#args = Args()



def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def main(env,agent,monitoring,args):
    logging.basicConfig(level=logging.INFO)
    args_string = '- ' + '\n- '.join(f'{k}={v}' for k, v in vars(args).items())

    with logging_redirect_tqdm():
        LOG.info(f'Running with the following args: \n{args_string}')

     

        if args.load_weights:
            LOG.info(f'Loading weights from {monitoring.load_path_root}')
            agent.load_weights(monitoring.load_path_root)
        else:
            LOG.info('Starting new training')
    with open(f'last_run.txt', 'w') as f:
        with trange(args.episodes) as t:
            minimum_val_error = float('inf') # to save one model for sure
            best_epoch = None
            best_model = None
            no_improvement_count= 0
            
            for ep, _ in enumerate(t):
                env.switch_mode("train")
                state=env.reset() # equal weight  initializan
                terminal=False
                total_reward=0
                values, target_values, errors, actor_losses, critic_losses, rewards,t_cost = [], [], [], [], [], [],[]
                transitions = []
                while not terminal:
                    raw_action=agent.act(state)
                    next_state,reward,terminal,transaction_cost=env.step(raw_action)
                    state_np = tf.squeeze(state).numpy()  # shape: (40,)
                    next_state_np = tf.squeeze(next_state).numpy()
                    raw_action_np = tf.squeeze(raw_action).numpy()
                    reward_np = float(reward) # scaled up for numerical reasons
                    transitions.append((state_np, next_state_np, raw_action_np, reward_np))

                    state=next_state
                    total_reward += reward
                    

                    rewards.append(reward)
                    t_cost.append(transaction_cost)


                a_raw, v, vn, error, actor_loss, actor_grad, critic_loss, critic_grad= agent.learn(transitions)
                values.append(v)
                target_values.append(vn)
                errors.append(error)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                if ep % 3 == 0 and terminal:
                        # Perform validation with the validation set
                        env.switch_mode("val") #load validation set
                        total_rewards, validation_loss,t_cost_val,wass = monitoring.stepwise_validation()
                    # Update monitoring metrics
                        monitoring.update_metric("Validation return", total_rewards)
                        monitoring.update_metric("Validation loss", validation_loss)
                        monitoring.update_metric("Validation transaction cost", t_cost_val)
                        # early stopping
                        # Warm up
                        if ep>=args.min_epochs:
                            if validation_loss < minimum_val_error:
                                minimum_val_error = validation_loss
                                best_epoch = ep
                                no_improvement_count = 0
                                # we save the latest best model and break if no improvement after a while
                                if args.save_weights:
                                    LOG.info('Saving weights to {}'.format(monitoring.save_path_root))
                                    agent.save_weights(monitoring.save_path_root)
                            else:
                                no_improvement_count +=1
                            if no_improvement_count > args.patience:
                                LOG.info(f"Early stopping at epoch {ep}. Best validation loss: {minimum_val_error} (Epoch {best_epoch})")
                                break
                    

                mean_v = np.mean(v, axis=0)
                mean_vn = np.mean(vn, axis=0)
                mean_error = np.mean(error)
                std_error = np.std(error)
                skew_error = skew(error)
                kurt_error = kurtosis(error)
                actor_loss = np.mean(actor_loss)
                critic_loss = np.mean(critic_loss)
                mean_r = np.mean(rewards)
                mean_cost = np.mean(t_cost)

                monitoring.update_metric("Value function distribution", mean_v)
                monitoring.update_metric("Value function distribution (target)", mean_vn)
                monitoring.update_metric("TD error", mean_error)
                monitoring.update_metric("std_error_td", std_error)
                monitoring.update_metric("skew_error_td", skew_error)
                monitoring.update_metric("kurt_error_td", kurt_error)

                monitoring.update_metric("Actor loss", actor_loss)
                monitoring.update_metric("Critic loss", critic_loss)
                monitoring.update_metric("Average return", mean_r)
                monitoring.update_metric("Average transaction cost", mean_cost)

    monitoring.save_metrics()
    monitoring.plot_training()
    env.switch_mode("train")
    monitoring.evaluate_on_training_testing_data_stepwise() 
    # need to switch   
    #env.switch_mode("train")
    #aggregated_quantile_means, aggregated_quantile_std, quantile_epistemic_std  = monitoring.monte_carlo_epistemic_uncertainty(iterations=20)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Step 1. create the environment
    data_file_etf = './Industry_long_daily.csv'
    stock_dimension = data_file_etf 
    data_file_exp = './Final_explanatory_Set.csv'
    date_level_exp_list = [
                            'Close_vix'
                           ] # date is included by default
    
    # 'US_leading_index', 'US_coincident_index', 'US_lagging_index','US_leadind_credit_index','Close_USD_EUR', 'AVG_PE_SPX','Close_treasury'
    tech_indicator_list = []
    set_global_determinism(seed=args.seed)

    env = Portfolio_engine(source_csv_file1 = data_file_etf,source_csv_file2 = data_file_exp,
                            date_level_exp_list = date_level_exp_list,
                            tech_indicator_list=tech_indicator_list,day = 0, args= args)
    agent = QACAgent(state_shape=env.state[0].shape,stock_dimension=env.stock_dimension, args=args)


    #agent = QACDirichletAgent(state_shape=batch_handler.states[0].shape, stock_dimension=7, args=args)

    monitoring = Monitoring(agent=agent,env=env, args=args)

    # Step 3. run the main script
    main(env,agent, monitoring, args)