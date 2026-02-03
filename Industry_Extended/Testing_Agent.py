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
parser.add_argument('--checkpoints_dir', type=str, nargs='?', default='./training_outcome') # For testing

parser.add_argument('--load_weights', type=str2bool, nargs='?', default=True) # For testing set to True
parser.add_argument('--load_from_file', type=str, nargs='?', default='20250911_final_weighted_q_spwise_standard_tanh_')

parser.add_argument('--save_weights', type=str2bool, nargs='?', default=True)
parser.add_argument('--save_as_file', type=str, nargs='?', default='test')
parser.add_argument('--mode', type=str, nargs='?', default='test')
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
parser.add_argument('--cost_fraction', type=float, nargs='?', default=0.0001) # 0.0014

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
        LOG.info(f'Testing with the following args: \n{args_string}')

        # Build the models by doing a forward pass with dummy data
        LOG.info('Building models...')
        dummy_state = env.reset()
        _ = agent.act(dummy_state)  # This builds actor and critic networks
        
        # Verify networks are built
        actor_built = len(agent.actor_network.model.get_weights()) > 0
        critic_built = len(agent.critic_network.get_weights()) > 0
        LOG.info(f'Networks initialized - Actor: {actor_built}, Critic: {critic_built}')
        
        # Capture initial random weights BEFORE loading
        actor_weights_before = agent.actor_network.model.get_weights()[0][0, :3].copy()
        critic_weights_before = agent.critic_network.get_weights()[0][0, :3].copy()
        LOG.info(f' BEFORE loading - Actor weights: {actor_weights_before}')
        LOG.info(f' BEFORE loading - Critic weights: {critic_weights_before}')
        
        if args.load_weights:
            LOG.info(f'Loading weights from {monitoring.load_path_root}')
            try:
                agent.load_weights(monitoring.load_path_root, raise_error=True)
                LOG.info(' Weights loaded successfully!')
                
                # Verify weights changed after loading (not just random initialization)
                actor_weights_after = agent.actor_network.model.get_weights()[0][0, :3]
                critic_weights_after = agent.critic_network.get_weights()[0][0, :3]
                LOG.info(f' AFTER loading - Actor weights: {actor_weights_after}')
                LOG.info(f' AFTER loading - Critic weights: {critic_weights_after}')
                
                # Confirm weights actually changed
                actor_changed = not np.allclose(actor_weights_before, actor_weights_after)
                critic_changed = not np.allclose(critic_weights_before, critic_weights_after)
                LOG.info(f' Weights changed - Actor: {actor_changed}, Critic: {critic_changed}')
            except Exception as e:
                LOG.error(f' Failed to load weights: {e}')
                LOG.error('Exiting - cannot test without trained weights')
                return
        else:
            LOG.info('You should load the weights for testing')
            return
    with open(f'last_run.txt', 'w') as f:
                env.switch_mode("test")

                monitoring.evaluate_on_training_testing_data_stepwise()
                #print("ac:",actor_loss)                                                                        



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Construct full folder name from base + seed + tau
    if args.seed is not None and args.learning_tau is not None:
        tau_code = str(int(args.learning_tau * 10)).zfill(2)  # 0.1->01, 0.5->05, 0.9->09
        # Build: 20250911_final_weighted_q_spwise_standard_tanh_53_01
        args.load_from_file = f"{args.load_from_file}{args.seed}_{tau_code}"
        print(f"Loading from: {args.load_from_file}")

    # Step 1. create the environment
    data_file_etf = './Industry_long_daily.csv'
    #stock_dimension = data_file_etf 
    data_file_exp = './Final_explanatory_Set.csv'
    #### Extension
    data_file_etf_extend = './Industry_long_daily_extended.csv'
    #stock_dimension = data_file_etf 
    data_file_exp_extend = './VIX_only.csv'
    date_level_exp_list = [
                            'Close_vix'
                           ] # date is included by default
    
    # 'US_leading_index', 'US_coincident_index', 'US_lagging_index','US_leadind_credit_index','Close_USD_EUR', 'AVG_PE_SPX','Close_treasury'
    tech_indicator_list = []
    set_global_determinism(seed=args.seed)

    env = Portfolio_engine(source_csv_file1 = data_file_etf,source_csv_file2 = data_file_exp,
                            source_csv_file3 = data_file_etf_extend, source_csv_file4 = data_file_exp_extend,
                            date_level_exp_list = date_level_exp_list,
                            tech_indicator_list=tech_indicator_list,day = 0, args= args)
    agent = QACAgent(state_shape=env.state[0].shape,stock_dimension=env.stock_dimension, args=args)


    #agent = QACDirichletAgent(state_shape=batch_handler.states[0].shape, stock_dimension=7, args=args)

    monitoring = Monitoring(agent=agent,env=env, args=args)

    # Step 3. run the main script
    main(env,agent, monitoring, args)

    '''
import numpy as np
import tensorflow as tf

#from models.qac_dirichlet import QACDirichletAgent
from models.QAC import QACAgent


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
%matplotlib inline

import logging
LOG = logging.getLogger()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# %%
def run_qac_batch(taus, seeds):
    ipython = get_ipython()
    for tau in taus:
        for seed in seeds:
            file_name = '20250911_final_weighted_q_spwise_standard_tanh_{}_{}'.format(

                str(seed), str(tau).replace('.', '')
            )

            cli_args = (
                f'--learning_tau {tau} --gamma 0.99 '
                f'--save_as_file {file_name} '
                f'--episodes 35 '
                f'--add_tic_date True '
                f'--min_epochs 20 '
                f'--patience 3 '
                f'--mc True '
                f'--critic_type standard '
                f'--rho 0.1 '
                f'--sigma_start 1.5 '
                f'--mode train '
                f'--transaction_cost 0.001 '
                f'--seed {seed} '
                f'--actor_loss weighted_quantile '
                f'--entropy_reg 1 '
	
            )

            print(f"\nRunning: Agent.py {cli_args}")
            ipython.run_line_magic('run', f'Agent.py {cli_args}')


taus = [0.1,0.5,0.9]
seeds = [53,274,1234,89]
run_qac_batch(taus,seeds)

    '''