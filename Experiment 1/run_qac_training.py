import argparse
import os
import random

import pandas as pd

from portfolio_env.data_processing import BatchHandler
from portfolio_env.monitoring import Monitoring

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

import logging
from tqdm.auto import trange
from tqdm.contrib.logging import logging_redirect_tqdm
from scipy.stats import skew
from scipy.stats import kurtosis

LOG = logging.getLogger()

# import sys
# original_stdout = sys.stdout

from models.qac import QACAgent
from models.qac_dirichlet import QACDirichletAgent

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
parser.add_argument("--evaluate", default=True, type=bool, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")

# General args for training
parser.add_argument("--batch_size", default=2048, type=int, help="Batch size.")
parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.5, type=float, help="Discounting factor.")

parser.add_argument("--learning_tau", type=float, default=0.2)
parser.add_argument('--tau_levels', type=int, nargs='?', default=10)

parser.add_argument('--critic_lr_start', type=float, nargs='?', default=0.01)
parser.add_argument('--critic_lr_end', type=float, nargs='?', default=0.001)
parser.add_argument('--actor_lr_start', type=float, nargs='?', default=0.001)
parser.add_argument('--actor_lr_end', type=float, nargs='?', default=0.0001)
parser.add_argument('--rho', type=float, nargs='?', default=0.1)
parser.add_argument('--entropy_reg', type=float, nargs='?', default=0.001)
parser.add_argument('--sigma_start', type=float, nargs='?', default=0.5)

# Args for saving
parser.add_argument('--checkpoints_dir', type=str, nargs='?', default='./training_outcome')

parser.add_argument('--load_weights', type=str2bool, nargs='?', default=False)
parser.add_argument('--load_from_file', type=str, nargs='?', default='test')

parser.add_argument('--save_weights', type=str2bool, nargs='?', default=True)
parser.add_argument('--save_as_file', type=str, nargs='?', default='test')
parser.add_argument('--mode', type=str, nargs='?', default='train')
parser.add_argument('--add_tic_date', type=str2bool, nargs='?', default=False)

# Validation 
parser.add_argument('--min_epochs', type=int, default=300, 
                    help='Minimum number of epochs to train before early stopping is applied.')

parser.add_argument('--patience', type=int, default=5, 
                    help='Number of epochs to wait without improvement in validation loss before early stopping.')

parser.add_argument('--mc', type=str2bool, nargs='?', default=False)

parser.add_argument( '--critic_type', type=str, nargs='?', default='standard', 
    choices=['standard', 'monte_carlo_dropout', 'bayesian','original'],
    help="Type of critic model to use: 'standard', 'monte_carlo_dropout', 'bayesian'"
)
parser.add_argument('--mode_explainer', type=str, nargs='?', default='gradient',choices=['gradient','deep'])
parser.add_argument('--policy_type', type=str, nargs='?', default='risk_neutral')
parser.add_argument( '--actor_loss', type=str, nargs='?', default='advantage', 
    choices=['advantage', 'is_negative', 'weighted_quantile','original','power'],
    help="Type of actor loss to use: 'standard', 'monte_carlo_dropout', 'bayesian'"
)
parser.add_argument('--average_actor_loss', type=str2bool, nargs='?', default=False)


def prepare_eval_df(agent, batch_handler, args):
    train_df = batch_handler.get_train_data()

    actions = agent.get_mean_of_action(
        train_df['state']
    )

    dates_df = pd.DataFrame(train_df["date"], columns=['date'])
    dates_df['date'] = dates_df['date'].dt.date
    price_df = pd.DataFrame(train_df["price_close"], columns=batch_handler.price_names)

    share_names = [f"{c}_share" for c in batch_handler.price_names]
    shares_df = pd.DataFrame(actions, columns=share_names)

    eval_df = pd.concat([dates_df, price_df, shares_df], axis=1)
    eval_df["portfolio_value"] = np.sum(train_df["price_close"] * actions, axis=1)
    eval_df['portfolio_value_change'] = eval_df['portfolio_value'].diff()

    eval_df.to_csv(f"{args.save_as_file}.csv", index=False)

def calculate_total_validation_samples(validation_batches):
    total_samples = sum(len(batch['state']) for batch in validation_batches)
    print(f"Total validation samples: {total_samples}")

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


def main(batch_handler, agent, monitoring, args):
    logging.basicConfig(level=logging.INFO)
    args_string = '- ' + '\n- '.join(f'{k}={v}' for k, v in vars(args).items())

    with logging_redirect_tqdm():
        LOG.info(f'Running with the following args: \n{args_string}')
        LOG.info(f'TensorFlow eager execution mode is {"enabled" if tf.executing_eagerly() else "disabled"}')

        # Fix random seeds and number of threads
        #np.random.seed(args.seed)
        #tf.random.set_seed(args.seed)

        if args.load_weights:
            LOG.info(f'Loading weights from {monitoring.load_path_root}')
            agent.load_weights(monitoring.load_path_root)
        else:
            LOG.info('Starting new training')

        # Training
        batch_handler.switch_mode('train') # initialize the training mode
        batches_train = batch_handler.prepare_batches(args.batch_size)

        batch_handler.switch_mode('validation')
        validation_batches = batch_handler.prepare_batches(args.batch_size)
        batch_handler.switch_mode('train') # switch back cause  monitoring.evaluate_on_training_testing_data() will invoke validation data otherwise

        with open(f'last_run.txt', 'w') as f:
            # sys.stdout = f
            with trange(args.episodes) as t:
                minimum_val_error = float('inf')
                best_epoch = None # if we wanna save differently
                best_model = None
                no_improvement_count= 0
                for ep, _ in enumerate(t):
                    for b, batch in enumerate(batches_train):

                        a_raw, rewards = agent.get_actions_and_rewards(
                            batch['state'],
                            batch['price_close'],
                            batch['price_close_next']
                        )
                        a_raw, v, vn, error, actor_loss, actor_grad, critic_loss, critic_grad = agent.learn(
                            s=batch['state'],
                            sn=batch['state_next'],
                            a_raw=a_raw,
                            r=rewards
                        )


                    #monitoring.save_gradient_magnitudes(actor_grad,ep, 'actor')
                    #monitoring.save_gradient_magnitudes(critic_grad,ep, 'critic')

                    if ep % 100 == 0:
                        mean_v = np.mean(v, axis=0)
                        mean_vn = np.mean(vn, axis=0)
                        mean_error = np.mean(error)
                        std_error = np.std(error)
                        skew_error = skew(error)
                        kurt_error = kurtosis(error)
                        actor_loss = np.mean(actor_loss)
                        critic_loss = np.mean(critic_loss)
                        mean_r = np.mean(rewards)
                        #calculate_total_validation_samples(validation_batches)
                        average_validation_loss, rewards_val = monitoring.validation(validation_batches)
                        t.set_postfix(
                            r=mean_r,
                            v=mean_v[agent.learning_tau_index],
                            v_min=mean_v[0],
                            v_max=mean_v[-1],
                            a_loss=actor_loss,
                            c_loss=critic_loss,
                            #sigma=np.exp(agent.actor_network.log_sigma),  # for Gaussian Policy
                            td_e=mean_error,
                            val_loss=average_validation_loss,
                            val_r=rewards_val
                        )

                        monitoring.update_metric("Value function distribution", mean_v)
                        monitoring.update_metric("Value function distribution (target)", mean_vn)
                        monitoring.update_metric("TD error", mean_error)
                        monitoring.update_metric("Actor loss", actor_loss)
                        monitoring.update_metric("Critic loss", critic_loss)
                        monitoring.update_metric("Average return", mean_r)
                        monitoring.update_metric("Validation return", rewards_val)
                        monitoring.update_metric("Validation loss", average_validation_loss)
                        monitoring.update_metric("std_error_td", std_error)
                        monitoring.update_metric("skew_error_td", skew_error)
                        monitoring.update_metric("kurt_error_td", kurt_error)

                        # early stopping
                        # Warm up
                        # not the best set up but ok
                        if ep>=args.min_epochs:
                            if average_validation_loss < minimum_val_error:
                                minimum_val_error = average_validation_loss
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
                    

    monitoring.save_metrics()
    monitoring.plot_training()
    batch_handler.switch_mode('train')
    monitoring.evaluate_on_training_testing_data()

    #monitoring.generate_shap_summary_plots(batches_train,validation_batches)
    # generate_shap_summary_plots_gradient generate_shap_summary_plots_kernel
    # we should check this on the training also and compare
   
    '''
    
    if args.mc:
        quantile_means, quantile_std, quantile_epistemic_std = monitoring.monte_carlo_epistemic_uncertainty(
        model=agent.critic_network,
        validation_data=batches_train,
        iterations=200
        )
    else:
        LOG.info('Using Standard model, no epidemic uncertainity check')
       '''
   

    monitoring.generate_shap_summary_plots_gradient(batches_train,validation_batches)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Step 1. create the environment
    data_file_etf = './Final_long_etf.csv'
    data_file_exp = './Final_explanatory_Set.csv'
    date_level_exp_list = ['US_leading_index', 'US_coincident_index',
                            'US_lagging_index','US_leadind_credit_index','Close_USD_EUR', 'AVG_PE_SPX', 'Close_vix',
                           'Close_treasury'] # date is included bty default
    # 'US_leading_index', 'US_coincident_index', 'US_lagging_index','US_leadind_credit_index',
    set_global_determinism(seed=args.seed)

    batch_handler = BatchHandler(source_csv_file1 = data_file_etf,source_csv_file2 = data_file_exp,date_level_exp_list = date_level_exp_list,args=args)
    # Step 2. select the model
    agent = QACAgent(state_shape=batch_handler.states[0].shape, stock_dimension=7, args=args)
    #agent = QACDirichletAgent(state_shape=batch_handler.states[0].shape, stock_dimension=7, args=args)

    monitoring = Monitoring(batch_handler=batch_handler, agent=agent, args=args)

    # Step 3. run the main script
    main(batch_handler, agent, monitoring, args)