import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from scipy.stats import wasserstein_distance
import seaborn as sns

EPSILON = 0.0000000001


class Monitoring:
    def __init__(self, agent,env, args):
        self.agent = agent
        self.env = env
        self.metrics = {}
        self.args = args
        self.save_path_root = args.checkpoints_dir + '/' + args.save_as_file + '/'
        self.load_path_root = args.checkpoints_dir + '/' + args.load_from_file + '/'
        self.args = args
        os.makedirs(self.save_path_root, exist_ok=True)

        # Set up TensorBoard logging
        self.log_dir = "logs/gradient_tracking/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
    def update_metric(self, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = [value]
        else:
            self.metrics[metric].append(value)

    def save_metrics(self):
        file_name = self.save_path_root + "metrics.pkl"
        with open(file_name, 'wb') as handle:
            pickle.dump(self.metrics, handle)

    def save_brain_weights(self):
        self.agent.save_weights(self.save_path_root)

    def save_episode_distributions(self):
    # Save the entire episode_data. We wanna track later the episodic weights distribution and rewards improvement 
        save_path = os.path.join(self.save_path_root, 'episode_data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.episode_data, f)

    def evaluate_on_training_testing_data_stepwise(self):
            current_mode = self.env.mode

            state = self.env.reset()
            terminal = False

            dates = []
            prices = []
            actions_list = []
            value_functions = []
            absolute_returns = []
            transaction_costs = []

            while not terminal:
                mean_action = self.agent.get_mean_of_action(state)
                dates.append(self.env.data['date'].iloc[0].date())
                prices.append(self.env.data['close'].values)
                actions_list.append(mean_action.numpy().flatten())

                value_function = self.agent.critic_network(state).numpy()
                value_functions.append(value_function.flatten())

                next_state, reward, terminal, t_cost_eval = self.env.step_evaluate(mean_action)
                absolute_returns.append(reward)
                transaction_costs.append(t_cost_eval)

                state = next_state

            dates_df = pd.DataFrame(dates, columns=['date'])
            prices_df = pd.DataFrame(prices, columns=self.env.df.loc[0].tic.unique())

            share_names = [f"{c}_share" for c in self.env.df.loc[0].tic.unique()]
            shares_df = pd.DataFrame(actions_list, columns=share_names)

            value_function_cols = [f"v_{str(tau).replace('.', '')}" for tau in self.agent.tau_levels.numpy()]
            value_df = pd.DataFrame(value_functions, columns=value_function_cols)
            if current_mode == 'train':
                value_df['value_function'] = value_df[f"v_{str(self.agent.learning_tau).replace('.', '')}"]

            abs_ret_df = pd.DataFrame(absolute_returns, columns=['absolute_return'])
            abs_ret_df['cumulative_return'] = (1.0 + abs_ret_df['absolute_return'] / self.env.reward_scaling).cumprod()

            cost_df = pd.DataFrame({'t_cost_eval': transaction_costs})

            if current_mode == 'train':
                eval_df = pd.concat([dates_df, prices_df, shares_df, value_df, abs_ret_df, cost_df], axis=1)
                filename = "train_df_with_shares.csv"
            else:
                eval_df = pd.concat([dates_df, prices_df, shares_df, abs_ret_df, cost_df], axis=1)
                if getattr(self.args, 'chunk_training', False):
                    chunk_idx = getattr(self.args, 'chunk_idx', 0)
                    filename = f"test_df_with_shares_chunk{chunk_idx:02d}.csv"
                else:
                    filename = "test_df_with_shares.csv"

            eval_path = os.path.join(self.save_path_root, filename)
            eval_df.to_csv(eval_path, index=False)

            self._plot_shares(eval_df)
            self._plot_return_dist(eval_df)
            self._plot_cumul_return(eval_df)

            return eval_df

    def stepwise_validation(self):
        """
        Evaluate policy episodically over the full validation trajectory, using a single loss computation at the end.
        """
        state = self.env.reset()
        terminal = False
        total_rewards = 0.0
        total_transaction_cost = 0.0
        transitions = []

        while not terminal:
            raw_action = self.agent.get_mean_of_action(state)  # mean-based for deterministic eval
            next_state, reward, terminal, t_cost = self.env.step_evaluate(raw_action)
            state_np = tf.squeeze(state).numpy()  # shape: (40,)
            next_state_np = tf.squeeze(next_state).numpy()
            raw_action_np = tf.squeeze(raw_action).numpy()
            reward_np = float(reward)
            transitions.append((state_np, next_state_np, raw_action_np, reward_np))
            total_rewards += reward
            total_transaction_cost += t_cost
            state = next_state

        #  unpack and process
        states, next_states, a_raws, rewards = zip(*transitions)
        s = tf.convert_to_tensor(states, dtype=tf.float32)
        sn = tf.convert_to_tensor(next_states, dtype=tf.float32)

        a_raw = tf.convert_to_tensor(a_raws, dtype=tf.float32)
        r = tf.convert_to_tensor(rewards, dtype=tf.float32)
     
        r = tf.reshape(r, (-1, 1))        




        vn = self.agent.critic_target(sn)
        v = self.agent.critic_network(s)
        y = r + self.agent.gamma * vn
        error = y - v
        increased_order_loss_weight = 5.0  # 5.0

        abs_error = tf.abs(error)
        is_negative = tf.where(tf.math.less(error, 0.0), 1.0, 0.0)
        q_order_loss = tf.reduce_mean(tf.maximum(0.0, v[:, :-1] - v[:, 1:] + EPSILON)) * increased_order_loss_weight
        loss = tf.math.multiply(tf.math.abs(tf.math.subtract(self.agent.tau_levels, is_negative)), abs_error)

        critic_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)+q_order_loss

        log_prob, entropy = self.agent.actor_network.get_log_action_prob(s, a_raw)
        #t_index = self.agent.learning_tau_index
        #error_t = error[:, t_index:t_index+1]
        t = self.agent.tau_levels[self.agent.learning_tau_index]
        error = error[:, self.agent.learning_tau_index:self.agent.learning_tau_index + 1]

        if self.args.actor_loss == 'weighted_quantile':

            t_weight = tf.where(error < 0.0, 1.0 - t, t)
            actor_loss = -tf.reduce_sum(log_prob * error * t_weight*10, axis=1) #- self.args.entropy_reg * entropy

        elif self.args.actor_loss == 'advantage':
            actor_loss = -tf.reduce_sum(log_prob * error, axis=1) #- self.args.entropy_reg * entropy

        elif self.args.actor_loss == 'is_negative':
            is_negative = tf.where(error < 0.0, 1.0, 0.0)
            actor_loss = tf.reduce_sum(log_prob * is_negative, axis=1) #- self.args.entropy_reg * entropy

        mean_actor_loss = tf.reduce_mean(actor_loss)- self.args.entropy_reg * entropy
        validation_loss = mean_actor_loss + critic_loss

        w_dist = wasserstein_distance(v.numpy().flatten(), y.numpy().flatten())

        avg_reward = total_rewards / len(transitions)
        avg_cost = total_transaction_cost / len(transitions)
        return avg_reward, validation_loss.numpy(), avg_cost, w_dist

    @staticmethod
    def _plot_shares(df):
        # Assuming 'df' is your DataFrame and is already loaded with data

        # Set the Date column as the index
        share_cols = [c for c in df.columns if c.endswith('_share')]
        df_shares = df[['date'] + share_cols]
        df_shares.set_index('date', inplace=True)

        # Plot the DataFrame
        ax = df_shares.plot(kind='area', stacked=True, figsize=(14, 6), alpha=0.8)
        ax.set_ylabel('Proportion')
        ax.set_title('Stacked Area Plot of Time Series Data')

        # Adjusting the legend to be outside the plot but inside the figure
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right space to make room for the legend
        plt.show()

    @staticmethod
    def _plot_return_dist(df):
        plt.figure(figsize=(10, 6))
        plt.hist(df['absolute_return'], bins=80, edgecolor='black', alpha=0.7)
        plt.title('Distribution of absolute returns')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def _plot_cumul_return(df):
        # Plot the column 'Value' over time

        idx = df['date']
        plt.figure(figsize=(10, 6))
        plt.plot(idx, df['cumulative_return'], linestyle='-', color='b')
        plt.title('Cumulative absolute return over time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    @staticmethod
    def _plot_cumul_transaction_cost(df):
        # Plot the column 'Value' over time

        idx = df['date']
        plt.figure(figsize=(10, 6))
        plt.plot(idx, df['cumulative_tranaction_cost'], linestyle='-', color='b')
        plt.title('Cumulative transaction cost over time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    def plot_training(self):
        for metric, values in self.metrics.items():
            plt.figure()
            if isinstance(values[0], np.ndarray):  # Check if the first item is a numpy array
                array_length = values[0].shape[0]
                for i in range(array_length):
                    # Extract the i-th element from each numpy array in values
                    element_values = [value[i] for value in values]
                    plt.plot(element_values, label=f'{metric}[{i}]')
            else:
                plt.plot(values, label=metric)
                if metric in ['Validation return', 'Average return', 'Test return','Test loss','Validation loss',"Validation transaction cost"]:
                    mean_value = np.mean(values)
                    plt.axhline(mean_value, color='r', linestyle='--', label=f'Mean {metric}')
        
            plt.xlabel('Sparse Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric}')
            plt.legend()
            plt.show()

   
   