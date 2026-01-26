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
        if not os.path.exists(self.save_path_root):
            os.mkdir(self.save_path_root)

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
      

        state = self.env.reset()
        terminal = False

        dates = []
        prices = []
        actions_list = [] # NOTE:in the step evaulat we also save it. 
        value_functions = []
        absolute_returns = []
        #cumulative_returns = []
        cumulative_tranaction_cost = []
        total_return = 0
        total_tranaction_cost = 0
        transitions= []
        while not terminal:
            # Calculate action based on the current state
            #raw_action = self.act(state)
            # lets take the mean everywhere
            mean_action = self.agent.get_mean_of_action(state) # dont be stochastic here, just take the mean of the actions
            dates.append(self.env.data['date'].iloc[0].date())
            prices.append(self.env.data['close'].values) # this is return in this set up
            # Track actions
            actions_list.append(mean_action.numpy().flatten())


            # Get value function estimate
            value_function = self.agent.critic_network(state).numpy()
            value_functions.append(value_function.flatten())

            # Take a step in the environment
            next_state, reward, terminal,t_cost_eval = self.env.step_evaluate(mean_action)
            #transitions.append((state, next_state, mean_action, reward))

            # Track returns
            total_return += reward
            total_tranaction_cost +=t_cost_eval
            absolute_returns.append(reward)
            #cumulative_returns.append(total_return)
            cumulative_tranaction_cost.append(total_tranaction_cost)
            

            # Update state for the next step
            state = next_state

        # Prepare DataFrames for output
        dates_df = pd.DataFrame(dates, columns=['date'])
        prices_df = pd.DataFrame(prices, columns=self.env.df.loc[0].tic.unique())#
        
        # Shares (actions)
        share_names = [f"{c}_share" for c in self.env.df.loc[0].tic.unique()]
        shares_df = pd.DataFrame(actions_list, columns=share_names)
        
        # Value function data
        value_function_cols = [f"v_{str(tau).replace('.', '')}" for tau in self.agent.tau_levels.numpy()]
        value_df = pd.DataFrame(value_functions, columns=value_function_cols)
        if self.args.mode == 'train':
            value_df['value_function'] = value_df[f'v_{str(self.agent.learning_tau).replace(".", "")}']
        
        # Returns data
        abs_ret_df = pd.DataFrame(absolute_returns, columns=['absolute_return'])
        #abs_ret_df = pd.DataFrame(absolute_returns, columns=['log_growth'])
        #abs_ret_df['cum_log_growth'] = (abs_ret_df['log_growth'] /self.env.reward_scaling).cumsum()


        abs_ret_df['cumulative_return'] = (1.0 + abs_ret_df['absolute_return']/self.env.reward_scaling).cumprod() * 1 # Start at $10,000
        #abs_ret_df['cumulative_return'] = np.exp(abs_ret_df['cum_log_growth']) - 1.0
        #abs_ret_df['cumulative_wealth'] = self.env.initial_amount * np.exp(abs_ret_df['cum_log_growth'])
        #returns_df = pd.DataFrame({'absolute_return': absolute_returns, 'cumulative_return': cumulative_returns,'cumulative_tranaction_cost': cumulative_tranaction_cost})

        # Combine all data into a single DataFrame
        if self.args.mode == 'train':
            eval_df = pd.concat([dates_df, prices_df, shares_df, value_df, abs_ret_df], axis=1)
            eval_df.to_csv(self.save_path_root + "train_df_with_shares.csv", index=False)
        else:
            eval_df = pd.concat([dates_df, prices_df, shares_df, abs_ret_df], axis=1)
            eval_df.to_csv(self.save_path_root + "test_df_with_shares.csv", index=False)

        # Additional metrics

        # Generate plots
        self._plot_shares(eval_df)
        self._plot_return_dist(eval_df)
        self._plot_cumul_return(eval_df)
        #self._plot_cumul_transaction_cost(eval_df)

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

   
   

    def monte_carlo_epistemic_uncertainty_original(self, iterations=200):

        state = self.env.reset() 
        terminal = False
        total_rewards = 0.0
        step_count = 0

        quantile_means_list = []
        quantile_std_list = []

        while not terminal:
            raw_action = self.agent.act(state) # use the sampled version
            next_state, reward, terminal,t_cost = self.env.step(raw_action)


            s = state

            all_estimations = []
            for _ in range(iterations):
                tf.random.set_seed(self.args.seed + i)
                if self.args.mc:
                    # Monte Carlo Dropout: Enable dropout layers at prediction time
                    predictions = self.agent.critic_network(s, training=True).numpy()  # Shape: (1, tau_levels)
                else:
                    # Bayesian Layers: Use model's inherent sampling mechanism
                    predictions = self.agent.critic_network(s, training=False).numpy() # Shape: (1, tau_levels)
                all_estimations.append(predictions)
            
            all_estimations = np.array(all_estimations).squeeze(axis=1)  # Shape: (iterations, tau_levels)
            quantile_means = all_estimations.mean(axis=0)                # Shape: (tau_levels,)
            quantile_std = all_estimations.std(axis=0)                  # Shape: (tau_levels,)

            quantile_means_list.append(quantile_means)
            quantile_std_list.append(quantile_std)

            # Update state
            state = next_state

        # Convert lists to arrays
        quantile_means_array = np.array(quantile_means_list)  # Shape: (steps, tau_levels)
        quantile_std_array = np.array(quantile_std_list)      # Shape: (steps, tau_levels)

        # Aggregate by averaging across steps
        aggregated_quantile_means = quantile_means_array.mean(axis=0)  # Shape: (tau_levels,)
        aggregated_quantile_std = quantile_std_array.mean(axis=0)      # Shape: (tau_levels,)

        quantiles = range(len(aggregated_quantile_means)) 

        # Identify indices of quantiles with smallest and largest uncertainties
        smallest_std_indices = np.argsort(aggregated_quantile_std)[:2]
        largest_std_indices = np.argsort(aggregated_quantile_std)[-2:]

        # Plotting Epistemic Uncertainty
        plt.figure(figsize=(8, 6))

        plt.errorbar(
            quantiles, 
            aggregated_quantile_means, 
            yerr=aggregated_quantile_std, 
            fmt='-o', 
            capsize=5, 
            color='skyblue', 
            label='Mean ± Std Dev'
        )

        for idx in smallest_std_indices:
            plt.errorbar(
                quantiles[idx], 
                aggregated_quantile_means[idx], 
                yerr=aggregated_quantile_std[idx], 
                fmt='o', 
                capsize=5,
                color='green', 
                label='Smallest Std Dev' if idx == smallest_std_indices[0] else ""
            )

        for idx in largest_std_indices:
            plt.errorbar(
                quantiles[idx], 
                aggregated_quantile_means[idx], 
                yerr=aggregated_quantile_std[idx], 
                fmt='o', 
                capsize=5,
                color='red', 
                label='Largest Std Dev' if idx == largest_std_indices[0] else ""
            )

        for i, (mean, std) in enumerate(zip(aggregated_quantile_means, aggregated_quantile_std)):
            plt.text(
                i, 
                mean + std + 0.02, 
                f"Std={std:.3f}", 
                fontsize=10, 
                ha='center', 
                color='black'
            )

        # Set plot title based on method
        title = "Quantile Predictions with Epistemic Uncertainty (MC Dropout)" if self.args.mc else "Quantile Predictions with Epistemic Uncertainty (Bayesian Layer)"
        plt.title(title, fontsize=14)
        plt.xlabel("Quantiles (Tau Levels)", fontsize=12)
        plt.ylabel("Predicted Mean Quantiles", fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        return quantile_means, quantile_std
    def monte_carlo_epistemic_uncertainty(self, iterations=20, window_size=60):
        """
        Stepwise Monte Carlo Epistemic Uncertainty Estimation (iterations x steps x quantiles).
        Returns quantile means, total std, and epistemic std with rolling heatmap.
        """

      
        all_estimations = []  # Shape: (iterations, steps, quantiles)
        step_dates = [] # i == to avoid multi dates

        for i in range(iterations):
            tf.random.set_seed(self.args.seed + i) # random initialization fopr each run!

            state = self.env.reset()
            terminal = False
            iter_predictions = [] # sample,quantile

            while not terminal:
                if i == 0:
                    step_dates.append(self.env.data['date'].iloc[0].date())

                s = state
                if self.args.mc:
                    predictions = self.agent.critic_network(s, training=True).numpy().squeeze(0)  # (quantiles,)
                else:
                    predictions = self.agent.critic_network(s, training=False).numpy().squeeze(0) # bayesian

                iter_predictions.append(predictions)

                raw_action = self.agent.act(state)
                next_state, reward, terminal, t_cost = self.env.step(raw_action)
                state = next_state

            all_estimations.append(iter_predictions)

        # Convert to (iterations, steps, quantiles)
        all_estimations = np.array(all_estimations)

        # Aggregate
        quantile_means = all_estimations.mean(axis=(0, 1))  # shape: (quantiles,)
        quantile_std = all_estimations.std(axis=(0, 1))     # total uncertainty: over iteration and data
        epistemic_std_array = np.std(all_estimations, axis=0)  # shape: (steps, quantiles): true epistemic uncertainity. At each day we check the std
        quantile_epistemic_std = epistemic_std_array.mean(axis=0) # mean over the sample

        # Plot: Mean ± Std Dev
        smallest_std_indices = np.argsort(quantile_std)[:2]
        largest_std_indices = np.argsort(quantile_std)[-2:]


        quantiles = list(range(len(quantile_means)))

        plt.figure(figsize=(8, 6))
        plt.errorbar(quantiles, quantile_means, yerr=quantile_std, fmt='-o',
                    capsize=5, color='skyblue', label='Mean ± Std Dev')

        for idx in smallest_std_indices:
            plt.errorbar(quantiles[idx], quantile_means[idx], yerr=quantile_std[idx],
                        fmt='o', capsize=5, color='green',
                        label='Smallest Std Dev' if idx == smallest_std_indices[0] else "")
        for idx in largest_std_indices:
            plt.errorbar(quantiles[idx], quantile_means[idx], yerr=quantile_std[idx],
                        fmt='o', capsize=5, color='red',
                        label='Largest Std Dev' if idx == largest_std_indices[0] else "")

        for i, (mean, std) in enumerate(zip(quantile_means, quantile_std)):
            plt.text(i, mean + std + 0.02, f"Std={std:.3f}", fontsize=9, ha='center')

        title = "Quantile Predictions with Epistemic Uncertainty (MC Dropout)" if self.args.mc else "Quantile Predictions with Epistemic Uncertainty (Bayesian Layer)"
        plt.xlabel("Quantile Index (τ)")
        plt.ylabel("Predicted Mean Quantile Value")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print table of epistemic std
        print("\nAverage Epistemic Std per Quantile:")
        print(pd.DataFrame({
            'Quantile': quantiles,
            'Epistemic Std': quantile_epistemic_std
        }))

        # Rolling Epistemic Uncertainty Heatmap
        if len(epistemic_std_array) >= window_size:
            rolling_uncertainty = []
            rolling_dates = []

            for i in range(len(epistemic_std_array) - window_size + 1):
                window = epistemic_std_array[i:i + window_size]
                window_mean = np.mean(window, axis=0)
                rolling_uncertainty.append(window_mean)

                if step_dates:
                    date_label = pd.to_datetime(str(step_dates[i + window_size - 1])).to_period("Q").strftime('%Y-Q%q')
                else:
                    date_label = f"Step {i + window_size - 1}"
                rolling_dates.append(date_label)

            rolling_uncertainty = np.array(rolling_uncertainty)

            plt.figure(figsize=(12, 6))
            ax = sns.heatmap(rolling_uncertainty, cmap="viridis", cbar_kws={'label': 'Rolling Epistemic Std'})
            ax.set_xlabel("Quantile Index (τ)")
            ax.set_ylabel("Quarter" if step_dates else "Step")
            ax.set_title(f"Epistemic Uncertainty (Rolling Mean over {window_size} Steps)")

            # Fewer y-ticks cause we wont see anything
            max_labels = 20
            n_labels = len(rolling_dates)
            step = max(1, n_labels // max_labels)
            ax.set_yticks(np.arange(0, n_labels, step))
            ax.set_yticklabels(rolling_dates[::step], rotation=45, ha='right', fontsize=8)

            plt.tight_layout()
            plt.show()

        return quantile_means, quantile_std, quantile_epistemic_std

    def generate_shap_summary_plots_kernel(self, batches_train, validation_batches, iterations=50):
        """
        Generate SHAP summary plots using Kernel Explainer for quantile predictions.

        """
        np.random.seed(42)
        
        self.agent.merged_state = self.agent.get_merged_states(batches_train)
        features_name = self.batch_handler.loader.final_feature_list

        background_indices = np.random.choice(len(self.agent.merged_state), size=500, replace=False)
        background = self.agent.merged_state[background_indices]

        def predict_with_uncertainty(inputs):
            """
            Wrapper for SHAP to handle stochastic forward passes for Bayesian models.
            Returns mean predictions across multiple iterations.
            self.agent.critic_network: gave an error, we may use training=False
            with this we reflect randomness a little bit, not sure...
            with gradient explainer it didnt work the bayesian layer.

            """
            stochastic_predictions = []
            for _ in range(iterations):
                preds = self.agent.critic_network(inputs, training=True).numpy()  # Enable stochasticity in Bayesian
                stochastic_predictions.append(preds)
            stochastic_predictions = np.array(stochastic_predictions)
            return stochastic_predictions.mean(axis=0)  # Mean predictions (batch_size, tau_levels)

        kernel_explainer = shap.KernelExplainer(predict_with_uncertainty, background)

        explanation_instances = []
        for batch in validation_batches:
            explanation_instances.append(batch['state'])
        explanation_instances = np.concatenate(explanation_instances, axis=0)  # Combine all batches
        explanation_instances = explanation_instances[:10] # very slow
        kernel_shap_values = kernel_explainer.shap_values(explanation_instances)

        # Step 7: Plot SHAP summary for each tau level
        for q in range(len(kernel_shap_values)):  # Iterate over tau levels
            LOG.info(f"Generating Kernel SHAP Summary Plot for Quantile {q + 1}")
            shap.summary_plot(kernel_shap_values[q], 
                              explanation_instances, 
                              feature_names=features_name, 
                              show=False,
                              max_display=10)
            plt.title(f"Kernel SHAP Summary Plot - Quantile {q + 1}")
            plt.show()


