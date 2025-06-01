import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import shap
import matplotlib.dates as mdates
import logging
from models.qac import get_critic_model
import seaborn as sns

LOG = logging.getLogger()

EPSILON = 0.0000000001


class Monitoring:
    def __init__(self, agent, batch_handler, args):
        self.agent = agent
        self.batch_handler = batch_handler
        self.metrics = {}
        self.args = args
        self.save_path_root = args.checkpoints_dir + '/' + args.save_as_file + '/'
        self.load_path_root = args.checkpoints_dir + '/' + args.load_from_file + '/'
        self.episodic_rewards = {}
        if not os.path.exists(self.save_path_root):
            os.mkdir(self.save_path_root)

        # Set up TensorBoard logging
        #self.log_dir = "logs/gradient_tracking/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.summary_writer = tf.summary.create_file_writer(self.log_dir)
    def update_metric(self, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = [value]
        else:
            self.metrics[metric].append(value)

    def log_episode_reward(self, episode, reward):
        self.episodic_rewards[episode] = reward

    def save_episode_rewards(self, path="episode_rewards.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self.episodic_rewards, f)

    def load_episode_rewards(self, path="episode_rewards.pkl"):
        with open(path, 'rb') as f:
            self.episodic_rewards = pickle.load(f)

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


    def evaluate_on_training_testing_data(self):

        train_df = self.batch_handler.get_train_data()
        # mean of action!
        actions = self.agent.get_mean_of_action(
            train_df['state']
        )
        #print(type(actions))
        #print(actions.shape)
        #actions = actions.numpy() # Data type has changed.

        # dates df
        dates_df = pd.DataFrame(train_df["date"], columns=['date'])
        dates_df['date'] = dates_df['date'].dt.date

        # price df
        price_df = pd.DataFrame(train_df["price_close"], columns=self.batch_handler.price_names)

        # shares df (actions)
        share_names = [f"{c}_share" for c in self.batch_handler.price_names]
        shares_df = pd.DataFrame(actions, columns=share_names)

        # value function
        if self.args.mode=='train':
            value_function = self.agent.critic_network(train_df['state']).numpy()
            value_function_cols = [f"v_{str(tau).replace('.', '')}" for tau in self.agent.tau_levels.numpy()]

            value_df = pd.DataFrame(value_function, columns=value_function_cols)
            value_df['value_function'] = value_df[f'v_{str(self.agent.learning_tau).replace(".", "")}']
        else:
            value_df = pd.DataFrame()

        # portfolio returns
        # TODO: for testing/ validation , should we do sampling or take the mean of actions?
        # get_actions_and_rewards: here you do sampling again, instead of using the raw actions from above  
        '''
        
        sampled_actions, returns = self.agent.get_actions_and_rewards(
            train_df['state'], train_df['price_close'], train_df['price_close_next']
        )

        '''
        return_rate = tf.reduce_sum(((train_df['price_close_next'] /  train_df['price_close']) - 1) * actions, axis=1, keepdims=True) + 1
        wealth_now = tf.math.cumprod(return_rate, axis=0, exclusive=True)
        wealth_next = tf.math.cumprod(return_rate, axis=0, exclusive=False)

        reward = wealth_next - wealth_now
        returns=reward * 84.38 # reward scale,84.38
        
        returns_df = pd.DataFrame(returns, columns=['absolute_return'])
        returns_df['cumulative_return'] = returns_df['absolute_return'].cumsum()
        if self.args.mode =='train':
            eval_df = pd.concat([dates_df, price_df, shares_df, value_df, returns_df], axis=1)
            eval_df.to_csv(self.save_path_root + "train_df_with_shares.csv", index=False)
        else:
            eval_df = pd.concat([dates_df, price_df, shares_df, returns_df], axis=1)
            eval_df.to_csv(self.save_path_root + "test_df_with_shares.csv", index=False)


        self._plot_shares(eval_df)
        self._plot_return_dist(eval_df)
        self._plot_cumul_return(eval_df)
        #self.save_episode_distributions() for later use





        return eval_df

    @staticmethod
    def _plot_shares(df):

        share_cols = [c for c in df.columns if c.endswith('_share')]
        df_shares = df[['date'] + share_cols]
        df_shares.set_index('date', inplace=True)

        # Plot the DataFrame
        ax = df_shares.plot(kind='area', stacked=True, figsize=(14, 6), alpha=0.8)
        ax.set_ylabel('Proportion')
        ax.set_title('Stacked Area Plot of Time Series Data')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
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
        df['date'] = pd.to_datetime(df['date'])

        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['cumulative_return'], linestyle='-', color='b')

        # Set the major locator to MonthLocator to get monthly ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        # Optionally: If you want every N-th month
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) 

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) 

        plt.title('Cumulative Absolute Return Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        
        # Optional: rotate x-axis labels if they overlap
        plt.xticks(rotation=45)
        
        plt.tight_layout()
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
                if metric in ['Validation return', 'Average return', 'Test return','Test loss','Validation loss']:
                    mean_value = np.mean(values)
                    plt.axhline(mean_value, color='r', linestyle='--', label=f'Mean {metric}')
        
            plt.xlabel('Sparse Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric}')
            plt.legend()
            plt.show()

    def save_gradient_magnitudes(self, grad, step, grad_type='actor'):
        if grad_type == 'actor':
            trainable_vars =  self.agent.actor_network.get_trainable_vars()
        else:
            trainable_vars = self.agent.critic_network.trainable_variables

        grad_magnitudes = {}
        for var, grad_tensor in zip(trainable_vars, grad):
            if grad_tensor is not None:
                grad_magnitude = tf.norm(grad_tensor).numpy()
                grad_magnitudes[var.name] = grad_magnitude
                print(f"Saving {grad_type} gradient for {var.name}: {grad_magnitude}")

        with self.summary_writer.as_default():
            for var_name, grad_magnitude in grad_magnitudes.items():
                tf.summary.scalar(f'{grad_type} Gradients/{var_name}', grad_magnitude, step=step)
        self.summary_writer.flush()  

        with self.summary_writer.as_default():
            for var_name, grad_magnitude in grad_magnitudes.items():
                tf.summary.histogram(f'{grad_type}_gradient_distribution/{var_name}', grad_magnitude, step=step)
        self.summary_writer.flush()

    def generate_shap_summary_plots_gradient(self, batches_train, validation_batches):
        '''
        
        source: https://kedion.medium.com/explainable-ai-framework-comparison-97ec0ff04a65
        https://gist.github.com/radi-cho/c75e128ec2c5f503c9eb4c5202e7987d
        https://github.com/shap/shap
        https://github.com/shap/shap/blob/master/notebooks/image_examples/image_classification/Multi-input%20Gradient%20Explainer%20MNIST%20Example.ipynb
        '''
        

        #np.random.seed(42)
        self.agent.merged_state = self.agent.get_merged_states(batches_train)
        features_name = self.batch_handler.loader.final_feature_list

        if self.args.mode_explainer == "gradient":
            LOG.info("Using GradientExplainer with the entire dataset as background.")
            background = self.agent.merged_state  # Use the whole dataset as background
            explainer = shap.GradientExplainer(self.agent.critic_network, background)
        
        elif self.args.mode_explainer == "deep":
  
            state_shape = self.agent.critic_network.input_shape[1:]
            tau_levels = self.agent.critic_network.output_shape[-1]
            self.agent.critic_network = get_critic_model(state_shape, tau_levels)
            weights_path = self.save_path_root + "cn.weights.h5" # TODO:make this more smooth
            LOG.info(f"Loading weights for DeepExplainer from: {weights_path}")
            self.agent.critic_network.load_weights(weights_path)
            
            # Choose a small subset as background
            background_indices = np.random.choice(len(self.agent.merged_state), size=500, replace=False)
            background = self.agent.merged_state[background_indices]
            explainer = shap.DeepExplainer(self.agent.critic_network, background)
        
        else:
            raise ValueError("Invalid mode. Choose 'gradient' or 'deep'.")

        # Prepare instances for explanation
        explanation_instances_list = []
        for batch in validation_batches:
            explanation_instances_list.append(batch['state'])
        explanation_instances = np.concatenate(explanation_instances_list, axis=0)

        # Compute SHAP values
        shap_values = explainer.shap_values(explanation_instances)

        # Generate SHAP summary plots for each quantile
        for q in range(shap_values.shape[2]):
            LOG.info(f"Generating SHAP Summary Plot for Quantile {q + 1} using {self.args.mode_explainer.capitalize()}Explainer.")
            shap_values_for_quantile = shap_values[:, :, q]  # Shape: (instances, features)
            shap.summary_plot(shap_values_for_quantile,
                            explanation_instances, feature_names=features_name, show=False,
                            max_display=10)
            plt.title(f"{self.args.mode_explainer.capitalize()}Explainer Summary Plot - Quantile {q + 1}")
            plt.show()

    
    def validation(self,validation_data):
            '''
            Here, there is no update at the network, only use the already x epochs trained network to validate.
            '''
            validation_loss = 0.0
            total_rewards = 0.0
            #num_batches = len(validation_data)
            total_samples = sum(len(batch['state']) for batch in validation_data) # Normalize with total sample
            # lets validate with stochastic process. Not the best
            for batch in validation_data:
                a_raw, rewards = self.agent.get_actions_and_rewards(
                                batch['state'],
                                batch['price_close'],
                                batch['price_close_next']
                            )
                
                s = batch['state']
                sn = batch['state_next']
                r = rewards
                vn = self.agent.critic_target(sn)

                y = r + self.agent.gamma * vn
                increased_order_loss_weight = 2.0
                v = self.agent.critic_network(s)
                error = y - v

                abs_error = tf.math.abs(error)
                is_negative = tf.where(tf.math.less(error, 0.0), 1.0, 0.0)
                q_order_loss = tf.reduce_mean(tf.maximum(0.0, v[:, :-1] - v[:, 1:] + EPSILON))*increased_order_loss_weight
                loss = tf.math.multiply(tf.math.abs(tf.math.subtract(self.agent.tau_levels, is_negative)), abs_error)
                critic_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)+q_order_loss

                
                #error = error[:,self.agent.learning_tau_index: self.agent.learning_tau_index + 1]
                #is_negative = tf.where(tf.math.less(error, 0.0), 1.0, 0.0)
                scale = 1
                if self.agent.args.actor_loss=='weighted_quantile':
                    t = self.agent.tau_levels[self.agent.learning_tau_index]
                    error = error[:, self.agent.learning_tau_index:self.agent.learning_tau_index + 1]
                    #is_negative = tf.where(error < 0.0, 1.0, 0.0)
                    #t_weight = t - is_negative
                    t_weight = tf.where(error < 0.0, 1.0 - t, t)

                elif self.agent.args.actor_loss == 'advantage':
                    error = error[:, self.agent.learning_tau_index: self.agent.learning_tau_index + 1]
                elif self.agent.args.actor_loss == 'is_negative':
                    error = error[:, self.agent.learning_tau_index: self.agent.learning_tau_index + 1]
                    is_negative = tf.where(tf.math.less(error, 0.0), 1.0, 0.0)
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
                log_prob, entropy = self.agent.actor_network.get_log_action_prob(state=s, sampled_raw_actions=a_raw)

                if self.agent.args.actor_loss=='is_negative':   
                    actor_loss = tf.reduce_mean(tf.reduce_sum(log_prob * scale * is_negative, axis=1))- self.agent.args.entropy_reg * entropy
                elif self.agent.args.actor_loss=='weighted_quantile':
                    actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * error * t_weight, axis=1)) - self.agent.args.entropy_reg * entropy
                elif self.agent.args.actor_loss=='advantage':
                    actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * error, axis=1)) - self.agent.args.entropy_reg * entropy

                elif self.agent.args.actor_loss =='expectation':
                    actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * avg_error, axis=1))
                elif self.agent.args.actor_loss =='power':
                    actor_loss = -tf.reduce_mean(tf.reduce_sum(log_prob * error*weight_pow, axis=1))

                losses = actor_loss + critic_loss
                total_rewards +=tf.reduce_sum(rewards).numpy()
                validation_loss += losses.numpy()
            average_validation_loss = validation_loss / total_samples # num_batches: not good for normalization
            average_rewards_total = total_rewards / total_samples

            return average_validation_loss, average_rewards_total
        
        
    def monte_carlo_epistemic_uncertainty_original(self, model, validation_data, iterations=200):
        '''
        If mc= True, we use monte carlo droput
        If False, we use bayesian layers. With bayesian layers, we cannot use SHAP gradient method only kernel
        '''
        all_estimations = [] 
        for batch in validation_data:
            s = batch['state'] 
            batch_estimations = []

            for _ in range(iterations):
                if self.args.mc==True:
                    predictions = model(s, training=True).numpy()  # training=True is important: this makes droput and sampling open
                else:
                     predictions = model(s, training=False).numpy() # bayesian by default using sampling, i thin kwe dont need True
                batch_estimations.append(predictions)  # Shape: (batch_size, tau_levels)

            batch_estimations = np.array(batch_estimations)  # Shape: (iterations, batch_size, tau_levels)
            all_estimations.append(batch_estimations)


        all_estimations = np.concatenate(all_estimations, axis=1)  # Shape: (iterations, total_samples, tau_levels)

        quantile_means = all_estimations.mean(axis=(0, 1))   
        quantile_std = all_estimations.std(axis=(0, 1))  

        quantiles = range(len(quantile_means)) 
        smallest_std_indices = np.argsort(quantile_std)[:2]
        largest_std_indices = np.argsort(quantile_std)[-2:]

        plt.figure(figsize=(8, 6))

        plt.errorbar(quantiles, quantile_means, yerr=quantile_std, fmt='-o', capsize=5, color='skyblue', label='Mean ± Std Dev')

        for idx in smallest_std_indices:
            plt.errorbar(quantiles[idx], quantile_means[idx], yerr=quantile_std[idx], fmt='o', capsize=5,
                        color='green', label='Smallest Std Dev' if idx == smallest_std_indices[0] else "")

        for idx in largest_std_indices:
            plt.errorbar(quantiles[idx], quantile_means[idx], yerr=quantile_std[idx], fmt='o', capsize=5,
                        color='red', label='Largest Std Dev' if idx == largest_std_indices[0] else "")

        for i, (mean, std) in enumerate(zip(quantile_means, quantile_std)):
            plt.text(i, mean + std + 0.02, f"Std={std:.3f}", fontsize=10, ha='center', color='black')

        title = "Quantile Predictions with Epistemic Uncertainty (MC Dropout)" if self.args.mc else "Quantile Predictions with Epistemic Uncertainty (Bayesian Layer)"

        plt.xlabel("Quantiles (Tau Levels)", fontsize=12)
        plt.ylabel("Predicted mean Quantiles", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.show()

        return quantile_means, quantile_std
    

    def monte_carlo_epistemic_uncertainty(self, model, validation_data, iterations=200, window_size=60):
        """
        Monte Carlo Epistemic Uncertainty Estimation with full visualization.
        - Returns overall quantile means and total std (epistemic + aleatoric)
        """
        all_estimations = [] 
        for batch in validation_data:
            s = batch['state'] 
            batch_estimations = []

            for i in range(iterations):
                tf.random.set_seed(self.args.seed + i)
                if self.args.mc:
                    predictions = model(s, training=True).numpy()
                else:
                    predictions = model(s, training=False).numpy()
                batch_estimations.append(predictions)

            batch_estimations = np.array(batch_estimations)  # (iterations, batch_size, quantiles)
            all_estimations.append(batch_estimations)
        # we have full batch!
        #all_estimations = np.concatenate(all_estimations, axis=1)  # (iterations, total_samples, quantiles)

        # Total variability: epistemic + aleatoric: meaning? difficult to interpret
        quantile_means = batch_estimations.mean(axis=(0, 1))     
        quantile_std = batch_estimations.std(axis=(0, 1))        
        quantiles = list(range(len(quantile_means)))

        smallest_std_indices = np.argsort(quantile_std)[:2]
        largest_std_indices = np.argsort(quantile_std)[-2:]

        plt.figure(figsize=(8, 6))
        plt.errorbar(quantiles, quantile_means, yerr=quantile_std, fmt='-o', capsize=5, color='skyblue', label='Mean ± Std Dev')

        for idx in smallest_std_indices:
            plt.errorbar(quantiles[idx], quantile_means[idx], yerr=quantile_std[idx], fmt='o', capsize=5,
                        color='green', label='Smallest Std Dev' if idx == smallest_std_indices[0] else "")
        for idx in largest_std_indices:
            plt.errorbar(quantiles[idx], quantile_means[idx], yerr=quantile_std[idx], fmt='o', capsize=5,
                        color='red', label='Largest Std Dev' if idx == largest_std_indices[0] else "")

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

        #Pure epistemic uncertainty
        epistemic_std = np.std(batch_estimations, axis=0)  # (samples, quantiles)
        quantile_epistemic_std = np.mean(epistemic_std, axis=0)  # (quantiles,)

        print("\nAverage Epistemic Std per Quantile:")
        print(pd.DataFrame({
            'Quantile': quantiles,
            'Epistemic Std': quantile_epistemic_std
        }))

        # Rolling heatmap of epistemic uncertainty: over examination period: measuring shifts
        if epistemic_std.shape[0] >= window_size:
                dates = np.array(batch['date']).flatten()

                rolling_uncertainty = []
                rolling_dates = []

                for i in range(epistemic_std.shape[0] - window_size + 1): # we loose the last one if not +1 cause right no..
                    window = epistemic_std[i:i + window_size]# right no
                    window_mean = np.mean(window, axis=0)
                    rolling_uncertainty.append(window_mean)

                    date_label = pd.to_datetime(str(dates[i + window_size - 1])).to_period("Q").strftime('%Y-Q%q')
                    rolling_dates.append(date_label)

                rolling_uncertainty = np.array(rolling_uncertainty)

                plt.figure(figsize=(12, 6))
                ax = sns.heatmap(rolling_uncertainty, cmap="viridis", cbar_kws={'label': 'Rolling Epistemic Std'})

                ax.set_xlabel("Quantile Index (τ)")
                ax.set_ylabel("Quarter")
                ax.set_title(f"Epistemic Uncertainty (Rolling Mean over {window_size} Days)")

                max_labels = 20 # only every 20th
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

        background_indices = np.random.choice(len(self.agent.merged_state), size=700, replace=False)
        background = self.agent.merged_state[background_indices]
        summarized_background = shap.kmeans(background, 50)

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


        kernel_explainer = shap.KernelExplainer( self.agent.critic_network, summarized_background)

        explanation_instances_list = []
        for batch in validation_batches:
            explanation_instances_list.append(batch['state'])
        explanation_instances = np.concatenate(explanation_instances_list, axis=0)  # Combine all batches
        #explanation_instances = explanation_instances[:10] # very slow
        kernel_shap_values = kernel_explainer.shap_values(explanation_instances)

        # Step 7: Plot SHAP summary for each tau level
        for q in range(kernel_shap_values.shape[2]):  # Iterate over tau levels
            LOG.info(f"Generating Kernel SHAP Summary Plot for Quantile {q + 1}")
            shap_values_for_quantile = kernel_shap_values[:, :, q]
            shap.summary_plot(shap_values_for_quantile, 
                              explanation_instances, 
                              feature_names=features_name, 
                              show=False,
                              max_display=10)
            plt.title(f"Kernel SHAP Summary Plot - Quantile {q + 1}")
            plt.show()
