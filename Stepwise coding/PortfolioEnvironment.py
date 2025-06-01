
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from functools import reduce


'''
Create a custom environment with functions
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from Methods import calculate_roc_per_ticker
class LoaderEnginering:
    def __init__(self, data_file1,data_file2,date_level_exp_list,args):

        self.args = args
        self.data_etf = pd.read_csv(data_file1)
        self.data_exp = pd.read_csv(data_file2)
        self.data_etf['date'] = pd.to_datetime(self.data_etf['date'])
        self.data_etf = self.data_etf[self.data_etf['tic'].isin(['min_vol', 'momentum', 'quality'])]

        self.data_exp['date'] = pd.to_datetime(self.data_exp['date'])
        data_exp_selected = self.data_exp[['date'] + date_level_exp_list]
        self.data = pd.merge(self.data_etf, data_exp_selected, on='date', how='inner')


        self.train_df = None
        self.test_df = None
        self.final_feature_list = None
        self.tic_date = None
        self.final_feature_list_scaled = None
        self.balance_memory = []


    def _add_features_technical(self):
        data = self.data
        roc = calculate_roc_per_ticker(data)
        #sma = calculate_sma_per_ticker(data)
        #bb = calculate_bollinger_bands(data)
        dataframes = [roc]
        self.tic_date = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dataframes)

        data = data.reset_index(drop=True)
        
        return data    

    def _add_cov_matrix(self, lookback, df):
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df.date.factorize()[0]

        cov_list = [] 

        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i-1, :]  # Pandas includes right end as well, very weird
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            covs = return_lookback.cov().values

            upper_tri_indices = covs[np.triu_indices(covs.shape[0])] # Get the upper  triangular values + diagonal elements. covs.shape[0] means the number of diagonal elements as I understood
            row = {f"cov_{lookback}_{i}": cov_value for i, cov_value in enumerate(upper_tri_indices)} 

            date = df.date.unique()[i]
            row['date'] = date

            cov_list.append(row)

        df_cov = pd.DataFrame(cov_list)
        df = df.merge(df_cov, on='date', how='inner')
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)

        return df
    

    def _get_cov_df(self, lookback, df):
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df.date.factorize()[0]
        rename_dict = {
                    'min_vol': 'MV',
                    'momentum': 'M',
                    'quality': 'Q'
                    }

        cov_list = []  # create empty list for storing coveriance matrices at each time step

        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i-1, :]  # Pandas includes right end as well, very weird
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            price_lookback = price_lookback.rename(columns=rename_dict)
            return_lookback = price_lookback.pct_change().dropna()
            covs = return_lookback.cov()# cov
            n = covs.shape[0] # getting the indexes
            upper_tri_row_idx, upper_tri_col_idx = np.triu_indices(n)

            row = {}
            tickers = covs.columns
            for r, c in zip(upper_tri_row_idx, upper_tri_col_idx):
                cov_value = covs.iloc[r, c]
                cov_key = f"cov_{lookback}_{tickers[r]}_{tickers[c]}"
                row[cov_key] = cov_value

            date = df.date.unique()[i] # assign the latest day of the window
            row["date"] = date
            cov_list.append(row)

        df_cov = pd.DataFrame(cov_list)
        df = df.merge(df_cov, on='date', how='inner')
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)

        return df

    def _rolling_window_scaling(self,df, feature_list, window_size=60):
        for ticker in df['tic'].unique():
            df_ticker = df[df['tic'] == ticker]
            for feature in feature_list:
                rolling_mean = df_ticker[feature].rolling(window=window_size, min_periods=3).mean() # we need min 2 periods cause we would get +- 1/sqrt(2) by default
                rolling_std = df_ticker[feature].rolling(window=window_size, min_periods=3).std()
                scaled_feature = (df_ticker[feature] - rolling_mean) / rolling_std
                df.loc[df['tic'] == ticker, f'{feature}_scaled'] = scaled_feature
        df.dropna(inplace=True) # The first values are  NaN-s cause the nominator is zero I guess.
        #df.reset_index(drop=True, inplace=True) we overwrite it anyway
        return df
    
    def data_split(self, df, train_size, val_size, feature_list):
        df_sorted = df.sort_values(['date', 'tic'], ignore_index=True) # ignore index = True: ignores the original index. Resets the row indices to a new one
        unique_dates = df_sorted['date'].unique()
        train_size_idx = int(len(unique_dates) * train_size)
        val_size_idx = int(len(unique_dates) * (train_size + val_size))
        
        train_dates = unique_dates[:train_size_idx]
        val_dates = unique_dates[train_size_idx:val_size_idx]
        test_dates = unique_dates[val_size_idx:]
        
        train_data = df_sorted[df_sorted['date'].isin(train_dates)]
        val_data = df_sorted[df_sorted['date'].isin(val_dates)]
        test_data = df_sorted[df_sorted['date'].isin(test_dates)]
                
        train_data = self._rolling_window_scaling(train_data, feature_list)
        val_data = self._rolling_window_scaling(val_data, feature_list)
        test_data = self._rolling_window_scaling(test_data, feature_list)

        train_data.index = train_data.date.factorize()[0]
        val_data.index = val_data.date.factorize()[0]
        test_data.index = test_data.date.factorize()[0]
        
        self.final_feature_list_scaled = [col for col in train_data.columns if '_scaled' in col]


        return train_data, val_data, test_data
    

    def prepare_train_test_df(self):
        '''
        here we add the technical indicators and cov matrix, that can be created inadvance
        '''

        # we do not add here the tic-date level data, just initialized at self.tic_date
        data_with_features = self._add_features_technical()
        #data_with_features = self._get_cov_df(lookback=128, df=data_with_features)
        data_with_features = self._get_cov_df(lookback=64, df=data_with_features)

        self.final_feature_list = list(data_with_features.columns)[3:]  
        train, val, test = self.data_split(data_with_features, 0.7,0.15, self.final_feature_list)

        self.train_df = train 
        self.test_df = test  
        self.val_df = val




class Portfolio_engine:

    def __init__(self,source_csv_file1, source_csv_file2, date_level_exp_list, 
                tech_indicator_list,
                args,
                mode='train',
                day = 0):
        
        self.loader = LoaderEnginering(source_csv_file1, source_csv_file2, date_level_exp_list,args)
        self.loader.prepare_train_test_df()
        self.mode = mode
        self.args = args
        #self.df = self.loader.train_df
        
        self.set_dataset(self.mode)
        self.day = day
        self.tech_indicator_list = tech_indicator_list
        self.stock_dimension = len(self.df.tic.unique())
        self.initial_weights =np.array([1 / self.stock_dimension for _ in range(self.stock_dimension)])

        self.initial_amount = args.initial_wealth
        self.reward_scaling = args.reward_scaling
        self.transaction_cost_pct =args.transaction_cost# 0.0001 # high? 0.0001 

        self.terminal = False 
        self.reset() # NOTE: Not the best solution. Even without doing anything we already initialize with    def _initialize_portfolio(self)

    def set_dataset(self, mode):

        if mode == 'train':
            self.df = self.loader.train_df
        elif mode == 'val':
            self.df = self.loader.val_df
        elif mode == 'test':
            self.df = self.loader.test_df
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

    def switch_mode(self, new_mode):

        self.mode = new_mode
        self.set_dataset(new_mode)

    def calculate_desired_shares_and_cost(self, weights, current_wealth,last_day_memory, initialization=False):
        '''
        total_cost > 0 : buying: cash outflow
        total_cost < 0 : selling: cash inflow
        transaction_cost: always positive

        Idea: Based on today information (price,indexes,cov matrix) rearrange the current value
        of the wealth (= cash + today price * number of shares) to a target portfolio allocation based on predicted weights
        If the allocation is succesfull tomorrow we will get a reward based on (new wealth- yesterday wealth)/(yesterday wealth)

        '''
        if self.args.action_interpret=='transaction':
            self.current_prices = last_day_memory.close.values # price at time t
            #weights = self.apply_stochastic_slippage_to_weights(weights) # 
            target_allocation = weights * current_wealth # based on last day memory at time t
            #target_shares = (target_allocation // self.current_prices).astype(int) # if we dont allow fractional  shares
            target_shares = np.round((target_allocation / self.current_prices),5)
            self.shares = np.array(self.shares) if hasattr(self, 'shares') else np.zeros_like(target_shares)
            desired_shares = target_shares - self.shares  # positive means buy, negative means sell (net cash inflow)

            buy_shares = np.maximum(desired_shares, 0)
            sell_shares = np.abs(np.minimum(desired_shares, 0))
            buy_volume = np.sum(self.current_prices * np.maximum(desired_shares, 0))
            sell_volume = np.sum(self.current_prices * np.abs(np.minimum(desired_shares, 0)))
            transaction_cost = self.transaction_cost_pct * (buy_volume + sell_volume)# transaction cost is charged on the total traded volume (buys + sells)
            total_cost = np.sum(self.current_prices * desired_shares) # net value of the trade (buy volume - sell )
        
            if (total_cost + transaction_cost) > self.balance:

                max_affordable_buy = self.balance + sell_volume - transaction_cost# from sell you have money
                if max_affordable_buy < 0:
                    buy_scaling = 0.0
                else:
                    # we do not want to buy more what was planned!! so even if we can afford
                    buy_scaling = min(max_affordable_buy / buy_volume, 1.0)  if buy_volume > 0 else 0.0

                buy_shares = np.round(buy_shares * buy_scaling, 5)

                desired_shares = buy_shares - sell_shares # NOTE: it is a vector where the zeros i.e in the buy shares are the sells. so by substracting we ll get back 
                # the original desired shares set up

                buy_volume = np.sum(self.current_prices * buy_shares)
                transaction_cost = self.transaction_cost_pct * (buy_volume + sell_volume)
                total_cost = buy_volume + transaction_cost - sell_volume
                #print('buy volume after adjust',buy_volume)
                balance_after_purchase = max(0, self.balance - (total_cost)) # 
                return desired_shares, total_cost, transaction_cost, balance_after_purchase*1.0002 
        
            else:
                return desired_shares, total_cost, transaction_cost,  max(0, self.balance - (total_cost))*1.0002 
        
        else:
                '''
                we just track the weight change 
                '''
                self.current_prices = last_day_memory.close.values 
                target_allocation = weights * 1

                return target_allocation,0,0,0


    
    def _initialize_portfolio(self):
        """
        Initialize portfolio with initial weights and amount.
        """
        #self.current_prices = self.data.close.values
        last_day_memory = self.data.loc[self.day, :]
        # Use the default input weights and initial amount of money
        desired_shares, total_cost, transaction_cost, balance = self.calculate_desired_shares_and_cost(
            self.initial_weights,self.initial_amount,last_day_memory,initialization=True)

        return desired_shares, balance

      
    def step(self, actions):
        # reaching the last day of the data = Terminal station
        # NOTE: We use the raw actions and do softmax here.
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal,self.transaction_cost
        else:
            weights = self.softmax_normalization(actions)

            self.actions_memory.append(weights.flatten()) 
            last_day_memory = self.data.loc[self.day, :]
            current_wealth = self.get_wealth() # based on last day memory.
            self.day += 1 # do the step and calculate the new wealth in the get state reward function
            self.data = self.df.loc[self.day,:] # next price for new portfolio value

            reward,new_wealth,transaction_cost = self.get_state_reward(weights,current_wealth,last_day_memory)

            self.portfolio_return_memory.append(reward)
            self.date_memory.append(self.data.date)  # First day of the new state       
            self.asset_memory.append(new_wealth)
            self.transaction_cost_memory.append(transaction_cost)
            self.state = self.get_state()

            
            self.reward = reward
            self.reward = self.reward*self.reward_scaling
            self.transaction_cost = transaction_cost
            # get_state_reward: here we initialize the next state
        return self.state, self.reward, self.terminal,self.transaction_cost
    

    def step_evaluate(self, mean_actions):
        # reaching the last day of the data = Terminal station
        # NOTE: we use the mean actions without normalizaion 
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal,self.transaction_cost
        else:

            weights = mean_actions.numpy() 
            self.actions_memory.append(weights.flatten()) 
            last_day_memory = self.data.loc[self.day, :] # day t data
            current_wealth = self.get_wealth() # based on last day memory.
            self.day += 1 # do the step and define the next state
            self.data = self.df.loc[self.day,:] # next price for new portfolio value

            reward,new_wealth,transaction_cost = self.get_state_reward(weights,current_wealth,last_day_memory)

            self.portfolio_return_memory.append(reward)
            self.date_memory.append(self.data.date)  # First day of the new state       
            self.asset_memory.append(new_wealth)
            self.transaction_cost_memory.append(transaction_cost)
            self.state = self.get_state()
            # the reward is the new portfolio value or end portfolo value
           
            # Check the +1 for the while loop
            self.terminal = self.day >= len(self.df.index.unique()) - 1

            self.reward = reward
            self.reward = self.reward*self.reward_scaling
            self.transaction_cost = transaction_cost # We shouldnt track it with self.
        return self.state, self.reward, self.terminal,self.transaction_cost


    def apply_stochastic_slippage(price, mean_slippage=0.0, std_dev_slippage=0.01):

        slippage_percentage = np.random.normal(loc=mean_slippage, scale=std_dev_slippage)
        price_with_slippage = price * (1 + slippage_percentage)
        return price_with_slippage


    def apply_stochastic_slippage_to_weights(self, weights, mean_slippage=0.0, std_dev_slippage=0.001):
    # Generate slippage as a percentage deviation for each weight
        slippage_percentage = np.random.normal(loc=mean_slippage, scale=std_dev_slippage, size=len(weights))
        weights_with_slippage = weights * (1 + slippage_percentage)
        
        # Ensure weights still sum to 1 after slippage
        weights_with_slippage=tf.nn.softmax(weights_with_slippage).numpy()
        return weights_with_slippage

    def get_state_reward(self, weights,current_wealth,last_day_memory):

            # Update current data
            
            # Use the helper function to calculate desired shares and costs
            desired_shares, total_cost, transaction_cost, balance = self.calculate_desired_shares_and_cost(
                weights,current_wealth,last_day_memory
            )

            # Update balance and shares based on transaction
            self.balance = balance
            self.shares = self.shares + desired_shares # old share + delta share

            # Set state by combining features, balance, and shares
            # here we initialize the next state cause the day is already +1

            # Calculate reward as the percentage change in wealth
            # tomorrows wealth, after rearrangement.
            new_wealth = self.get_wealth()
            #print('new_wealth:',new_wealth)
            #print('current_wealth:',current_wealth)
            reward = (new_wealth - current_wealth) / current_wealth
            self.portfolio_value = new_wealth
            #print('reward:',reward)
            return reward,new_wealth,transaction_cost
    def get_wealth(self):
        return self.balance + np.sum(self.data.close.values *np.array(self.shares).flatten())
    
    def flatten_list_columns(self,features):
        list_columns = features.applymap(lambda x: isinstance(x, list)).all()

        flattened_features = features.copy()

        for col in features.columns[list_columns]:
            flattened_df = pd.DataFrame(flattened_features[col].tolist(), index=flattened_features.index)

            flattened_df.columns = [f"{col}_{i+1}" for i in range(flattened_df.shape[1])]

            flattened_features = pd.concat([flattened_features.drop(columns=[col]), flattened_df], axis=1)

        return flattened_features

    def get_state(self,add_tic_data=False):
        #self.data = self.df.loc[self.day,:]

        tic_features_flattened= self.flatten_list_columns(self.loader.tic_date)
        # merge the tic-date level
        global_ticker_date_level = self.data.merge(tic_features_flattened, how='inner', on='date')


        tic_date_features = [col for col in tic_features_flattened.columns if col != 'date']
        final_columns = self.loader.final_feature_list_scaled + tic_date_features

        if add_tic_data:
            features = global_ticker_date_level[final_columns].iloc[0] # check this again
            
        else:
            features = self.data[self.loader.final_feature_list_scaled].iloc[0] # contains both scaled and not scaled
            # iloc[0] cause it contains day many observations and all of the state variables are global
            # here we are at self.data = self.df.loc[self.day,:]
        features = np.array(features, dtype=np.float32)

        features_balance = np.append(features, self.balance*10)  # wqe need to rescale not to overemphasize the data.
        features_balance_share = np.append(features_balance, self.shares*0.01)   
        # add the tic-date level data
        # add wealth
        features_balance_share_wealth = np.append(features_balance_share,self.asset_memory[-1]*0.0005)   #0.0001,0.0005:original
        state = np.expand_dims(features_balance_share_wealth, axis=0)
        return state


    def reset(self):
        # Reset asset memory and portfolio value
        #self.asset_memory = [self.initial_amount]
        self.portfolio_value = self.initial_amount
        
        # Reset counters and flags
        self.day = 0
        self.terminal = False
        self.balance = self.initial_amount

        self.data = self.df.loc[self.day, :]
        self.shares = np.zeros_like(self.initial_weights) # we need to initialize like that cause
        # in the initialize portfolio if it is not zero we ll have a problem..
        desired_shares, balance = self._initialize_portfolio()
        self.shares = desired_shares # initialize the equal weighted shares
        self.balance = balance

        #self.actions_memory=[self.shares]
        # Reset tracking lists for each episode
        self.portfolio_return_memory = [0]
        self.actions_memory = [self.initial_weights]
        self.date_memory = [self.data.date]
        self.transaction_cost_memory =[0]
        real_wealth = self.get_wealth() #  after initialization weights
        self.asset_memory = [real_wealth] # lets contain after the initialization
        # Reinitialize state for the start of the episode
        self.state = self.get_state()
        self.transaction_cost = 0

        return self.state   
    

        
    def softmax_normalization(self, actions):
        softmax_output = tf.nn.softmax(actions).numpy()

        return softmax_output    
     
   # long term memory for rewards
   # we dont use it :)
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value
    
    # long term memory for weights
    # we dont use it:)
    def save_action_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list) # this gives error
        df_date.columns = ['date']
        # Saving the portfolio weights = actions
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions
    
    def save_transaction_cost_memory(self):
        

        return
    


