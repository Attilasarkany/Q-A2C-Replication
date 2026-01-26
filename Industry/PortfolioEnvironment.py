
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

from Methods import calculate_roc_per_ticker,calculate_rocp_per_ticker
class LoaderEnginering:
    def __init__(self, data_file1,data_file2,date_level_exp_list,args):
        '''
     
        'NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth',
       'Utils', 'Other'
        '''

        self.args = args
        self.data_etf = pd.read_csv(data_file1)
        self.data_exp = pd.read_csv(data_file2)
        self.data_etf['date'] = pd.to_datetime(self.data_etf['date'])
        # Use a subset of stocks
        #self.data_etf = self.data_etf[self.data_etf['tic'].isin(['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec'])]
        self.data_etf['close'] = self.data_etf['close']/100 #


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
        #roc = calculate_roc_per_ticker(data)
        roc = calculate_rocp_per_ticker(data)
        #sma = calculate_sma_per_ticker(data)
        #bb = calculate_bollinger_bands(data)
        dataframes = [roc] # ordered per ticker
        self.tic_date = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dataframes) # if we have more

        data = data.reset_index(drop=True)
        
        return data    

    def _add_cov_matrix(self, lookback, df):
        df = df.sort_values(['date', 'tic'], ignore_index=True).copy()
        df.index = df.date.factorize()[0]

        cov_list = [] 

        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i-1, :]  # Pandas includes right end as well, very weird
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            #return_lookback = price_lookback.pct_change().dropna()
            covs = price_lookback.cov().values

            upper_tri_indices = covs[np.triu_indices(covs.shape[0])] # Get the upper  triangular values + diagonal elements. covs.shape[0] means the number of diagonal elements as I understood
            row = {f"cov_{lookback}_{i}": cov_value for i, cov_value in enumerate(upper_tri_indices)} 

            date = df.date.unique()[i]
            row['date'] = date

            cov_list.append(row)

        df_cov = pd.DataFrame(cov_list)
        df = df.merge(df_cov, on='date', how='inner')
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)

        return df
    
    '''
    def _get_cov_df(self, lookback, df):

        df = df.sort_values(['date', 'tic'], ignore_index=True).copy()
        df.index = df.date.factorize()[0]


        cov_list = []  # create empty list for storing coveriance matrices at each time step

        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i-1, :]  # Pandas includes right end as well, very weird
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            #return_lookback = price_lookback.pct_change().dropna()
            covs = price_lookback.cov()# cov
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
    '''

    def _get_cov_df(self, lookback, df):
        """
        Keep only diagonal elements (per-asset variances) of the rolling covariance.
        """
        df = df.sort_values(['date', 'tic'], ignore_index=True).copy()
        df.index = df.date.factorize()[0]

        cov_list = []
        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i-1, :]
            # 'close' already holds daily returns
            ret_mat = data_lookback.pivot_table(index='date', columns='tic', values='close')
            covs = ret_mat.cov()  # sample covariance (N-1)
            tickers = covs.columns

            row = {}
            # <-- only diagonal elements
            r_idx, c_idx = np.diag_indices(covs.shape[0])
            for r, c in zip(r_idx, c_idx):
                var_value = float(covs.iloc[r, c])
                cov_key = f"var_{lookback}_{tickers[r]}"                  
                row[cov_key] = var_value

            row["date"] = df.date.unique()[i]
            cov_list.append(row)

        df_var = pd.DataFrame(cov_list)
        df = df.merge(df_var, on='date', how='inner')
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
        check this
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

        self.reward_scaling = args.reward_scaling
        #self.transaction_cost_pct =args.transaction_cost# 0.0001 # high? 0.0001 
        self.c = args.cost_fraction

        self.terminal = False 
        self.r_f= args.r_f
        self.N = len(self.loader.data_etf.tic.unique())

        self.initial_amount = args.initial_wealth
        self.reset() # NOTE: Not the best solution. Even without doing anything we already initialize with    def _initialize_portfolio(self)


        self.portfolio_value = float(self.initial_amount)
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

    
    
    '''
    def cost_turnover(self, w_target, w_prev_post):

        w_target = np.asarray(w_target, float)
        w_prev_post = np.asarray(w_prev_post, float)

        w_target = np.clip(w_target, 0, None) # for safety


        tc = float(np.sum(np.abs(w_target - w_prev_post )))      
  
        txn_cost = self.c * tc
        
        return txn_cost
    
    '''
    
    def cost_turnover(self, w_target, w_prev_post):
        '''
        we have to be careful cause of the ordering in assets...It is fixed tho:) so ok
        target: new weight
        w_prev_post: previos weights without a drift
        '''
        w_target = np.asarray(w_target, float).ravel()
        w_target = np.clip(w_target, 0.0, None) # no short, just to make sure


        # returns used to drift to pre-trade weights: R_t from (t-1 -> t)
        if self.day > 0:
            R_t = self.df.loc[self.day - 1, :].close.values # we did a step day = t+1--> -1--> t return for adjustment
        else:
            R_t = np.zeros_like(w_target)

        # drift yesterday's post-trade weights to today's pre-trade weights
        w_prev_post = np.asarray(w_prev_post, float).ravel()
        g = 1.0 + np.asarray(R_t, float).ravel()
        numer = w_prev_post * g
        denom = float(numer.sum())
        w_pre = numer / denom if abs(denom) > 1e-12 else w_prev_post

        # turnover cost
        tau = 0.5 * float(np.sum(np.abs(w_target - w_pre)))

        # c: regularizor
        return self.c * tau
    

      
    def step(self, actions):
        # reaching the last day of the data = Terminal station
        # NOTE: We use the raw actions and do softmax here.
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']

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

            self.actions_memory.append(weights.flatten()) # super important, we do append the actions memory
            # so we need -2 from the list to have pre weight
            #last_day_memory = self.data.loc[self.day, :]
            #current_wealth = self.get_wealth() # cumulative wealth based on last day memory
            self.day += 1 # do the step and calculate the new wealth in the get state reward function
            self.data = self.df.loc[self.day,:] # next price for new portfolio value
            # self.actions_memory[-2]: we just did a step but we need the previous weight not the current

            next_returns = self.data.close.values  # simple returns for next day. Very important we are at t+1--> reward!

            #wealth_prev = self.portfolio_value
            reward_pct, transaction_cost,wealth_new = self.get_state_reward(
                weights, self.actions_memory[-2], next_day_memory=next_returns
            )
            #wealth_new = wealth_prev * (1.0 + reward_pct)

            # 
            self.portfolio_return_memory.append(reward_pct)
            self.asset_memory.append(wealth_new)

            self.date_memory.append(self.data.date)
            self.transaction_cost_memory.append(transaction_cost)

            self.state = self.get_state() # self.data is alrady forwarded to next day
            self.reward = reward_pct * self.reward_scaling  # scale wealth change if desired
            self.transaction_cost = transaction_cost
            return self.state, self.reward, self.terminal, self.transaction_cost
    

    def step_evaluate(self, mean_actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            df_daily_return = pd.DataFrame(self.portfolio_return_memory, columns=['daily_return'])
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252**0.5) * df_daily_return['daily_return'].mean() / df_daily_return['daily_return'].std()
                print("Sharpe: ", sharpe)
            print("=================================")
            return self.state, self.reward, self.terminal, self.transaction_cost
        else:
            weights = mean_actions.numpy()  
            self.actions_memory.append(weights.flatten())

            self.day += 1
            self.data = self.df.loc[self.day, :]
            next_returns = self.data.close.values # forward looking

            #wealth_prev = self.portfolio_value # no change so far
            reward_pct, transaction_cost,wealth_new = self.get_state_reward(
                weights=weights, w_prev_post=self.actions_memory[-2], next_day_memory=next_returns
            )
            #wealth_new = wealth_prev * (1.0 + reward_pct)
            #self.portfolio_value = wealth_new # we already assigne in the get_state_reward but ok
            self.asset_memory.append(wealth_new)


            self.portfolio_return_memory.append(reward_pct)

            self.date_memory.append(self.data.date)
            self.transaction_cost_memory.append(transaction_cost)
            self.state = self.get_state()

            self.terminal = self.day >= len(self.df.index.unique()) - 1
            self.reward = reward_pct * self.reward_scaling
            self.transaction_cost = transaction_cost
            return self.state, self.reward, self.terminal, self.transaction_cost


    def returns(self, w, R_risky):
        '''
        We dont use risk free here
        R_risky: based on tomorrow rearrangement you ll get a return (next_day_memory)
        weights: based on today decision output using today information
        '''
        w = np.asarray(w).reshape(self.N,) 
        risky = w @ R_risky    #               
        #w_cash = 1.0 - np.sum(w) # r_f share
        return risky

    
    def get_state_reward(self, weights, w_prev_post, next_day_memory):
        """
        Wealth-based reward with simple return and additive cost:
        r_net = r_p - tc
        V_{t+1} = V_t * (1 + r_net)
        where tc = cost_turnover(...) is a fraction of wealth.
        weights: current choice
        w_prev_post: actions_memory[-2]: so after append--> t-1 weight
        """
        wealth_prev = self.portfolio_value

        tc = self.cost_turnover(weights, w_prev_post)          # fraction (e.g., 0.0003)
        r_p = self.returns(weights, next_day_memory)           # simple portfolio return

        r_net = r_p - tc                                       # net simple return
        wealth_new = wealth_prev * (1.0 + r_net)               
        reward = r_net                                       

        self.portfolio_value = wealth_new
        return reward, tc, wealth_new
        '''  

    def get_state_reward(self, weights, w_prev_post, next_day_memory):
        """
        Wealth-based reward using log growth:
            reward = log(V_{t+1}/V_t) = log( (1 + r_p) * (1 - tc) )
        This makes the sum of rewards over an episode equal to log(V_T / V_0).
        """
        wealth_prev = self.portfolio_value

        transaction_cost = self.cost_turnover(weights, w_prev_post)
        r_p = self.returns(weights, next_day_memory)

        growth = (1.0 + r_p) * (1.0 - transaction_cost)
        growth = max(growth, 1e-12)  # numerical safety

        wealth_new = wealth_prev * growth
        reward = np.log(growth)      
        self.portfolio_value = wealth_new
        return reward, transaction_cost, wealth_new  
    '''
    
    def flatten_list_columns(self,features):
        list_columns = features.applymap(lambda x: isinstance(x, list)).all()

        flattened_features = features.copy()

        for col in features.columns[list_columns]:
            flattened_df = pd.DataFrame(flattened_features[col].tolist(), index=flattened_features.index)

            flattened_df.columns = [f"{col}_{i+1}" for i in range(flattened_df.shape[1])]

            flattened_features = pd.concat([flattened_features.drop(columns=[col]), flattened_df], axis=1)

        return flattened_features
    
    
    def get_state(self,add_tic_data=True): # False, should be hard coded....
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
        features_balance = np.append(features, self.transaction_cost_memory[-1]*10)
        features_balance_share = np.append(features_balance, self.actions_memory[-1])

        # NEW: include (scaled) wealth instead of last return; use log wealth ratio for stationarity
        # lets assume that we have unit wealth and starting position of equal weighted portfolios
        features_balance_share_wealth = np.append(features_balance_share, self.asset_memory[-1]*15)

        # add current drawdown
        peakV = max(self.asset_memory) if self.asset_memory else 1.0
        drawdown = 1.0 - self.portfolio_value/(peakV + 1e-8)
        features_balance_share_wealth = np.append(features_balance_share_wealth, drawdown)

        state = np.expand_dims(features_balance_share_wealth, axis=0)
        return state
        '''
    def get_state(self, add_tic_data=True):
        # --- build the base features exactly as you do now ---
        tic_features_flattened = self.flatten_list_columns(self.loader.tic_date)
        global_ticker_date_level = self.data.merge(tic_features_flattened, on='date', how='inner')

        tic_date_features = [c for c in tic_features_flattened.columns if c != 'date']
        final_columns = self.loader.final_feature_list_scaled + tic_date_features

        base = (global_ticker_date_level[final_columns].iloc[0]
                if add_tic_data else
                self.data[self.loader.final_feature_list_scaled].iloc[0])

        # --- portfolio context (keep as-is, maybe drop the *10/*15 if you like) ---
        feat = np.append(base, self.transaction_cost_memory[-1] * 10)
        feat = np.append(feat, self.actions_memory[-1])                # prev weights
        feat = np.append(feat, self.asset_memory[-1] * 15)             # wealth proxy

        peakV = max(self.asset_memory) if self.asset_memory else 1.0
        drawdown = 1.0 - self.portfolio_value / (peakV + 1e-8)
        feat = np.append(feat, drawdown)

        # --- NEW: re-pack per-asset features into aligned blocks ---
        # example for ROCP_20; extend to any ROCP_* you computed
        N = self.stock_dimension
        rocp20_cols = [f'ROCP_20_list_{i+1}' for i in range(N) if f'ROCP_20_list_{i+1}' in global_ticker_date_level.columns]
        if len(rocp20_cols) == N:
            rocp20 = global_ticker_date_level[rocp20_cols].iloc[0].to_numpy()
            prev_w = np.asarray(self.actions_memory[-1]).reshape(-1)
            # add a tiny per-asset block [rocp20_i, prev_w_i] for each asset, flattened
            per_asset_block = np.column_stack([rocp20, prev_w]).reshape(-1)
            feat = np.append(feat, per_asset_block)

        state = np.expand_dims(feat, axis=0)
        return state
        '''


    def reset(self):
        self.day = 0
        self.terminal = False
        self.data = self.df.loc[self.day, :]

        # reset wealth and memories
        self.portfolio_value = float(self.initial_amount)
        self.portfolio_return_memory = [0.0]  # net returns for Sharpe
        self.actions_memory = [self.initial_weights]
        self.date_memory = [self.data.date]
        self.transaction_cost_memory = [0.0]
        self.asset_memory = [1.0] # initial let it be 1
        self.transaction_cost = 0.0

        self.state = self.get_state()
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
    


