
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
    def __init__(self, data_file1,data_file2,data_file3,data_file4,date_level_exp_list,args):
        '''
     
        'NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth',
       'Utils', 'Other'
       we added a longer extended test after splitting the data.
       we will do chunk training + need exploration too which is decaying over time.
       start learning fom different points of the data
       when we are creating the added values, we can loose data points at the beginning.
       So lets add the features in the beginning and then do the split
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
        ###############################
        # Do the extension

        self.data_etf_extend = pd.read_csv(data_file3)
        self.data_exp_extend = pd.read_csv(data_file4)
        self.data_etf_extend['date'] = pd.to_datetime(self.data_etf_extend['date'])
        # Use a subset of stocks
        #self.data_etf = self.data_etf[self.data_etf['tic'].isin(['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec'])]
        self.data_etf_extend['close'] = self.data_etf_extend['close']/100 #


        self.data_exp_extend['date'] = pd.to_datetime(self.data_exp_extend['date'])
        data_exp_selected = self.data_exp_extend[['date'] + date_level_exp_list]
        self.data_exp_extend = pd.merge(self.data_etf_extend, data_exp_selected, on='date', how='inner')
        ############################## 
        self.train_df = None
        self.test_df = None
        self.final_feature_list = None
        self.tic_date = None
        self.final_feature_list_scaled = None
        self.balance_memory = []

    def _print_data_summary(self, raw_start, raw_end, processed_df, train_df, val_df, test_df):
        processed_start = processed_df['date'].min()
        processed_end = processed_df['date'].max()

        print("\n=== DATA RANGE SUMMARY ===")
        print(f"Raw data range: {raw_start} to {raw_end}")
        print(f"Processed data range: {processed_start} to {processed_end}")
        print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Val: {val_df['date'].min()} to {val_df['date'].max()}") # !!! again, with chunk training no val
        print(f"Test: {test_df['date'].min()} to {test_df['date'].max()}")

        # Loss due to rolling features/scaling
        loss_preprocess_days = (processed_start - raw_start).days
        print(f"Data lost to preprocessing: {loss_preprocess_days} days")

        if self.args.chunk_training:
            unused_after_test_days = (raw_end - test_df['date'].max()).days
            unused_after_test_days = max(unused_after_test_days, 0)
            print(f"Data unused after chunk test end: {unused_after_test_days} days")




    def _add_features_technical(self):
    # Always concatenate and calculate continuously for consistency
        if self.args.use_extended:
            data_combined = pd.concat([self.data, self.data_exp_extend], ignore_index=True)
        else:
            data_combined = self.data.copy()
        data_combined = data_combined.sort_values(['date', 'tic']).drop_duplicates(subset=['date', 'tic'])
        
        # Calculate ROCP once on full continuous dataset
        roc_combined = calculate_rocp_per_ticker(data_combined)
        
        # Store the full tic_date (always available)
        self.tic_date_full = roc_combined
        
        # Create filtered versions based on date ranges
        original_dates = self.data['date'].unique()
        
        # For non-extended mode: only use original dates
        self.tic_date = roc_combined[roc_combined['date'].isin(original_dates)]
        
        # For extended mode: use all dates (assigned in data_split if extended_data is not None)
        self.tic_date_extend = roc_combined if self.args.use_extended else self.tic_date
        
        # Return original references for backward compatibility
        data = self.data.reset_index(drop=True)
        data_extend = self.data_exp_extend.reset_index(drop=True) if self.args.use_extended else None
        
        return data, data_extend   

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
    
    def data_split(self, df, train_size, val_size, feature_list, extended_data=None):
        df_sorted = df.sort_values(['date', 'tic'], ignore_index=True)
        unique_dates = df_sorted['date'].unique()
        if self.args.chunk_training:
            start_date = pd.to_datetime(df_sorted['date'].min())
            max_date = pd.to_datetime(df_sorted['date'].max())

            # Expanding window: train starts at the beginning and grows by chunk_step_years
            # No validation window in chunk mode; test follows train directly
            train_end = start_date + pd.DateOffset(years=self.args.train_years + self.args.chunk_idx * self.args.chunk_step_years)
            test_end = train_end + pd.DateOffset(years=self.args.test_years)

            train_mask = (df_sorted['date'] >= start_date) & (df_sorted['date'] < train_end)
            test_mask = (df_sorted['date'] >= train_end) & (df_sorted['date'] <= test_end)

            train_data = df_sorted[train_mask]
            test_data = df_sorted[test_mask]

            # No eval split in chunk mode
            val_data = train_data.copy() # we need to return something
            print(f"Chunk split (expanding): train {start_date.date()} -> {train_end.date()}, test {train_end.date()} -> {test_end.date()}")
        else:
            train_size_idx = int(len(unique_dates) * train_size)
            val_size_idx = int(len(unique_dates) * (train_size + val_size))
            
            train_dates = unique_dates[:train_size_idx]
            val_dates = unique_dates[train_size_idx:val_size_idx]
            test_dates = unique_dates[val_size_idx:]
            
            train_data = df_sorted[df_sorted['date'].isin(train_dates)]
            val_data = df_sorted[df_sorted['date'].isin(val_dates)]
            test_data = df_sorted[df_sorted['date'].isin(test_dates)]
        
        # Concatenate extended data to test (already scaled) (only if args.use_extended is True)
        is_extended = False
        if self.args.use_extended:
            if extended_data is None:
                raise ValueError("use_extended=True but extended_data is None")
            test_last_date = test_data['date'].max()
            extended_after = extended_data[extended_data['date'] > test_last_date]
            test_data = pd.concat([test_data, extended_after], ignore_index=True)
            print(f"Extended test with {len(extended_after)} additional rows")
            is_extended = True
            
            # Use full continuous tic_date for extended mode
            self.tic_date = self.tic_date_extend
                
        train_data.index = train_data.date.factorize()[0]
        val_data.index = val_data.date.factorize()[0]
        test_data.index = test_data.date.factorize()[0]
        
        # Save CSV output
        if self.args.chunk_training:
            train_filename = "train_data_chunk.csv"
            train_data.to_csv(train_filename, index=False)
            print(f"Saved train data to {train_filename}")
        else:
            extension_label = 'extended' if is_extended else 'not_extended'
            test_filename = f'test_data_{extension_label}.csv'
            test_data.to_csv(test_filename, index=False)
            print(f"Saved test data to {test_filename}")

        return train_data, val_data, test_data
    

    def prepare_train_test_df(self):
        '''
        here we add the technical indicators and cov matrix, that can be created inadvance
        check this
        '''
        print("\n=== DATA PREP MODE ===")
        print(f"use_extended={self.args.use_extended}, chunk_training={self.args.chunk_training}, chunk_idx={getattr(self.args, 'chunk_idx', None)}")
        print(f"train_years={getattr(self.args, 'train_years', None)}, test_years={getattr(self.args, 'test_years', None)}")

        # Process original data (and extended data if enabled)
        data_with_features, data_extend_with_features = self._add_features_technical()
        
        # Track original data start/end (information starting point)
        original_start = data_with_features['date'].min()
        original_end = data_with_features['date'].max()
        print(f"\n=== RAW DATA INFORMATION POINTS ===")
        print(f"Original data starts: {original_start}")
        print(f"Original data ends: {original_end}")
        print(f"Total days in original data: {len(data_with_features['date'].unique())}")
        
        # Apply cov to original (and extended if enabled) (this consumes first 64 days)
        data_with_features = self._get_cov_df(lookback=64, df=data_with_features)
        if self.args.use_extended and data_extend_with_features is not None:
            data_extend_with_features = self._get_cov_df(lookback=64, df=data_extend_with_features)
        
        after_cov_start = data_with_features['date'].min()
        print(f"\n=== AFTER COVARIANCE CALCULATION (lookback=64) ===")
        print(f"Data starts after cov: {after_cov_start}")
        print(f"Days consumed by cov lookback: {(after_cov_start - original_start).days}")

        self.final_feature_list = list(data_with_features.columns)[3:]  

        # Apply rolling scaling ONCE on full continuous data (avoid repeated loss per split)
        data_with_features = self._rolling_window_scaling(data_with_features, self.final_feature_list)
        if self.args.use_extended and data_extend_with_features is not None:
            data_extend_with_features = self._rolling_window_scaling(data_extend_with_features, self.final_feature_list)

        self.final_feature_list_scaled = [col for col in data_with_features.columns if '_scaled' in col]
        
        # Pass extended data to data_split (only when enabled)
        train, val, test = self.data_split(
            data_with_features,
            0.7,
            0.15,
            self.final_feature_list,
            extended_data=data_extend_with_features if self.args.use_extended else None
        )

        self.train_df = train 
        self.test_df = test  
        self.val_df = val
        
        print(f"\n=== FINAL USABLE DATA (after scaling with window=60) ===")
        print(f"Train: {train['date'].min()} to {train['date'].max()}")
        print(f"Val: {val['date'].min()} to {val['date'].max()}")
        print(f"Test: {test['date'].min()} to {test['date'].max()}")
        print(f"\n=== LOOKBACK REQUIREMENTS ===")
        print(f"Information must start at: {original_start}")
        print(f"First usable training date: {train['date'].min()}")
        print(f"Total lookback period: {(train['date'].min() - original_start).days} days")

        # Summary (raw vs processed vs splits)
        self._print_data_summary(original_start, original_end, data_with_features, train, val, test)

        # Persist a simple preprocessing log to help verify no leakage across splits
        log_lines = [
            "=== PREPROCESS LOG ===",
            f"use_extended={self.args.use_extended}",
            f"chunk_training={self.args.chunk_training}",
            f"chunk_idx={getattr(self.args, 'chunk_idx', None)}",
            f"train_years={getattr(self.args, 'train_years', None)}",
            f"test_years={getattr(self.args, 'test_years', None)}",
            f"raw_start={original_start}",
            f"raw_end={original_end}",
            f"after_cov_start={after_cov_start}",
            f"train_range={train['date'].min()} -> {train['date'].max()}",
            f"val_range={val['date'].min()} -> {val['date'].max()}",
            f"test_range={test['date'].min()} -> {test['date'].max()}",
            f"lookback_days={(train['date'].min() - original_start).days}",
        ]
        log_path = os.path.join(os.getcwd(), "preprocess_summary.txt")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(str(line) for line in log_lines))



class Portfolio_engine:

    def __init__(self, source_csv_file1, source_csv_file2, source_csv_file3, source_csv_file4, 
                date_level_exp_list, 
                tech_indicator_list,
                args,
                mode='train',
                day = 0):
        
        self.loader = LoaderEnginering(source_csv_file1, source_csv_file2, source_csv_file3, source_csv_file4, 
                                       date_level_exp_list, args)
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
        Calculate transaction cost based on weight drift from (t-1) to t.
        
        At time t (decision time):
        - w_prev_post: weights decided at t-1
        - Need returns from (t-1)→t to drift w_prev_post to w_pre(t)
        - self.day is currently at t (before next increment in step_evaluate)
        
        Timeline:
        - At start of step_evaluate(t): self.day = t
        - We decide weights at t: w_target
        - Then self.day += 1 (moves to t+1)
        - We call get_state_reward(..., next_returns from t→t+1)
        - Inside get_state_reward: cost_turnover is called
        - NOW self.day = t+1, so we need self.day-2 to get (t-1)→t returns
        '''
        w_target = np.asarray(w_target, float).ravel()
        w_target = np.clip(w_target, 0.0, None) # no short, just to make sure

        # Get returns from (t-1)→t for drift calculation
        # Since step_evaluate increments self.day BEFORE calling this,
        # self.day is now at t+1, so:
        # - self.day-1 = t (current returns t→t+1, WRONG for drift)
        # - self.day-2 = t-1 (past returns (t-1)→t, CORRECT for drift)
        if self.day >= 2:
            # Returns from (t-1)→t to drift yesterday's weights
            R_prev = self.df.loc[self.day - 2, :].close.values
        else:
            # First or second step: no previous returns for drift
            R_prev = np.zeros_like(w_target)

        # Drift yesterday's post-trade weights to today's pre-trade weights
        w_prev_post = np.asarray(w_prev_post, float).ravel()
        g = 1.0 + np.asarray(R_prev, float).ravel()
        numer = w_prev_post * g
        denom = float(numer.sum())
        w_pre = numer / denom if abs(denom) > 1e-12 else w_prev_post

        # Turnover cost based on absolute weight change
        tau = 0.5 * float(np.sum(np.abs(w_target - w_pre)))

        # Apply cost fraction
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
    


