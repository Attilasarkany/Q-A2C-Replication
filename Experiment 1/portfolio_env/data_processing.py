import pandas as pd
import numpy as np
# import ta
# from ta import add_all_ta_features
# from ta.utils import dropna
# from sklearn.preprocessing import MinMaxScaler

from Methods import calculate_roc_per_ticker, calculate_bollinger_bands, calculate_sma_per_ticker
from functools import reduce


class LoaderEngineering:
    def __init__(self, data_file1,data_file2,date_level_exp_list,args):

        self.args = args
        self.data_etf = pd.read_csv(data_file1)
        self.data_exp = pd.read_csv(data_file2)
        self.data_etf['date'] = pd.to_datetime(self.data_etf['date'])
        self.data_exp['date'] = pd.to_datetime(self.data_exp['date'])
        data_exp_selected = self.data_exp[['date'] + date_level_exp_list]
        self.data = pd.merge(self.data_etf, data_exp_selected, on='date', how='inner')


        self.train_df = None
        self.test_df = None
        self.final_feature_list = None
        self.tic_date = None
        self.final_feature_list_scaled = None
        self.val_df = None

    # I will order by date and ticker the data later.
    # self.data = self.data.sort_values(['date', 'tic'], ignore_index=True)

    def _add_features_technical(self):
        data = self.data


        roc = calculate_roc_per_ticker(data)
        sma = calculate_sma_per_ticker(data)
        bb = calculate_bollinger_bands(data)
        # TODO: SMA, Bollinger: cov matrix and other variables are 
        dataframes = [roc,bb]
        self.tic_date = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dataframes)

        data = data.reset_index(drop=True)
        return data

    def _get_cov_df_original_columns(self, lookback, df):
        df = df.sort_values(['date', 'tic'], ignore_index=True)
        df.index = df.date.factorize()[0]

        cov_list = []  # create empty list for storing coveriance matrices at each time step

        for i in range(lookback, len(df.index.unique())):
            data_lookback = df.loc[i - lookback:i-1, :]  # Pandas includes right end as well, very weird
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            covs = return_lookback.cov().values

            #row = {f"cov_{lookback}_{i}": cov_value for i, cov_value in enumerate(covs.flatten())}  # TODO remove repeating items
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
                    'emerging_equity': 'EE',
                    'min_vol': 'MV',
                    'momentum': 'M',
                    'quality': 'Q',
                    'size': 'S',
                    'value': 'V',
                    'fixed_income_balanced':'FB'
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
    # Simple moving average is not the best. that is smoothing only but do not scale.
    def _rolling_window_scaling(self,df, feature_list, window_size=60):
        for ticker in df['tic'].unique():
            df_ticker = df[df['tic'] == ticker]
            for feature in feature_list:
                rolling_mean = df_ticker[feature].rolling(window=window_size, min_periods=3).mean() # we need min 2 periods cause we would get +- 1/sqrt(2) by default
                rolling_std = df_ticker[feature].rolling(window=window_size, min_periods=3).std()
                scaled_feature = (df_ticker[feature] - rolling_mean) / rolling_std
                df.loc[df['tic'] == ticker, f'{feature}_scaled'] = scaled_feature
        df.dropna(inplace=True) # The first values are  NaN-s cause the nominator is zero I guess.
        return df
    
    def data_split(self, df, train_size, val_size, feature_list):
        df_sorted = df.sort_values(['date', 'tic'], ignore_index=True)
        unique_dates = df_sorted['date'].unique()
        train_size_idx = int(len(unique_dates) * train_size)
        val_size_idx = int(len(unique_dates) * (train_size + val_size))
        
        train_dates = unique_dates[:train_size_idx]
        val_dates = unique_dates[train_size_idx:val_size_idx]
        test_dates = unique_dates[val_size_idx:]
        
        train_data = df_sorted[df_sorted['date'].isin(train_dates)]
        val_data = df_sorted[df_sorted['date'].isin(val_dates)]
        test_data = df_sorted[df_sorted['date'].isin(test_dates)]
        
        train_data.index = train_data.date.factorize()[0]
        val_data.index = val_data.date.factorize()[0]
        test_data.index = test_data.date.factorize()[0]
        
        train_data = self._rolling_window_scaling(train_data, feature_list)
        val_data = self._rolling_window_scaling(val_data, feature_list)
        test_data = self._rolling_window_scaling(test_data, feature_list)
        
        self.final_feature_list_scaled = [col for col in train_data.columns if '_scaled' in col]


        return train_data, val_data, test_data

    def prepare_train_test_df(self):


        data_with_features = self._add_features_technical()
        data_with_features = self._get_cov_df(lookback=128, df=data_with_features)
        data_with_features = self._get_cov_df(lookback=64, df=data_with_features)

        self.final_feature_list = list(data_with_features.columns)[3:]  # 11: matrix only
        train, val, test = self.data_split(data_with_features, 0.7,0.15, self.final_feature_list)

        self.train_df = train  # TODO should be already on date level only
        self.test_df = test  # TODO should be already on date level only
        self.val_df = val
       
    def flatten_list_columns(self,features):
        list_columns = features.applymap(lambda x: isinstance(x, list)).all()

        flattened_features = features.copy()

        for col in features.columns[list_columns]:
            flattened_df = pd.DataFrame(flattened_features[col].tolist(), index=flattened_features.index)

            flattened_df.columns = [f"{col}_{i+1}" for i in range(flattened_df.shape[1])]

            flattened_features = pd.concat([flattened_features.drop(columns=[col]), flattened_df], axis=1)

        return flattened_features
    
    def get_states(self,mode):
    # Use the same function for training and testing
    # TODO: shorted?
        if mode =='train':
            df = self.train_df
        elif mode=='test':
            df = self.test_df
        else:
            df = self.val_df

        if self.args.add_tic_date ==True:
                tic_features_flattened= self.flatten_list_columns(self.tic_date)
                global_ticker_date_level = df.merge(tic_features_flattened, how='inner', on='date')
                tic_date_features = [col for col in tic_features_flattened.columns if col != 'date']
                final_columns = self.final_feature_list_scaled + tic_date_features
                features = global_ticker_date_level[final_columns].drop_duplicates()
                features = np.array(features, dtype=np.float32)


        else:
            # Only use the final feature list  tic_date
            features = df[self.final_feature_list_scaled].drop_duplicates()
            features = np.array(features, dtype=np.float32)

        # Return the resulting features
        return features

    def get_prices(self,mode):
        if mode =='train':
            df = self.train_df
        elif mode=='test':
            df = self.test_df
        else:
            df = self.val_df
            
        df_prices = df[['tic', 'close']]
        df_prices = df_prices.pivot(columns='tic', values=['close']).reset_index(drop=True)

        prices = np.array(df_prices, dtype=np.float32)

        return prices, list(df_prices.columns.levels[1])

    def get_dates(self,mode):
        if mode =='train':
            df = self.train_df
        elif mode=='test':
            df = self.test_df
        else:
            df = self.val_df
            
        date_df = df[['date']].drop_duplicates()  
        dates = np.array(date_df, dtype=np.datetime64)

        return dates


class BatchHandler:
    def __init__(self, source_csv_file1, source_csv_file2, date_level_exp_list,args):
        self.loader = LoaderEngineering(source_csv_file1, source_csv_file2, date_level_exp_list,args)
        self.loader.prepare_train_test_df()
        self.mode = args.mode
        self.states = self.loader.get_states(self.mode) # full train/test val df
        self.prices, self.price_names = self.loader.get_prices(self.mode)
        self.dates = self.loader.get_dates(self.mode)

        self.batches = None

    def switch_mode(self, mode):
        """
        Switches between 'train', 'validation', and 'test' modes to load the corresponding data.
        """
        self.mode = mode
        self.states = self.loader.get_states(mode=self.mode)
        self.prices, self.price_names = self.loader.get_prices(mode=self.mode)
        self.dates = self.loader.get_dates(mode=self.mode)

    def get_states(self, index_from, index_to):
        return self.states[index_from: index_to, :]

    def get_prices(self, index_from, index_to):
        return self.prices[index_from: index_to, :]

    def get_dates(self, index_from, index_to):
        return self.dates[index_from: index_to, :]

    def prepare_batches(self, batch_size):
        batches = []
        done = False
        index_from = 1 # -1 obs
        index_to = batch_size + 1
        n = self.states.shape[0]

        while not done:
            if index_to >= n:
                index_to = n - 1
                done = True

            batches.append(
                {
                    "state": self.get_states(index_from, index_to),
                    "state_next": self.get_states(index_from + 1, index_to + 1),
                    "price_close": self.get_prices(index_from, index_to),
                    "price_close_next": self.get_prices(index_from + 1, index_to + 1),
                    "date": self.get_dates(index_from, index_to)
                }
            )

            index_from = index_to
            index_to = index_from + batch_size

        return batches

    def get_train_data(self):
        n = self.states.shape[0]

        return self.prepare_batches(n)[0]

    def get_batch(self, i):
        return self.batches[i]


