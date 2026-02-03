import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def calculate_rolling_max_drawdown(portfolio_values, window=252):
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.rolling(window, min_periods=1).max()
    daily_drawdown = portfolio_series / rolling_max - 1.0
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
    
    return max_daily_drawdown



def rolling_return_over_std(reward_memory, window=252):
    if len(reward_memory) < window:
        return 0  
    else:
        rolling_returns = pd.Series(reward_memory).rolling(window).mean()
        rolling_std = pd.Series(reward_memory).rolling(window).std()
        return (rolling_returns / rolling_std).iloc[-1]

def cumulative_sharpe_ratio(reward_memory, risk_free_rate=0.0):
    excess_returns = np.array(reward_memory) - risk_free_rate
    cumulative_return = np.cumsum(excess_returns)
    if np.std(excess_returns) != 0:
        sharpe_ratio = cumulative_return[-1] / np.std(excess_returns)
    else:
        sharpe_ratio = 0.0
    return sharpe_ratio


# Define custom functions to add explanatory technical analysis variables
# Focus on the Momentum and Volatility indicators
#20,50 seeems ok
def calculate_roc_per_ticker(df, ticker_col='tic', close_col='close', periods=[5, 10]):
    def roc_for_ticker(ticker_group, period):
        return ticker_group[close_col].pct_change(periods=period)
    data=df.copy()
    data = data.sort_values(['date', 'tic'], ignore_index=True)

    roc_data = pd.DataFrame()

    for period in periods:
        data[f'ROC_{period}'] = data.groupby(ticker_col, group_keys=True).apply(lambda x: roc_for_ticker(x, period)).reset_index(level=0, drop=True)
        data[f'ROC_{period}'] = data[f'ROC_{period}'].round(3)
        roc_list_by_date = data.groupby('date')[f'ROC_{period}'].apply(list).reset_index()

        roc_list_by_date.rename(columns={f'ROC_{period}': f'ROC_{period}_list'}, inplace=True)

        if roc_data.empty:
            roc_data = roc_list_by_date
        else:
            roc_data = roc_data.merge(roc_list_by_date, on='date')

    for period in periods:
        roc_data = roc_data[roc_data[f'ROC_{period}_list'].apply(lambda x: not any(pd.isna(val) for val in x))]

    return roc_data

def calculate_bollinger_bands(df, ticker_col='tic', close_col='close', period=20):
    def bollinger_for_ticker(ticker_group, period=20):
        sma = ticker_group[close_col].rolling(window=period).mean()/100
        std = ticker_group[close_col].rolling(window=period).std()/100
        upper_band = sma + (std * 2)/ 100
        lower_band = sma - (std * 2)/ 100
        return pd.DataFrame({'SMA': sma, 'Upper_Band': upper_band, 'Lower_Band': lower_band})
    data=df.copy()
    data = data.sort_values(['date', 'tic'], ignore_index=True)
    bollinger_values = data.groupby(ticker_col, group_keys=True).apply(lambda x: bollinger_for_ticker(x, period)).reset_index(level=0, drop=True)
    bollinger_values = bollinger_values.round(3)
    data = data.join(bollinger_values, how='left')
    
    # Convert Bollinger Bands to list per date
    bollinger_data = pd.DataFrame()
    for col in ['SMA', 'Upper_Band', 'Lower_Band']:
        bollinger_list_by_date = data.groupby('date')[col].apply(list).reset_index()
        bollinger_list_by_date.rename(columns={col: f'{col}_list'}, inplace=True)
        
        if bollinger_data.empty:
            bollinger_data = bollinger_list_by_date
        else:
            bollinger_data = bollinger_data.merge(bollinger_list_by_date, on='date')
    
    # Filter out rows with NaN values in any list
    for col in ['SMA_list', 'Upper_Band_list', 'Lower_Band_list']:
        bollinger_data = bollinger_data[bollinger_data[col].apply(lambda x: not any(pd.isna(val) for val in x))]
    
    return bollinger_data.reset_index(drop=True)

def calculate_sma_per_ticker(df, ticker_col='tic', close_col='close', periods=[5]):
    def sma_for_ticker(ticker_group, period):
        return ticker_group[close_col].rolling(window=period).mean()/ 100
    data=df.copy()
    data = data.sort_values(['date', 'tic'], ignore_index=True)
    sma_data = pd.DataFrame()
    
    for period in periods:
        data[f'SMA_{period}'] = data.groupby(ticker_col, group_keys=True).apply(lambda x: sma_for_ticker(x, period)).reset_index(level=0, drop=True)
        data[f'SMA_{period}'] = data[f'SMA_{period}'].round(3)
        sma_list_by_date = data.groupby('date')[f'SMA_{period}'].apply(list).reset_index()
        sma_list_by_date.rename(columns={f'SMA_{period}': f'SMA_{period}_list'}, inplace=True)
        
        if sma_data.empty:
            sma_data = sma_list_by_date
        else:
            sma_data = sma_data.merge(sma_list_by_date, on='date')
    
    # Filter out rows with NaN values in any list
    for period in periods:
        sma_data = sma_data[sma_data[f'SMA_{period}_list'].apply(lambda x: not any(pd.isna(val) for val in x))]
    
    return sma_data.reset_index(drop=True)


import numpy as np

def calculate_rocp_per_ticker(df, ticker_col='tic', close_col='close', periods=[20]):
    def rocp_for_ticker(ticker_group, period):
        r = ticker_group[close_col]  # daily simple returns (decimals)
        # exp(sum(log(1+r))) - 1  = compounded return over the window
        # momentum
        return np.expm1(np.log1p(r).rolling(window=period, min_periods=period).sum())

    data = df.copy()
    data = data.sort_values(['date', 'tic'], ignore_index=True)
    rocp_data = pd.DataFrame()
    
    for period in periods:
        data[f'ROCP_{period}'] = (
            data.groupby(ticker_col, group_keys=True)
                .apply(lambda x: rocp_for_ticker(x, period))
                .reset_index(level=0, drop=True)
        )
        data[f'ROCP_{period}'] = data[f'ROCP_{period}'].round(3)

        rocp_list_by_date = (
            data.groupby('date')[f'ROCP_{period}']
                .apply(list).reset_index()
                .rename(columns={f'ROCP_{period}': f'ROCP_{period}_list'})
        )
        rocp_data = rocp_list_by_date if rocp_data.empty else rocp_data.merge(rocp_list_by_date, on='date')

    # Filter out rows with NaNs in any list (warmup window)
    for period in periods:
        rocp_data = rocp_data[rocp_data[f'ROCP_{period}_list'].apply(lambda x: not any(pd.isna(val) for val in x))]

    return rocp_data



def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    
    def _qloss(y_true, y_pred):
        I = tf.cast(y_true <= y_pred, tf.float32)
        d = K.abs(y_true - y_pred)
        correction = I * (1 - perc) + (1 - I) * perc
        huber_loss = K.sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), axis=-1)
        q_order_loss = K.sum(K.maximum(0.0, y_pred[:, :-1] - y_pred[:, 1:] + 1e-6), axis=-1)
        return huber_loss + q_order_loss

    return _qloss


'''
or we can use hinge loss:
    diff = f[:, 1:] - f[:, :-1]
    penalty = K.mean(K.maximum(0.0, margin - diff)) * alpha

'''
