import pandas as pd
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp



def compute_monthly_factor_returns_from_daily(daily_factors: pd.DataFrame, factor_cols=None):
    """
    Input: simple daily returns (not log). Monthly return is multiplicative.
    Output index = last *trading* day available in the data for each month
    (so it matches weights/costs indexed by last trading day).
    """
    if factor_cols is None:
        factor_cols = list(daily_factors.columns)

    x = daily_factors[factor_cols].copy().dropna(how="all")
    month = x.index.to_period("M")  # It assigns the same Period to all days within that calendar month.


    monthly = (1.0 + x).groupby(month).prod() - 1.0

    # index = last trading day in each month 
    assert x.index.is_monotonic_increasing
    month_end_dates = x.groupby(month).tail(1).index # Gets the last trading day in each calendar month from the actual data
    monthly.index = month_end_dates

    return monthly.sort_index()


def mv_simple(daily_ret,cols=None,gamma=5.0,ridge=1e-10,sum_to_one_constraint=True,long_only=True,maxiter=10000,):
    if cols is not None:
        daily_ret = daily_ret[cols]

    Rm_df = compute_monthly_factor_returns_from_daily(daily_ret)
    asset_cols = list(Rm_df.columns)

    R = Rm_df.to_numpy()
    T, K = R.shape
    if K < 1:
        raise ValueError("Need at least 1 asset column")

    Sigma = np.cov(R, rowvar=False) + float(ridge) * np.eye(K)
    mu = np.mean(R, axis=0)

    def objective(w):
        w = np.asarray(w, float)
        return float(0.5 * gamma * (w @ Sigma @ w) - (w @ mu))

    constraints = []
    if sum_to_one_constraint:
        constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

    bounds = [(0.0, 1.0)] * K if long_only else None
    x0 = np.ones(K) / K

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter)},
    )

    w = np.asarray(res.x, float)
    w_ser = pd.Series(w, index=asset_cols, name="w_mv")

    port_monthly = pd.Series(R @ w, index=Rm_df.index, name="port_mv")

    return {
        "weights": w_ser,
        "monthly_asset_returns": Rm_df,
        "port_monthly": port_monthly,
        "Sigma": Sigma,
        "opt_result": res,
        "sum_to_one_constraint": sum_to_one_constraint,
        "long_only": long_only,
    }


def mw_cvar_simple(daily_ret,cols=None,beta=0.95,gamma=5.0,sum_to_one_constraint=True,
    long_only=True,upper_bound=1.0,solver="ECOS",verbose=False,
):
# https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html
    if cols is not None:
        daily_ret = daily_ret[cols]

    Rm_df = compute_monthly_factor_returns_from_daily(daily_ret)
    asset_cols = list(Rm_df.columns)

    R = Rm_df.to_numpy()  # (T,K)
    T, K = R.shape
    if K < 1:
        raise ValueError("Need at least 1 asset column")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1)")

    mu_vec = R.mean(axis=0)

    w = cp.Variable(K)
    alpha = cp.Variable()
    u = cp.Variable(T)

    losses = -(R @ w)  # portfolio loss series

    cvar = alpha + (1.0 / ((1.0 - beta) * T)) * cp.sum(u)

    constraints = [u >= 0, u >= losses - alpha]

    if sum_to_one_constraint:
        constraints.append(cp.sum(w) == 1.0)
    if long_only:
        constraints.append(w >= 0.0)
    if upper_bound is not None:
        constraints.append(w <= float(upper_bound))

    objective = cp.Minimize(float(gamma) * cvar - mu_vec @ w)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=bool(verbose))

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("CVaR optimization failed. Status: %s" % prob.status)

    w_val = np.asarray(w.value, float).reshape(-1)
    w_ser = pd.Series(w_val, index=asset_cols, name="w_cvar")

    port_monthly = pd.Series(R @ w_val, index=Rm_df.index, name="port_cvar")

    return {
        "weights": w_ser,
        "monthly_asset_returns": Rm_df,
        "port_monthly": port_monthly,
        "mu_vec": pd.Series(mu_vec, index=asset_cols, name="mu"),
        "beta": beta,
        "gamma": gamma,
        "cvar_loss": float(cvar.value),
        "var_alpha": float(alpha.value),
        "expected_return": float(mu_vec @ w_val),
        "opt_status": prob.status,
        "opt_value": float(prob.value),
        "sum_to_one_constraint": sum_to_one_constraint,
        "long_only": long_only,
        "upper_bound": upper_bound,
    }


def Markowitz_with_turnover_TC_diffobj(
    daily_factors, factor_cols,
    gamma=5.0, ridge=1e-8, nonneg=False,
    market_vol_proxy="Mkt-rf",
    c_tc=0.0021, abs_eps=1e-10,
    use_drift_turnover=True,              # TRUE = drift turnover (RL,Vol paper style); FALSE = no costs at all
    bounded_b: bool = False,
    use_avg_vol_proxy= True
                        ):

    _printed = False

    def _check_month_loss_once(monthly_F, rv2_aligned):
        nonlocal _printed
        if _printed:
            return
        _printed = True

        months_all = monthly_F.index.to_period("M")
        months_valid = rv2_aligned[rv2_aligned.notna()].index.to_period("M")

        lost = pd.Index(months_all).difference(pd.Index(months_valid))


    factor_cols = [c for c in factor_cols if c != market_vol_proxy] # for optimization (input to markowitz), safety
    print(f"Using factor columns: {factor_cols}")
    monthly_F = compute_monthly_factor_returns_from_daily(daily_factors, factor_cols=factor_cols) # monthly returns, do not include MKT-RF
    m = daily_factors.dropna().index.to_period("M")

    # If market_vol_proxy column doesn't exist (simulated data), 
    # #create equal-weighted market index or average sigma
   
    if market_vol_proxy not in daily_factors.columns:
        if use_avg_vol_proxy:
            #average monthly volatility across factors
            rv_factors = daily_factors[factor_cols].groupby(m).std(ddof=1)  
            rv = rv_factors.mean(axis=1)                                  
            mkt_col = "avg_factor_vol"
        else:
            # equal-weighted factor index, then monthly volatility
            market_index = daily_factors[factor_cols].mean(axis=1)
            daily_factors_with_mkt = daily_factors.copy()
            daily_factors_with_mkt[market_vol_proxy] = market_index
            mkt_col = market_vol_proxy
            rv = daily_factors_with_mkt[mkt_col].groupby(m).std(ddof=1)
    else:
        mkt_col = market_vol_proxy
        rv = daily_factors[mkt_col].groupby(m).std(ddof=1)
    
    rv2 = rv.shift(1)  # use lagged volatility (no look-ahead)

    #align lagged vol to monthly_F using MONTH PERIOD, then restore trading month-end dates 
    rv2 = rv2.reindex(monthly_F.index.to_period("M")) # 
    rv2.index = monthly_F.index
    assert rv2.index.equals(monthly_F.index) # month end


    _check_month_loss_once(monthly_F, rv2)


    # align and drop months where lagged vol is missing
    valid = rv2.notna()
    Rm = monthly_F.loc[valid]
    s = rv2.loc[valid].to_numpy().reshape(-1, 1)# we need to reshape cause of the division part

    R = Rm.to_numpy()  # (T x K)
    K = R.shape[1]
    T = R.shape[0]
    s = np.maximum(s, 1e-12)  # safety
    ### we use these above R,s inside the objective function and
    def smooth_abs(x):
        return np.sqrt(x*x + abs_eps)

    def eta_to_theta(eta):
        a = eta[:K]
        b = eta[K:]
        theta = a[None, :] + b[None, :] / s # already lagged, it will create s many theta 
        # Normalize to sum to 1
        denom = np.sum(theta, axis=1, keepdims=True)
        theta = theta / np.where(np.abs(denom) > 1e-12, denom, 1.0)
        return theta

    def taus_series(eta):
        # TRUE = drift turnover (RL-style pre-trade drift turnover)
        # FALSE = do not even calculate costs: if we do not calculate costs, we should stuck to long-short also
        if not use_drift_turnover:
            return np.zeros(R.shape[0], dtype=float)

        theta = eta_to_theta(eta)
        T = theta.shape[0] # time dimension, it is the same as R.shape[0]
        taus = np.zeros(T, dtype=float)
        if T <= 1:
            return taus

        for t in range(1, T):
            # last month target (post-trade)
            w_prev_post = theta[t-1]

            # realized monthly returns during last month
            R_prev = R[t-1]

            # drift to pre-trade weights
            g = 1.0 + R_prev
            numer = w_prev_post * g
            denom = float(np.sum(numer))
            w_pre = numer / denom if abs(denom) > 1e-12 else w_prev_post

            # trade to current target
            w_target = theta[t]

            # turnover
            tau_t = 0.5 * float(np.sum(smooth_abs(w_target - w_pre)))
            taus[t] = tau_t # save monthly costs--> use it for net retunrs, we do not multiply with c_tc here!
            # later for cost calculation you need

        return taus

    def TC(eta):
        # TRUE = drift turnover (RL-style pre-trade drift turnover)
        # FALSE = do not even calculate costs: if we do not calculate costs, we should stuck to long-short also
        if not use_drift_turnover:
            return 0.0

        taus = taus_series(eta)
        return c_tc * float(np.mean(taus[1:])) if taus.shape[0] > 1 else 0.0 # no drift in the beginning

    def objective(eta):
        theta = eta_to_theta(eta)
        rp = np.sum(theta * R, axis=1)      # monthly portfolio return series, no R_ext!!!
        mu_p = float(rp.mean())
        var_p = float(rp.var(ddof=0))
        return 0.5 * gamma * var_p - mu_p + TC(eta)

    # initial guess
    guess = np.ones(2 * K) / (2 * K)

    constraints = []  # keep empty (no equality constraint): add up to one effect?

    # If unconditional benchmark (no timing): force b = 0 AND enforce sum(a)=1 (fully invested)
    # Fully invested constraint is needed for valid cost calculation (drift assumes normalized weights)
    if bounded_b:
        constraints.append({"type": "eq", "fun": lambda eta: np.sum(eta[:K]) - 1.0})  # sum(a) = 1

    # bounds: if nonneg True => both a,b >= 0; if False, a >= 0 but b unbounded for volatility timing
    if nonneg:
        bounds = [(0.0, None)] * (2 * K)  # Both a and b >= 0
    else:
        bounds = [(0.0, None)] * K + [(None, None)] * K  # a >= 0, b unbounded

    # If unconditional benchmark (bounded_b=True): force b = 0
    if bounded_b:
        bounds = [(0, None)] * K + [(0, 0)] * K  # Force all b values to exactly 0
    result = minimize(
        objective,
        x0=guess,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 100000}
    )

    eta = result.x
    a = eta[:K]
    b = eta[K:]

    theta = eta_to_theta(eta)
    weights = pd.DataFrame(theta, index=Rm.index, columns=Rm.columns) # Rm.index ,month end

    # portfolio return in month t: sum_k w_{k,t} * R_{k,t}
    port_ret_gross = (weights * Rm).sum(axis=1)

    # Substract the costs (month-by-month, consistent with drift turnover)
    taus = taus_series(eta) # just
    costs = c_tc * taus # monthly
    port_ret_net = port_ret_gross - pd.Series(costs, index=Rm.index)

    return {
        "a": pd.Series(a, index=Rm.columns, name="a"),
        "b": pd.Series(b, index=Rm.columns, name="b"),
        "weights": weights,
        "portfolio_returns": port_ret_gross,
        "portfolio_returns_gross": port_ret_gross,
        "portfolio_returns_net": port_ret_net,
        "turnover": pd.Series(taus, index=Rm.index, name="turnover"),
        "costs": pd.Series(costs, index=Rm.index, name="costs"), # monthly costs
        "opt_result": result,
        "TC_in_sample": TC(eta),
        "use_drift_turnover": use_drift_turnover,
        "bounded_b": bounded_b
    }



def Markowitz_with_turnover_TC_diffobj_daily(
    daily_factors: pd.DataFrame,
    factor_cols,
    gamma=3.0, # Match with RL
    ridge=1e-8,
    nonneg=False,
    market_vol_proxy="Mkt-rf",
    c_tc=0.0001, # Match with RL
    abs_eps=1e-10,
    use_drift_turnover=True,
    bounded_b: bool = False,          
    use_avg_vol_proxy: bool = True,   
    use_assetwise_vol_proxy: bool = False,  # per-asset vol scaling, similar to Demiguel
    vol_window: int = 20,            
    maxiter: int = 100000,):
    '''
    The data generating process is tricky We have monthly regimes but daily VAR. It means, depending on the
    rolling window size, we can mix regimes. Interpretation?
    We dont use compute_monthly_factor_returns_from_daily
    
    '''
    factor_cols = [c for c in factor_cols if c != market_vol_proxy]
    X = daily_factors[factor_cols].copy().dropna(how="any")  # daily simple returns
    if X.empty:
        raise ValueError("No usable daily data after dropping NaNs.")
    idx = X.index

    if (market_vol_proxy is not None) and (market_vol_proxy in daily_factors.columns):
        proxy = daily_factors.loc[idx, market_vol_proxy].astype(float)
        s = proxy.rolling(int(vol_window)).std(ddof=1)
    else:
        if use_assetwise_vol_proxy:
            # per-asset rolling std, s is DataFrame (T,N)!!
            s = X.rolling(int(vol_window)).std(ddof=1)
        else:
            if use_avg_vol_proxy:
                # average of per-asset rolling vols
                s = X.rolling(int(vol_window)).std(ddof=1).mean(axis=1)
            else:
                # equal-weighted index vol
                proxy = X.mean(axis=1)
                s = proxy.rolling(int(vol_window)).std(ddof=1)

    s = s.shift(1)  # day ahead: should be low signal
    valid = s.notna() # bollean

    # If s is DataFrame, valid is DataFrame -> reduce to row mask
    if isinstance(valid, pd.DataFrame):
        valid = valid.all(axis=1)

    Xv = X.loc[valid]
    
    # Build sv with correct shape
    if use_assetwise_vol_proxy and isinstance(s, pd.DataFrame):
        # sv: (T,N)
        sv = s.loc[valid, Xv.columns].to_numpy()
    else:
        # sv: (T,1)
        sv = s.loc[valid].to_numpy().reshape(-1, 1) # no columns to specify
    sv = np.maximum(sv, 1e-12)

    R = Xv.to_numpy()  # (T,N)
    T, N = R.shape

    def smooth_abs(x):
        return np.sqrt(x * x + abs_eps)

    def eta_to_theta(eta):
        a = eta[:N]
        b = eta[N:]
        theta = a[None, :] + b[None, :] / sv  # (T,N)

        denom = np.sum(theta, axis=1, keepdims=True)
        theta = theta / np.where(np.abs(denom) > 1e-12, denom, 1.0)
        return theta

    def taus_series(eta):
        if not use_drift_turnover:
            return np.zeros(T, dtype=float)

        theta = eta_to_theta(eta)
        taus = np.zeros(T, dtype=float)
        if T <= 1:
            return taus

        for t in range(1, T):
            w_prev_post = theta[t - 1]   
            R_prev = R[t - 1]           

            # drift to today's pre-trade weights
            g = 1.0 + R_prev
            numer = w_prev_post * g
            denom = float(np.sum(numer))
            w_pre = numer / denom if abs(denom) > 1e-12 else w_prev_post

            # trade to today's target
            w_target = theta[t]
            taus[t] = 0.5 * float(np.sum(smooth_abs(w_target - w_pre)))

        return taus

    def TC(eta):
        if not use_drift_turnover:
            return 0.0
        taus = taus_series(eta)
        return c_tc * float(np.mean(taus[1:])) if taus.shape[0] > 1 else 0.0

    def objective(eta):
        theta = eta_to_theta(eta)
        rp = np.sum(theta * R, axis=1)        
        mu_p = float(rp.mean())
        var_p = float(rp.var(ddof=0)) + float(ridge)
        return 0.5 * gamma * var_p - mu_p + TC(eta)

    guess = np.ones(2 * N) / (2 * N)

    constraints = []
    if bounded_b:
        constraints.append({"type": "eq", "fun": lambda eta: np.sum(eta[:N]) - 1.0})

    if nonneg:
        bounds = [(0.0, None)] * (2 * N)
    else:
        bounds = [(0.0, None)] * N + [(None, None)] * N

    if bounded_b:
        bounds = [(0.0, None)] * N + [(0.0, 0.0)] * N  # b = 0

    res = minimize(
        objective,
        x0=guess,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": int(maxiter)},
    )

    eta = res.x
    theta = eta_to_theta(eta)
    weights = pd.DataFrame(theta, index=Xv.index, columns=Xv.columns) # same logic

    port_ret_gross = (weights * Xv).sum(axis=1)         
    taus = taus_series(eta)
    costs = c_tc * taus                               
    port_ret_net = port_ret_gross - pd.Series(costs, index=Xv.index)

    # Store vol_signal with correct format
    if use_assetwise_vol_proxy and isinstance(s, pd.DataFrame):
        vol_signal = s.loc[valid, Xv.columns].copy()
    else:
        vol_signal = pd.Series(s.loc[valid].values, index=Xv.index, name="s_lag")
    
    return {
        "a": pd.Series(eta[:N], index=Xv.columns, name="a"),
        "b": pd.Series(eta[N:], index=Xv.columns, name="b"),
        "weights": weights,
        "portfolio_returns_gross": port_ret_gross,
        "portfolio_returns_net": port_ret_net,
        "turnover": pd.Series(taus, index=Xv.index, name="turnover"),
        "costs": pd.Series(costs, index=Xv.index, name="costs"),
        "vol_signal": vol_signal,
        "opt_result": res,
        "use_drift_turnover": use_drift_turnover,
        "bounded_b": bounded_b,
        "vol_window": int(vol_window),
    }