
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import random
import requests
import io
import zipfile
from scipy.stats import norm

    
# we need daily and not monthly returns
def daily_portfolio_returns_from_monthly_weights(daily_returns: pd.DataFrame,
                                                 W_monthly: pd.DataFrame) -> pd.Series:

    R = daily_returns.dropna(how="any").copy()
    month = R.index.to_period("M")

    W = W_monthly.copy()
    W["m"] = W.index.to_period("M")
    W = W.set_index("m").sort_index()

    out = []
    months = sorted(month.unique())
    for m_trade in months:
        if m_trade not in W.index:
            continue

        w = W.loc[m_trade]
        cols = [c for c in w.index if c in R.columns]
        Rt = R.loc[month == m_trade, cols]
        if Rt.empty:
            continue

        out.append(pd.Series(Rt.values @ w[cols].values, index=Rt.index))

    return pd.concat(out).sort_index()


def portfolio_stats_paper_style(returns,
                                periods_per_year=252,
                                rf_annual=0.0,
                                target=0.0,
                                alpha=0.95):

    r = pd.Series(returns).dropna().astype(float).to_numpy()
    if len(r) < 2:
        raise ValueError("need at least 2 observations")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    mu = float(np.mean(r))
    sigma2 = float(np.var(r, ddof=0))
    sigma = float(np.sqrt(sigma2))

    mu_ann = mu * periods_per_year
    sigma2_ann = sigma2 * periods_per_year
    sigma_ann = float(np.sqrt(sigma2_ann))

    downside = (r[r < target] - target)
    semivar = 0.0 if downside.size == 0 else float(np.mean(downside**2))
    semidev_ann = float(np.sqrt(semivar * periods_per_year))

    # VaR/CVaR
    q = 1.0 - alpha
    VaR = float(np.quantile(r, q))
    tail = r[r <= VaR]
    CVaR = VaR if tail.size == 0 else float(np.mean(tail))

    # drawdowns
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / peak
    pos_dd = dd[dd > 0]
    avg_dd = 0.0 if pos_dd.size == 0 else float(np.mean(pos_dd))

    # excess mean (per-period rf from annual rf)
    rf_per = (1.0 + rf_annual)**(1.0 / periods_per_year) - 1.0
    excess_ann_mean = (mu - rf_per) * periods_per_year

    sharpe = np.nan if sigma_ann == 0 else float(excess_ann_mean / sigma_ann)
    sortino = np.nan if semidev_ann == 0 else float(excess_ann_mean / semidev_ann)

    # tail-adjusted Sharpe (NO annualization of CVaR/mVaR)
    ta_sharpe_cvar = np.nan if CVaR == 0 else float(excess_ann_mean / abs(CVaR))

    # Cornish-Fisher modified VaR
    if sigma == 0:
        skew = 0.0
        exkurt = 0.0
    else:
        xc = r - mu
        m3 = float(np.mean(xc**3))
        m4 = float(np.mean(xc**4))
        skew = m3 / (sigma**3)
        kurt = m4 / (sigma**4)
        exkurt = kurt - 3.0

    z = float(norm.ppf(q))
    z_cf = (z
            + (1/6)  * (z**2 - 1)   * skew
            + (1/24) * (z**3 - 3*z) * exkurt
            - (1/36) * (2*z**3 - 5*z) * (skew**2))

    mVaR = float(mu + sigma * z_cf)
    ta_sharpe_mvar = np.nan if mVaR == 0 else float(excess_ann_mean / abs(mVaR))

    return {
        "Ann. Mean (%)": 100 * mu_ann,
        "Ann. StdDev (%)": 100 * sigma_ann,
        "Ann. SemiDev (%)": 100 * semidev_ann,
        "CVaR 95% (%)": 100 * CVaR,
        "Avg DD (%)": 100 * avg_dd,
        "VaR 95% (%)": 100 * VaR,
        "Sharpe (ann.)": sharpe,
        "Sortino (ann.)": sortino,
        "Tail-Adj Sharpe (CVaR95)": ta_sharpe_cvar,
        "Tail-Adj Sharpe (mVaR95)": ta_sharpe_mvar,
    }


def make_table_for_portfolios(portfolios: dict,
                              periods_per_year=252,
                              rf_annual=0.0,
                              target=0.0,
                              alpha=0.95) -> pd.DataFrame:
    rows = [
        "Ann. Mean (%)",
        "Ann. StdDev (%)",
        "Ann. SemiDev (%)",
        "CVaR 95% (%)",
        "Avg DD (%)",
        "VaR 95% (%)",
        "Sharpe (ann.)",
        "Sortino (ann.)",
        "Tail-Adj Sharpe (CVaR95)",
        "Tail-Adj Sharpe (mVaR95)",
    ]

    out = pd.DataFrame(index=rows)
    for name, r in portfolios.items():
        st = portfolio_stats_paper_style(r, periods_per_year=periods_per_year,
                                         rf_annual=rf_annual, target=target, alpha=alpha)
        out[name] = [st[k] for k in rows]
    return out

def df_to_booktabs_latex(df: pd.DataFrame, caption=None, label=None) -> str:
    latex = df.to_latex(
        escape=True,
        float_format=lambda x: f"{x:.2f}",
        column_format="l" + "r"*df.shape[1],
        bold_rows=False
    )
    # convert to booktabs style
    latex = latex.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule").replace("\\bottomrule", "\\bottomrule")
    if caption or label:
        # wrap in table environment if requested
        body = latex
        lines = ["\\begin{table}[!htbp]", "\\centering"]
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append(body.strip())
        lines.append("\\end{table}")
        latex = "\n".join(lines)
    return latex



def set_numpy_determinism(seed=0):
    '''
    Similarly to RL set up random sets and go over a list of seeds
    '''
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)



def check_phi_stability(Phi, tol=1.0 - 1e-8):
    Phi = np.asarray(Phi, float)
    if Phi.ndim == 2:
        eigs = np.linalg.eigvals(Phi)
        max_abs = float(np.max(np.abs(eigs)))
        if max_abs >= tol:
            raise RuntimeError(f"Unstable VAR: max |eig| = {max_abs:.6f} >= 1.0")
        print(f"Phi stable: max |eig| = {max_abs:.6f}")
        return
    if Phi.ndim == 3:
        for k in range(Phi.shape[0]):
            eigs = np.linalg.eigvals(Phi[k])
            max_abs = float(np.max(np.abs(eigs)))
            if max_abs >= tol:
                raise RuntimeError(f"Unstable VAR in regime {k}: max |eig| = {max_abs:.6f} >= 1.0")
        print("Phi stable in all regimes.")
        return
    raise ValueError("Phi must be 2D or 3D")

# Check current Phi_fixed


def load_seed_outputs(
    base_path,
    tau_levels=(0.1, 0.5, 0.9),
    seed_list=(53,274,1234,89),
    folder_template="20250911_final_weighted_q_spwise_standard_tanh_{seed}_{tau_str}",
    filename="snapshot_train_end_summary_full.pkl",
                        ):

    base_path = Path(base_path) # Creating a path object
    tau_levels_str = [str(t).replace(".", "") for t in tau_levels]

    out, paths = {}, {}
    for tau_str in tau_levels_str:
        out[tau_str] = {}
        paths[tau_str] = {}
        for seed in seed_list:
            folder_name = folder_template.format(seed=seed, tau_str=tau_str)
            fpath = base_path / folder_name / filename
            if not fpath.exists():
                print(f"File not found: {fpath}")
                continue
            with open(fpath, "rb") as fh:
                payload = pickle.load(fh)
            out[tau_str][seed] = payload
            paths[tau_str][seed] = str(fpath)
    return out, paths#




def summarize_path_metrics_paper_style(
    dfs_by_scenario,
    periods_per_year=252,
    rf_annual=0.0,
    target=0.0,
    alpha=0.95,
    use_key="port_ret_net",
    REGIME_NAME=None,
):
    """
    Builds path-level rows for:
      - Regime-specific metrics (Bull/Neutral/Bear): computed on r[k==reg]
      - Overall metrics ("All"): computed on full r
      Scenario: BB, BN,  NB
    """
    rows = []

    for scenario, df in dfs_by_scenario.items():
        for tau_str, d_train in df.items():
            # policy , train seed
            for train_seed, d_test in d_train.items():
                # train seed, test seed
                for test_seed, payload in d_test.items():
                    if not isinstance(payload, dict):
                        continue

                    r = np.asarray(payload.get(use_key, []), float)
                    k = np.asarray(payload.get("regime_days", []), int)

                    n = min(r.size, k.size)
                    if n < 2:
                        continue
                    r = r[:n]
                    k = k[:n]

                    base = {
                        "Scenario": scenario,
                        "Tau": f"tau_{tau_str}",
                        "tau_str": tau_str,
                        "TrainSeed": int(train_seed),
                        "TestSeed": int(test_seed),
                    }
                    # take one realization of the path and compute metrics on it, by regime and overall
                    m_all = portfolio_stats_paper_style(
                        r, periods_per_year=periods_per_year, rf_annual=rf_annual, target=target, alpha=alpha
                    )
                    rows.append({
                        **base,
                        "Regime": "All",
                        "N_obs": int(n),
                        **m_all
                    })

                    for reg_id, reg_name in REGIME_NAME.items():
                        mask = (k == reg_id)
                        r_reg = r[mask]
                        m = portfolio_stats_paper_style(
                            r_reg, periods_per_year=periods_per_year, rf_annual=rf_annual, target=target, alpha=alpha
                        )
                        rows.append({
                            **base,
                            "Regime": reg_name,
                            "N_obs": int(r_reg.size),
                            **m
                        })

    return pd.DataFrame(rows)




def weighted_group_mean(df: pd.DataFrame, group_cols, weight_col: str, metric_cols):
    d = df.copy()
    out = []
    # we will build the final result as "one dictionary per group", then convert to a DataFrame

    for keys, g in d.groupby(group_cols, dropna=False):
        gw = g[weight_col].astype(float).clip(lower=0)
        denom = gw.sum()

        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["N_paths"] = int(len(g)) # training seed x test seed paths 16 this case
        row["SumWeights"] = float(denom)

        for m in metric_cols:
            # now we compute the weighted mean for EACH metric (Sharpe, CVaR, etc.)

            vals = g[m].astype(float)
            row[m] = float((vals * gw).sum() / denom) if denom > 0 else np.nan

        out.append(row)

    return pd.DataFrame(out).sort_values(group_cols).reset_index(drop=True)


def unweighted_group_mean(df: pd.DataFrame, group_cols, metric_cols):
    agg = {m: "mean" for m in metric_cols}
    out = df.groupby(group_cols, as_index=False).agg(agg)
    out["N_paths"] = df.groupby(group_cols).size().values
    return out.sort_values(group_cols).reset_index(drop=True)