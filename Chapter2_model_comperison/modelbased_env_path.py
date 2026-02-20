import numpy as np
import pandas as pd
import tensorflow as tf

EPS = 1e-12


def w_pre_from_log_risky(w_prev, r_now, eps=EPS):

    w_prev = np.asarray(w_prev, float).reshape(-1)
    g = np.exp(np.asarray(r_now, float).reshape(-1))
    numer = w_prev * g
    denom = float(numer.sum()) + eps
    return numer / denom


def multivariate_t_eps_with_target_cov(rng, Sigma, df):
    """
    Draw eps ~ multivariate Student-t with df, scaled so Cov(eps)=Sigma for df>2.

      z - N(0, Sigma)
      u - chi2(df)
      t_raw = z / sqrt(u/df)
      scale by sqrt((df-2)/df) so covariance matches Sigma when df>2.
    """
    Sigma = np.asarray(Sigma, float)
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal(Sigma.shape[0])
    u = rng.chisquare(df)
    t_raw = (L @ z) / np.sqrt(u / df)
    if df > 2:
        return np.sqrt((df - 2.0) / df) * t_raw
    return t_raw


def simulate_markov_chain(Q, T, k0=0, rng=None):
    """
    Simulate k_t Markov chain with transition matrix Q.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Q = np.asarray(Q, float)
    K = Q.shape[0] # number of states
    k = np.empty(T, dtype=int)
    k[0] = int(k0)
    for t in range(1, T):
        k[t] = rng.choice(K, p=Q[k[t - 1]])
    return k


def simulate_rs_var1_monthly_regimes_RS_SV_T(
    T_days,
    Q,
    c_list,
    Phi,
    Sigma_list,
    k0=0,
    burn_in_months=50,
    rng=None,
    start_date="2000-01-03",
    df_list=None,
    sv_rho=0.97,
    sv_sigma=0.20,
    logh_mu_list=None,
):
    """
    Regime-switching VAR(1) with:
      - monthly Markov regimes (inside each month there is no regime change)
      - daily VAR(1) in log returns
      - Student-t shocks (df per regime)
      - one-factor stochastic volatility scale h_t:
            log h_t = mu_k + rho*(log h_{t-1}-mu_k) + sigma*eta_t
        so Cov(eps_t | k_t, h_t) = h_t * Sigma_k
    """
    rng = np.random.default_rng(0) if rng is None else rng

    Q = np.asarray(Q, float)
    c_list = np.asarray(c_list, float)
    Sigma_list = np.asarray(Sigma_list, float)

    K = Q.shape[0] # number of states
    N = c_list.shape[1] # number of assets  

    # Phi can be (N,N) shared or (K,N,N) regime-specific
    Phi_arr = np.asarray(Phi, float)
    if Phi_arr.ndim == 2:
        Phi_arr = np.repeat(Phi_arr[None, :, :], K, axis=0)
    else:
        assert Phi_arr.shape == (K, N, N), "Phi must be (N,N) or (K,N,N)."

    # defaults
    if df_list is None:
        df_list = np.full(K, 8.0)
    df_list = np.asarray(df_list, float).reshape(-1)
    assert df_list.size == K

    if logh_mu_list is None:
        logh_mu_list = np.array([-2.5, -1.5, -0.5], dtype=float)[:K]
    logh_mu_list = np.asarray(logh_mu_list, float).reshape(-1)
    assert logh_mu_list.size == K

    # business-day sampling dates
    sample_dates = pd.bdate_range(start_date, periods=int(T_days))
    start_month = sample_dates[0].to_period("M")
    end_month = sample_dates[-1].to_period("M")

    sample_months = pd.period_range(start=start_month, end=end_month, freq="M")
    T_months = len(sample_months)

    # burn-in months
    full_months = pd.period_range(
        start=(start_month - int(burn_in_months)),
        end=end_month,
        freq="M"
    )
    TT_months = len(full_months)

    # monthly regimes
    k_month_full = simulate_markov_chain(Q, TT_months, k0=k0, rng=rng)

    # expand monthly regimes to business days
    all_dates_full = []
    k_days_full = []
    for m_idx, per in enumerate(full_months):
        month_start = per.to_timestamp(how="start")
        month_end = per.to_timestamp(how="end")
        dts = pd.bdate_range(month_start, month_end)
        all_dates_full.append(dts)
        k_days_full.append(np.full(len(dts), int(k_month_full[m_idx]), dtype=int))

    all_dates_full = all_dates_full[0].append(all_dates_full[1:]) if len(all_dates_full) > 1 else all_dates_full[0]
    k_days_full = np.concatenate(k_days_full, axis=0)

    # simulate daily log returns
    r_full = np.zeros((len(all_dates_full), N), dtype=float)
    h_full = np.zeros(len(all_dates_full), dtype=float)

    # init log-vol at regime mean
    k0_day = int(k_days_full[0])
    logh = float(logh_mu_list[k0_day])
    h_full[0] = float(np.exp(logh))

    for t in range(1, len(all_dates_full)):
        kt = int(k_days_full[t])
        df = float(df_list[kt])

        # stochastic volatility AR(1) around regime-specific mean
        logh = (
            float(logh_mu_list[kt])
            + float(sv_rho) * (logh - float(logh_mu_list[kt]))
            + float(sv_sigma) * rng.standard_normal()
        )
        h = float(np.exp(logh))
        h_full[t] = h

        # heavy-tailed shock with target covariance Sigma_list[kt]
        eps = multivariate_t_eps_with_target_cov(rng, Sigma_list[kt], df=df)

        # apply SV scaling (now Cov = h * Sigma_k)
        eps = np.sqrt(h) * eps

        r_full[t] = c_list[kt] + Phi_arr[kt] @ r_full[t - 1] + eps

    # slice out requested sample dates (drop burn-in)
    pos = all_dates_full.get_indexer(sample_dates)
    r_days = r_full[pos]
    k_days = k_days_full[pos]
    k_month = k_month_full[-T_months:]
    h_days = h_full[pos]

    return r_days, k_days, k_month, sample_dates, sample_months, h_days


def build_state_obs(w_prev, r_now, k, K):
    """
    State: [w_prev(N), r_now(N), onehot(K)]
    
    Note: h_t (stochastic volatility) is NOT included because:
      - It's unobservable in practice (latent factor)
      - Regime k already captures vol regime (Bull/Neutral/Bear have different base vols)
      - Recent returns r_now contain information about realized volatility
    """
    onehot = np.zeros(int(K), float)
    onehot[int(k)] = 1.0
    return np.concatenate([np.asarray(w_prev, float), np.asarray(r_now, float), onehot], axis=0)


class RSVARPathSampler:
    """
    PATH-BASED sampler using simulate_rs_var1_monthly_regimes_RS_SV_T.

    Produces sequential transitions (s, s_next, a, r).  We dont need q cause we just sampling. If we have enough data we are ok
    """

    def __init__(
        self,
        c_list,
        Phi,
        Sigma_list,
        Q,
        c=0.0021,
        gamma_crra=5.0,
        T_days=2048,
        burn_in_months=50,
        df_list=None,
        sv_rho=0.97,
        sv_sigma=0.20,
        logh_mu_list=None,
        k0=0,
        seed=0,
        initial_w_prev=None,
        start_date="2000-01-03",
    ):
        self.c_list = np.asarray(c_list, float)          # (K, N)
        self.Phi = np.asarray(Phi, float)                # (N, N) or (K, N, N)
        self.Sigma_list = np.asarray(Sigma_list, float)  # (K, N, N)
        self.Q = np.asarray(Q, float)                    # (K, K)

        self.K, self.N = self.c_list.shape
        self.c = float(c)
        self.gamma = float(gamma_crra)

        assert self.Sigma_list.shape == (self.K, self.N, self.N)
        assert self.Q.shape == (self.K, self.K)
        assert np.allclose(self.Q.sum(axis=1), 1.0), "Rows of Q must sum to 1."

        self.T_days = int(T_days)
        self.burn_in_months = int(burn_in_months)
        self.start_date = str(start_date)

        self.df_list = df_list
        self.sv_rho = float(sv_rho)
        self.sv_sigma = float(sv_sigma)
        self.logh_mu_list = logh_mu_list
        self.k0 = int(k0)

        if initial_w_prev is None:
            self.initial_w_prev = np.ones(self.N) / self.N
        else:
            w0 = np.asarray(initial_w_prev, float).reshape(-1)
            assert w0.size == self.N
            self.initial_w_prev = w0

        self.rng = np.random.default_rng(int(seed))

    def _utility(self, x):
        if np.isclose(self.gamma, 1.0):
            return np.log(x)
        return (x ** (1.0 - self.gamma) - 1.0) / (1.0 - self.gamma)

    def cost(self, w_target_full, w_prev_risky, r_now=None):
        w_target_full = np.asarray(w_target_full, float).reshape(-1)
        assert w_target_full.size == self.N
        if r_now is not None:
            w_pre = w_pre_from_log_risky(w_prev_risky, r_now)
        else:
            w_pre = np.asarray(w_prev_risky, float).reshape(-1)
        tau = 0.5 * float(np.sum(np.abs(w_target_full - w_pre)))
        return self.c * tau

    def _reward(self, w_full, R_risky, w_prev_risky, r_now=None):
        w_full = np.asarray(w_full, float).reshape(-1)
        Rp = float(w_full @ np.asarray(R_risky, float).reshape(-1))
        fee = self.cost(w_target_full=w_full, w_prev_risky=w_prev_risky, r_now=r_now)
        x = max(Rp - fee, 1e-8)
        return float(self._utility(x))

    @staticmethod
    def _normalize_simplex(a, eps=EPS):
        a = np.asarray(a, float).reshape(-1)
        a = np.clip(a, 0.0, np.inf)
        s = float(a.sum())
        if not np.isfinite(s) or s <= 0.0:
            return None
        return a / max(s, eps)

    def make_batch(self, actor, rng_eps=None):

        rng = self.rng if rng_eps is None else rng_eps

        r_days, k_days, _k_month, _dates, _months, h_days = simulate_rs_var1_monthly_regimes_RS_SV_T(
            T_days=self.T_days,
            Q=self.Q,
            c_list=self.c_list,
            Phi=self.Phi,
            Sigma_list=self.Sigma_list,
            k0=self.k0,
            burn_in_months=self.burn_in_months,
            rng=rng,
            start_date=self.start_date,
            df_list=self.df_list,
            sv_rho=self.sv_rho,
            sv_sigma=self.sv_sigma,
            logh_mu_list=self.logh_mu_list,
        )

        transitions = []
        w_prev = self.initial_w_prev.copy()

        for t in range(1, len(r_days)):
            r_now = r_days[t - 1] # simulated returns 
            k_now = int(k_days[t - 1]) # simulated regime at time t-1
            k_next = int(k_days[t]) # simulated regime at time t
            # h_now removed from state - unobservable latent factor

            s_obs = build_state_obs(w_prev, r_now, k_now, self.K) # state: [w_prev, r_now, onehot(k)]
            s_tf = tf.convert_to_tensor(s_obs[None, :], dtype=tf.float32)

            a_full = actor.sample_action(s_tf).numpy()[0]
            a_full = self._normalize_simplex(a_full) # dont need, safety
 
            R_next = np.exp(r_days[t]) #not log return, for reward calculation!
            rwd = self._reward(
                w_full=a_full,
                R_risky=R_next,
                w_prev_risky=w_prev,
                r_now=r_now,
            )

            s_next = build_state_obs(a_full, r_days[t], k_next, self.K)
            transitions.append((s_obs, s_next, a_full, rwd))

            w_prev = a_full.copy()

        return transitions
