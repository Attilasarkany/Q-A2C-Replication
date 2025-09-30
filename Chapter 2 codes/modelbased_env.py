import numpy as np
import tensorflow as tf

"""
Dirichlet policy version
- Actor.sample_action(s) should return a FULL allocation on the simplex of length N+1
  (N risky assets + 1 cash), i.e. w_full >= 0 and sum(w_full) = 1.
- Fees are charged as a turnover today's pre-trade weights w_pre_full computed from
  yesterday's risky weights and today's log-returns (cash grows at r_f). See function
  w_pre_from_log_full

  NOTE: when we create the transitions we just add the risky assets
  as we did in the expected case but the turnover is over all of the weights 
  (including risk free), it is not a problem if we just wanna compare. 
  We descreased dimensionality without loosnig anything. 
  Also riskty assets already contains all information ( reward too in transition),
 adding cash would be redundant and doesnt change the Markov property
"""

def w_pre_from_log_full(w_prev_risky, r_now, r_f=1.0, eps=1e-12):
    """
    Compute pre-trade FULL weights (risky + cash) after today's return realization.
    Inputs:
        w_prev_risky : (N,) risky weights held yesterday (sum <= 1)
        r_now        : (N,) log-returns realized today
        r_f          : gross risk-free (cash) return
    Returns:
        (N+1,) vector: [pre-trade risky weights, pre-trade cash]
    """
    w_prev = np.asarray(w_prev_risky, float).reshape(-1)     # (N,)
    g = np.exp(np.asarray(r_now, float).reshape(-1))         # (N,)
    numer_risky = w_prev * g
    numer_cash  = (1.0 - float(w_prev.sum())) * r_f
    denom = float(numer_risky.sum() + numer_cash)
 
    w_pre_risky = numer_risky / denom
    w_pre_cash  = numer_cash  / denom
    return np.append(w_pre_risky, w_pre_cash)                # (N+1,)

def build_state_obs(w_prev, r_now, k, K):
    """
    Build state as [w_prev (N), r_now (N, log), onehot(K)].
    w_prev: risky-only previous weights (N,)
    r_now : current log-returns (N,)
    k     : current regime index: vector i.e (0,1,0): so k will be the indication of the the place in the vector
    K     : number of regimes: 3
    """
    onehot = np.zeros(K, float); onehot[int(k)] = 1.0
    return np.concatenate([w_prev, r_now, onehot], axis=0)

class RSVARBatchSampler:
    def __init__(self, const_k, Phi_k, Sigma_k, Q,
                 Wgrid, Rgrid, r_f=1.001, c=1e-3, gamma_crra=3.0,
                 n_scen=64, seed=0):
        # model params
        self.const_k = np.asarray(const_k, float)   # (K, N)
        self.Phi_k   = np.asarray(Phi_k,   float)   # (K, N, N)
        self.Sigma_k = np.asarray(Sigma_k, float)   # (K, N, N)
        self.Q       = np.asarray(Q,       float)   # (K, K), Transition probability
        self.Wgrid   = np.asarray(Wgrid,   float)   # (S, N)
        self.Rgrid   = np.asarray(Rgrid,   float)   # (M, N)

        self.K, self.N = self.const_k.shape # risky assets
        self.S = self.Wgrid.shape[0] # weight grid
        self.M = self.Rgrid.shape[0] # price grid

        self.r_f   = float(r_f)
        self.c     = float(c)
        self.gamma = float(gamma_crra)

        # checks
        assert np.allclose(self.Q.sum(axis=1), 1.0), "Rows of Q must sum to 1."


        # chol factors
        self.L_k = np.stack([np.linalg.cholesky(Sk) for Sk in self.Sigma_k], axis=0)

        self.n_scen = int(n_scen)
        self.rng = np.random.default_rng(int(seed))

    def _utility(self, x):
        '''
        x is:wealth
        '''
        if np.isclose(self.gamma, 1.0): return np.log(x)
        return (x**(1.0 - self.gamma) - 1.0) / (1.0 - self.gamma)

    def cost(self, w_target_full, w_prev_risky, r_now=None):
        """
        Additive proportional fee on FULL weights:
            tau = 0.5 * || w_target_full - w_pre_full ||_1
            fee = c * tau
            ye, well, again we have previous risky only on the grid
        """
        w_target_full = np.asarray(w_target_full, float).reshape(-1)
        assert w_target_full.size == self.N + 1, "Expect N+1 weights (risky + cash)."
        if r_now is not None:
            w_pre_full = w_pre_from_log_full(w_prev_risky, r_now, r_f=self.r_f)
        else:
            cash_prev = max(0.0, 1.0 - float(np.sum(w_prev_risky)))
            w_pre_full = np.append(np.asarray(w_prev_risky, float).reshape(-1), cash_prev)
        tau = 0.5 * float(np.sum(np.abs(w_target_full - w_pre_full)))
        return self.c * tau

    def _reward(self, w_full, R_risky, w_prev_risky, r_now=None):
        """
        CRRA utility of next-step gross return after additive fee.
        Inputs:
            w_full      : (N+1,) risky + cash allocation used today
            R_risky     : (N,)    gross returns for risky assets (exp(log r))
            w_prev_risky: (N,)    yesterday's risky weights
            r_now       : (N,)    today's log-returns (for w_pre computation), one day shift
        """
        w_full = np.asarray(w_full, float).reshape(-1)
        w_risky, w_cash = w_full[:-1], float(w_full[-1]) # first two fill be risky and the last one will be the chosen risk-free
        # portfolio gross before fee
        Rp = float(w_risky @ R_risky) + w_cash * self.r_f
        # additive fee
        fee = self.cost(w_target_full=w_full, w_prev_risky=w_prev_risky, r_now=r_now)
        x = max(Rp - fee, 1e-8)
        return float(self._utility(x))

    def make_batch(self, actor, rng_eps=None):
        """
        Build transitions:
        For each (w_prev, r_now, k) enumerate kp in {0..K-1}, draw n_scen shocks,
        and attach sample weight q = Q[k, kp].
        Returns a list of (s, s_next, a_full, reward, q).

        We have to creat an environment bascially:
        state, next state, reward, action and because we dont sample enough
        i think we need to add the transition probability.

        """
        if rng_eps is None:
            rng_eps = self.rng

        transitions = []
        eps = rng_eps.standard_normal((self.N, self.n_scen))   # (N, n_scen)

        for s_idx in range(self.S):
            w_prev = self.Wgrid[s_idx]                         # (N,)
            for m_idx in range(self.M):
                r_now = self.Rgrid[m_idx]                      # (N,)
                for k in range(self.K):
                    s_obs = build_state_obs(w_prev, r_now, k, self.K) # onehot( we are at a specific regime)
                    s_tf  = tf.convert_to_tensor(s_obs[None, :], dtype=tf.float32)

                    # Dirichlet actor returns FULL allocation (N+1) !!!
                    # so the state is risky,r_now and which regime
                    a_full = actor.sample_action(s_tf).numpy()[0]   # (N+1,)
                    # safety
                    assert a_full.size == self.N + 1
                    # risky part becomes next state's "previous weights"
                    # Tricky: This logic is from the Gaussian set up
                    # but there we had risk free as residual
                    # Even if now we add the Risky only to the next state
                    # when we calculate the turnover cost we do
                    # take cash into account--> rebalaing is expensive here too!!
                    w_prev_next = a_full[:-1].copy()                 # (N,) only risky!
                    # 

                    for kp in range(self.K):
                        q = float(self.Q[k, kp]) # transitiong from k to k'

                        # VAR(1) under next regime kp
                        xi_kp  = self.const_k[kp] + self.Phi_k[kp] @ r_now      # (N,)
                        u_kp   = self.L_k[kp] @ eps                              # (N, n_scen)
                        r_next = xi_kp[:, None] + u_kp                           # (N, n_scen)
                        R_next = np.exp(r_next)                                  # gross
                        # super unefficient 
                        for j in range(self.n_scen):
                            rj = r_next[:, j]
                            Rj = R_next[:, j]

                            rwd = self._reward(
                                w_full=a_full,
                                R_risky=Rj,
                                w_prev_risky=w_prev,
                                r_now=r_now
                            )
                            # next regime: previously chosen weight, r_next (log), new regime
                            # NOTE: Under current regime we created a new one, so state and next state are dependent
                            # NOTE: we need to save the reward, and the action cause of the RL set up
                            # NOTE: cause we are not sampling correctly, we must save the transition probability q for training
                            s_next = build_state_obs(w_prev_next, rj, kp, self.K)
                            transitions.append((s_obs, s_next, a_full, rwd, q))

        return transitions
