import numpy as np


def _prepare_exogenous_variables(self):
    """
    Precompute r_{t+1} and R_{t+1} for every (m,k,scenario) once.
    Also pre-sample k' for each (k,scenario).
    m: weight
    k: 
    """


    # Shock o VAR(1)
    eps= self.rng.standard_normal((self.N, self.n_scen))

    # 2) transform per regime: u_k = L_k @ eps  → (K, N, n_scen)
    u_per_k = np.einsum('kij,jn->kin', self.L_k, eps)

    # 3) xi_all(m,k,i) = const_k[k,i] + sum_j Phi_k[k,i,j] * r_now[m,j]  → (M, K, N)
    xi_all = np.einsum('kij,mj->mki', self.Phi_k, self.Rgrid) + self.const_k[None, :, :]

    # 4) r_next_all(m,k,i,n) = xi_all(m,k,i) + u_per_k(k,i,n)  → (M, K, N, n_scen)
    r_next_all = xi_all[:, :, :, None] + u_per_k[None, :, :, :]

    # 5) gross factors
    R_risky_all = np.exp(r_next_all)

    # 6) pre-sample next regimes per (k, scenario)
    kprime_bank = np.empty((self.K, self.n_scen), dtype=np.int32)
    for k in range(self.K):
        kprime_bank[k] = self.rng.choice(self.K, size=self.n_scen, p=self.Q[k])

    # stash
    self._eps_bank     = eps_bank
    self._u_per_k      = u_per_k
    self._xi_all       = xi_all
    self._r_next_all   = r_next_all
    self._R_risky_all  = R_risky_all
    self._kprime_bank  = kprime_bank
    self._exo_ready    = True

def refresh_exogenous_bank(self, seed=None):
    """Optional: call if you want to regenerate the exogenous bank every N epochs."""
    if seed is not None:
        self.rng = np.random.default_rng(int(seed))
    self._exo_ready = False
    self._prepare_exogenous_bank()




# --- choose action once for this (s_idx, m_idx, k) ---
s_obs = build_state_obs(w_prev, r_now, k, self.K)
s_tf  = tf.convert_to_tensor(s_obs[None, :], dtype=tf.float32)
a_raw = actor.sample_action(s_tf).numpy()[0]
w_risky = project_to_simplex_leq1(a_raw)
w_prev_next = w_risky.copy()

    # --- sample next regimes for each scenario ---
    k_primes = self.rng.choice(self.K, size=self.n_scen, p=self.Q[k])

    for j in range(self.n_scen):
        kp = int(k_primes[j])  # NEXT regime

        # simulate r_{t+1} under NEXT regime kp
        xi_kp = self.const_k[kp] + self.Phi_k[kp] @ r_now         # (N,)
        u_kp  = self.L_k[kp] @ eps[:, j]                          # (N,)
        rj    = xi_kp + u_kp                                      # (N,)
        Rj    = np.exp(rj)                                        # (N,)

        # reward must use NEXT regime returns too
        rwd = self._reward(w_risky, Rj, w_prev, r_now=r_now, k=kp)

        # next state uses w_t, r_{t+1} and kp
        s_next = build_state_obs(w_prev_next, rj, kp, self.K)

        transitions.append((s_obs, s_next, a_raw, rwd))
