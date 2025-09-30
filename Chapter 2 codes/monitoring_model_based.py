# monitoring_model_based.py
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns 
from modelbased_env import build_state_obs

EPSILON = 1e-12


class Monitoring:
    def __init__(self, agent, env, args):
        self.agent = agent
        self.env = env
        self.args = args
        self.metrics = {}

        self.save_path_root = os.path.join(args.checkpoints_dir, args.save_as_file)
        self.load_path_root = os.path.join(args.checkpoints_dir, args.load_from_file)
        os.makedirs(self.save_path_root, exist_ok=True)

        # optional episodic store, dont need
        self.episode_data = {}

    def update_metric(self, metric, value):
        self.metrics.setdefault(metric, []).append(value)

    def save_metrics(self):
        file_name = os.path.join(self.save_path_root, "metrics.pkl")
        with open(file_name, "wb") as handle:
            pickle.dump(self.metrics, handle)

    def save_brain_weights(self):
        self.agent.save_weights(self.save_path_root)

    def save_episode_distributions(self):
        save_path = os.path.join(self.save_path_root, "episode_data.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.episode_data, f)

    @staticmethod
    def _w_pre_from_log_full(w_prev_risky, r_now, r_f=1.0, eps=1e-12):
        """
        Compute pre-trade FULL weights (risky + cash) after today's realization.
        Inputs:
            w_prev_risky : (N,) risky weights held yesterday (sum <= 1)
            r_now        : (N,) today's log-returns
            r_f          : gross cash return
        Returns:
            (N+1,) vector: [pre-trade risky, pre-trade cash]

        In general this part was motivated by the Gaussian policy, we should have change it for clarity...
        """
        w_prev = np.asarray(w_prev_risky, float).reshape(-1)   # (N,)
        g = np.exp(np.asarray(r_now, float).reshape(-1))       # (N,)
        numer_risky = w_prev * g
        numer_cash = (1.0 - float(w_prev.sum())) * r_f
        denom = float(numer_risky.sum() + numer_cash)

        w_pre_risky = numer_risky / denom
        w_pre_cash = numer_cash / denom
        return np.append(w_pre_risky, w_pre_cash)

    def eval_one_state(self, w_prev, r_now, k, rng, n_scen=2048):
        """
        [ U(Rp - fee) | state=(w_prev, r_now, k) ].
        The evaulation done by one fixed state here.
        We have the input state and we just calculate the
        utility, no continution--> not much sense but ok

        We dont need, we focus o

        """
        # load the params and agent, plus environment
        env, agent = self.env, self.agent
        const_k, Phi_k, L_k = env.const_k, env.Phi_k, env.L_k
        Qmat, r_f, c, gamma = env.Q, float(env.r_f), float(env.c), float(env.gamma)
        N, K = const_k.shape[1], const_k.shape[0]

        s_obs = build_state_obs(w_prev, r_now, k, K)
        s_tf  = tf.convert_to_tensor(s_obs[None, :], dtype=tf.float32)
        # expect agent.get_mean_of_action to return (B, N+1)
        w_out = np.squeeze(agent.get_mean_of_action(s_tf).numpy(), axis=0)
        # ensure valid simplex for safety, dont need we have dirichlet
        w_full = np.clip(w_out, 0.0, 1.0)
        ssum = float(w_full.sum())
        if ssum > 0.0:
            w_full = w_full / ssum
        # split risky/cash
        # dont need projection weights add up
        w_risky, w_cash = w_full[:-1], float(w_full[-1])

        # pre-trade FULL weights and additive fee components
        w_pre_full = self._w_pre_from_log_full(w_prev, r_now, r_f=r_f)
        tau = 0.5 * float(np.sum(np.abs(w_full - w_pre_full)))
        fee = c * tau  # additive proportional fee

        # CRNs: we use a separate one here for testing
        eps = rng.standard_normal((N, n_scen))  # (N, n_scen)

        # expectation over kp and shocks
        u_acc = 0.0
        for kp in range(K):
            q = float(Qmat[k, kp])
            mu_kp = const_k[kp] + Phi_k[kp] @ r_now                      # (N,)
            r_next = mu_kp[:, None] + L_k[kp] @ eps                      # (N, n_scen)
            R_next = np.exp(r_next)                                      # gross

            risky_payoff = w_risky @ R_next                              # (n_scen,)
            Rp = risky_payoff + w_cash * r_f                             # (n_scen,)
            x = np.maximum(Rp - fee, 1e-8)

            if np.isclose(gamma, 1.0):
                u = np.log(x)
            else:
                u = (np.power(x, 1.0 - gamma) - 1.0) / (1.0 - gamma)

            u_acc += q * float(np.mean(u)) # we need to multipply it to mimic the expected outer regime utility
            # NOTE: this is just a step ahead, no continuation.  

        return u_acc, tau, fee, w_risky, w_cash

    def evaluate_on_training_testing_data_stepwise(self, save: bool = False, tag: str = "eval"):
        """
        evaulate at the end of the training
        Strategy: increase the number of shocks and use different.
        Go over all of the grid and estimate the expectation
        """
        
        env, agent = self.env, self.agent # load the trained model
        # Wgrid is defined on the weight of Risky assets..Cash is not included
        # Effect?L Its fine, cause the agent will still explore by taking cash into account. The look-up table is just a mapping, what i track
        Wgrid = np.asarray(env.Wgrid, float)  # (S, N)
        Rgrid = np.asarray(env.Rgrid, float)  # (M, N)
        S, N = Wgrid.shape
        M = Rgrid.shape[0]
        K = int(env.K)



        policy_mean = np.zeros((S, M, K, N), float)   # risky only
        cash_mean   = np.zeros((S, M, K),    float)
        reward_mc   = np.zeros((S, M, K),    float)
        tc_grid     = np.zeros((S, M, K),    float)
        fee_grid    = np.zeros((S, M, K),    float)
        critic_out  = np.zeros((S, M, K, agent.Nq), float)

        rng_eval = np.random.default_rng(12345)
        n_scen_eval = 2048
        '''
        So basically we are making a look up table
        '''

        for s in range(S):
            w_prev = Wgrid[s]
            for m in range(M):
                r_now = Rgrid[m]
                for k in range(K): # this is just one step ahead utility, no continution, just to check
                    u, tau, fee, w_risky, w_cash = self.eval_one_state(
                        w_prev, r_now, k, rng_eval, n_scen=n_scen_eval
                    )
                    reward_mc[s, m, k] = u
                    tc_grid[s, m, k]   = tau
                    fee_grid[s, m, k]  = fee
                    policy_mean[s, m, k, :] = w_risky
                    cash_mean[s, m, k]      = w_cash

                    # value function
                    onehot = np.zeros(K, float); onehot[k] = 1.0
                    state_vec = np.concatenate([w_prev, r_now, onehot], 0)[None, :].astype(np.float32)
                    critic_out[s, m, k, :] = agent.critic_network(state_vec).numpy().squeeze()

        summary = {
            "policy_mean": policy_mean,
            "cash_mean":   cash_mean,
            "reward":      reward_mc,
            "tc":          tc_grid,
            "fee":         fee_grid,
            "critic_q":    critic_out,
            "avg_reward":  float(reward_mc.mean()),
            "avg_tc":      float(tc_grid.mean()),
            "avg_fee":     float(fee_grid.mean()),
            "avg_tau":     float(tc_grid.mean()),
            "per_regime": {
                "tc_mean_per_k":  tc_grid.mean(axis=(0, 1)),
                "fee_mean_per_k": fee_grid.mean(axis=(0, 1)),
                "u_mean_per_k":   reward_mc.mean(axis=(0, 1)),
            },
            "Wgrid": Wgrid, "Rgrid": Rgrid,
        }


        if save:
            os.makedirs(self.save_path_root, exist_ok=True)
            base = os.path.join(self.save_path_root, f"snapshot_{tag}")
            out_path = base + "_summary_full.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
        return summary
    
    

    @staticmethod
    def _plot_shares(df):
        share_cols = [c for c in df.columns if c.endswith('_share')]
        df_shares = df[['date'] + share_cols].copy()
        df_shares.set_index('date', inplace=True)
        ax = df_shares.plot(kind='area', stacked=True, figsize=(14, 6), alpha=0.8)
        ax.set_ylabel('Proportion'); ax.set_title('Stacked Area Plot of Time Series Data')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
