# monitoring_model_based_path.py  (PATH-BASED TESTING, SAVE PER-SEED)
import os
import pickle
import numpy as np
import tensorflow as tf

from modelbased_env_path import (
    build_state_obs,
    w_pre_from_log_risky,
    simulate_rs_var1_monthly_regimes_RS_SV_T,
)

EPS = 1e-12


class Monitoring:


    def __init__(self, agent, env, args):
        self.agent = agent
        self.env = env
        self.args = args
        self.metrics = {}

        self.save_path_root = os.path.join(args.checkpoints_dir, args.save_as_file)
        self.load_path_root = os.path.join(args.checkpoints_dir, args.load_from_file)
        os.makedirs(self.save_path_root, exist_ok=True)

        self.episode_data = {}

    def update_metric(self, metric, value):
        self.metrics.setdefault(metric, []).append(value)

    def save_metrics(self):
        file_name = os.path.join(self.save_path_root, "metrics.pkl")
        with open(file_name, "wb") as handle:
            pickle.dump(self.metrics, handle)

    def save_brain_weights(self):
        self.agent.save_weights(self.save_path_root)

    def _project_simplex(self, w):
        w = np.asarray(w, float).reshape(-1)
        w = np.clip(w, 0.0, np.inf)
        s = float(w.sum())
        if (not np.isfinite(s)) or s <= 0.0:
            return np.ones_like(w) / float(w.size)
        return w / s

    def _utility(self, x):
        gamma = float(self.env.gamma)
        x = np.maximum(x, 1e-8)
        if np.isclose(gamma, 1.0):
            return np.log(x)
        return (np.power(x, 1.0 - gamma) - 1.0) / (1.0 - gamma)

    def _rollout_one_seed(
        self,
        seed: int,
        T_days: int,
        burn_in_months: int = 50,
        start_date: str = "2000-01-03",
        use_mean_action: bool = True,
        store_daily_weights: bool = True,
    ):
 
        env, agent = self.env, self.agent
        rng = np.random.default_rng(int(seed))

        r_days, k_days, k_month, dates, sample_months, h_days = simulate_rs_var1_monthly_regimes_RS_SV_T(
            T_days=int(T_days),
            Q=env.Q,
            c_list=env.c_list,
            Phi=env.Phi,
            Sigma_list=env.Sigma_list,
            k0=int(getattr(env, "k0", 0)),
            burn_in_months=int(burn_in_months),
            rng=rng,
            start_date=str(start_date),
            df_list=getattr(env, "df_list", None),
            sv_rho=float(getattr(env, "sv_rho", 0.97)),
            sv_sigma=float(getattr(env, "sv_sigma", 0.20)),
            logh_mu_list=getattr(env, "logh_mu_list", None),
        )

        K = int(env.K)
        N = int(env.N)

        w_prev = np.asarray(getattr(env, "initial_w_prev", np.zeros(N)), float).reshape(-1)
        w_prev = self._project_simplex(w_prev)

        nT = len(r_days)
        # portfolio return realized at t from weights chosen at t-1
        port_ret_gross = np.zeros(nT, dtype=float)
        port_ret_net = np.zeros(nT, dtype=float)
        tau_series = np.zeros(nT, dtype=float)
        fee_series = np.zeros(nT, dtype=float)
        util_reward = np.zeros(nT, dtype=float)

        if store_daily_weights:
            w_series = np.zeros((nT, N), dtype=float)
            w_series[0] = w_prev
        else:
            w_series = None

        # we start from t=1 since need next return
        for t in range(1, nT):
            r_now = r_days[t - 1]        # log returns observed now (state)
            k_now = int(k_days[t - 1])
            r_next = r_days[t]           # next log returns (realization)
            k_next = int(k_days[t])
            h_now = float(h_days[t - 1])
            h_next = float(h_days[t])

            s_obs = build_state_obs(w_prev, r_now, h_now, k_now, K) #
            s_tf = tf.convert_to_tensor(s_obs[None, :], dtype=tf.float32)
            critic_out = agent.critic_network(s_tf, training=False).numpy().squeeze()

            if use_mean_action:
                w_raw = np.squeeze(agent.get_mean_of_action(s_tf).numpy(), axis=0)
            else:
                w_raw = np.squeeze(agent.act(s_tf).numpy(), axis=0)

            w = self._project_simplex(w_raw)

            w_pre = w_pre_from_log_risky(w_prev, r_now)
            tau = 0.5 * float(np.sum(np.abs(w - w_pre)))
            fee = float(env.c) * tau

            # realized next-step risky gross returns
            R_next = np.exp(r_next)              # gross per asset
            Rp_gross = float(np.dot(w, R_next))  # gross portfolio return (>=0)

            # net gross after fee (fee subtracts in gross space, like your env reward)
            Rp_net = max(Rp_gross - fee, 1e-8)

            # store in *simple return* for later Sharpe etc
            port_ret_gross[t] = Rp_gross - 1.0
            port_ret_net[t] = Rp_net - 1.0

            tau_series[t] = tau
            fee_series[t] = fee
            util_reward[t] = float(self._utility(Rp_net)) # you realize at t+1

            w_prev = w.copy()
            if store_daily_weights:
                w_series[t] = w_prev

        out = {
            "seed": int(seed),
            "T_days": int(T_days),
            "burn_in_months": int(burn_in_months),
            "start_date": str(start_date),
            "use_mean_action": bool(use_mean_action),

            "dates": dates,
            "log_returns": r_days,
            "h_days": h_days,
            "regime_days": k_days,
            "regime_month": k_month,
            "sample_months": sample_months,

            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "turnover": tau_series,
            "fee": fee_series,
            "utility_reward": util_reward,
            "critic": critic_out,
            "weights": w_series,  # can be None if store_daily_weights=False
        }
        return out

    def test_on_seed_list(
        self,
        test_seeds,
        T_days: int,
        burn_in_months: int = 50,
        start_date: str = "2000-01-03",
        use_mean_action: bool = True,
        store_daily_weights: bool = True,
        tag: str = "test",
        save: bool = True,
    ):

        os.makedirs(self.save_path_root, exist_ok=True)

        results_index = []
        for seed in list(test_seeds):
            res = self._rollout_one_seed(
                seed=int(seed),
                T_days=int(T_days),
                burn_in_months=int(burn_in_months),
                start_date=str(start_date),
                use_mean_action=bool(use_mean_action),
                store_daily_weights=bool(store_daily_weights),
            )

            results_index.append(
                {
                    "seed": int(seed),
                    "file": f"{tag}_seed_{int(seed)}.pkl",
                    "T_days": int(T_days),
                    "use_mean_action": bool(use_mean_action),
                }
            )

            if save:
                out_path = os.path.join(self.save_path_root, f"{tag}_seed_{int(seed)}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save an index file listing all produced outputs
        if save:
            idx_path = os.path.join(self.save_path_root, f"{tag}_index.pkl")
            with open(idx_path, "wb") as f:
                pickle.dump(results_index, f, protocol=pickle.HIGHEST_PROTOCOL)

        return results_index
