# %%
def plot_saved_snapshot(self, tag: str = "train_end", max_points: int = 200_000):
    """
    Load snapshot_{tag}.npz from self.save_path_root and plot:
      - histograms of reward, returns-aware turnover (tc), and fee
      - per-regime bar charts (mean reward, tc, fee)
      - average policy (stacked bars of risky assets + cash) per regime
      - critic quantile curves averaged over grid, per regime
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    base = os.path.join(self.save_path_root, f"snapshot_{tag}.npz")
    if not os.path.exists(base):
        print(f"[plot_saved_snapshot] Not found: {base}")
        return

    Z = np.load(base, allow_pickle=False)
    policy_mean = Z["policy_mean"]   # (S,M,K,N)
    cash_mean   = Z["cash_mean"]     # (S,M,K)
    reward      = Z["reward"]        # (S,M,K)
    tc          = Z["tc"]            # (S,M,K)
    fee         = Z["fee"]           # (S,M,K)
    critic_q    = Z["critic_q"]      # (S,M,K,Q)

    Wgrid = Z["Wgrid"]               # (S,N)
    Rgrid = Z["Rgrid"]               # (M,N)

    # Scalars / per-regime aggregates (present in file)
    avg_reward  = float(Z["avg_reward"])
    avg_fee     = float(Z["avg_fee"])
    u_mean_per_k   = Z["u_mean_per_k"]      # (K,)
    tc_mean_per_k  = Z["tc_mean_per_k"]     # (K,)
    fee_mean_per_k = Z["fee_mean_per_k"]    # (K,)

    S, M, K, N = policy_mean.shape
    Q = critic_q.shape[-1]

    # Flatten (optionally subsample for speed)
    def _flatten_sample(a):
        x = a.reshape(-1)
        if x.size > max_points:
            idx = np.random.default_rng(0).choice(x.size, max_points, replace=False)
            x = x[idx]
        return x

    reward_f = _flatten_sample(reward)
    tc_f     = _flatten_sample(tc)
    fee_f    = _flatten_sample(fee)

    # ---- Print summary ----
    print("=== Snapshot summary ===")
    print(f"path: {base}")
    print(f"S={S}, M={M}, K={K}, N={N}, Q={Q}")
    print(f"avg_reward={avg_reward:.6f} | avg_fee={avg_fee:.6f}")
    print("per-regime means:")
    for k in range(K):
        print(f"  k={k}: u={u_mean_per_k[k]:.6f}, tc={tc_mean_per_k[k]:.6f}, fee={fee_mean_per_k[k]:.6f}")

    # ---- Histograms ----
    plt.figure(); plt.hist(reward_f, bins=60); plt.title("Reward (u)"); plt.xlabel("u"); plt.ylabel("count"); plt.tight_layout()
    plt.figure(); plt.hist(tc_f,     bins=60); plt.title("Returns-aware turnover τ = 0.5‖w - w_pre‖₁"); plt.xlabel("τ"); plt.ylabel("count"); plt.tight_layout()
    plt.figure(); plt.hist(fee_f,    bins=60); plt.title("Fee = c · τ"); plt.xlabel("fee"); plt.ylabel("count"); plt.tight_layout()

    # ---- Per-regime bars ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].bar(np.arange(K), u_mean_per_k);   ax[0].set_title("Mean reward per regime"); ax[0].set_xlabel("regime"); ax[0].set_ylabel("u")
    ax[1].bar(np.arange(K), tc_mean_per_k);  ax[1].set_title("Mean τ per regime");      ax[1].set_xlabel("regime"); ax[1].set_ylabel("τ")
    ax[2].bar(np.arange(K), fee_mean_per_k); ax[2].set_title("Mean fee per regime");    ax[2].set_xlabel("regime"); ax[2].set_ylabel("fee")
    fig.tight_layout()

    # ---- Average policy per regime (stacked risky + cash) ----
    # avg over (S,M) → (K,N) and cash → (K,)
    w_avg_KN = policy_mean.mean(axis=(0,1))          # (K,N)
    cash_avg_K = cash_mean.mean(axis=(0,1))          # (K,)
    # stack risky + cash for plotting
    fig, ax = plt.subplots(figsize=(max(6, K*1.5), 4))
    idx = np.arange(K)
    bottom = np.zeros(K)
    for n in range(N):
        ax.bar(idx, w_avg_KN[:, n], bottom=bottom, label=f"asset {n}")
        bottom += w_avg_KN[:, n]
    ax.bar(idx, cash_avg_K, bottom=bottom, label="cash")
    ax.set_title("Average portfolio composition per regime")
    ax.set_xlabel("regime"); ax.set_ylabel("share"); ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    # ---- Critic quantiles (avg over S,M) per regime ----
    vq_KQ = critic_q.mean(axis=(0,1))  # (K,Q)
    plt.figure(figsize=(max(6, K*1.8), 3.6))
    for k in range(K):
        plt.plot(np.linspace(0, 1, Q, endpoint=False)[1:], vq_KQ[k], label=f"k={k}")  # tau grid (1..Q)/Q; your array excludes 0
    plt.title("Critic quantiles (avg over grid)"); plt.xlabel("τ"); plt.ylabel("V_τ"); plt.legend(frameon=False); plt.tight_layout()

    plt.show()
