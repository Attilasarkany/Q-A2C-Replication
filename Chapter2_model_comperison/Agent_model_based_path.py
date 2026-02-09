
from model_dirichlet_path import QACDirichletAgent
from modelbased_env_path import RSVARPathSampler
from monitoring_model_based_path import Monitoring

import os, json, logging, random, warnings, datetime, sys, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')

LOG = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)

EPS = 1e-12


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_float_list(s):
    if s is None:
        return None
    return [float(x) for x in s.split(',') if x.strip()]


def parse_int_list(s):
    if s is None:
        return None
    return [int(x) for x in s.split(',') if x.strip()]


def parse_str_list(s):
    if s is None:
        return None
    return [x.strip() for x in s.split(',') if x.strip()]


parser = argparse.ArgumentParser()
parser.add_argument("--render_each", default=100, type=int)
parser.add_argument("--evaluate", default=True, type=str2bool)
parser.add_argument("--seed", default=None, type=int)

parser.add_argument("--episodes", default=30, type=int)
parser.add_argument("--gamma", default=0.96, type=float)
parser.add_argument("--learning_tau", type=float, default=0.2)
parser.add_argument('--tau_levels', type=int, default=10)

# learning rates (two-time scale is a design choice you already used)
parser.add_argument('--critic_lr_start', type=float, default=0.01)
parser.add_argument('--critic_lr_end', type=float, default=0.001)
parser.add_argument('--actor_lr_start', type=float, default=0.001)
parser.add_argument('--actor_lr_end', type=float, default=0.0001)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--entropy_reg', type=float, default=0.01)  # keep for agent args compat

parser.add_argument('--transaction_cost', type=float, default=0.0021)
parser.add_argument('--gamma_crra', type=float, default=5.0)     # your chosen default
parser.add_argument('--n_scen', type=int, default=128)           # not used in path sampler but kept for compat

# minibatch
parser.add_argument('--batch_size', type=int, default=256, help="Minibatch size; <=0 uses full batch")

# saving
parser.add_argument('--checkpoints_dir', type=str, default='./training_outcome')
parser.add_argument('--load_weights', type=str2bool, default=False)
parser.add_argument('--load_from_file', type=str, default='test')
parser.add_argument('--save_weights', type=str2bool, default=True)
parser.add_argument('--save_as_file', type=str, default='test')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--add_tic_date', type=str2bool, default=False)
parser.add_argument('--reward_scaling', type=float, default=1)
parser.add_argument('--batches_per_epoch', type=float, default=200)

parser.add_argument('--actor_loss', type=str, nargs='?', default='weighted_quantile',
                    choices=['advantage', 'is_negative', 'weighted_quantile', 'original', 'expectation', 'power'])
parser.add_argument('--critic_type', type=str, nargs='?', default='standard',
                    choices=['standard', 'monte_carlo_dropout', 'bayesian'])

# parallel batch sweep
parser.add_argument('--parallel_batch', type=str2bool, default=False)
parser.add_argument('--parallel_taus', type=str, default=None)
parser.add_argument('--parallel_seeds', type=str, default=None)
parser.add_argument('--parallel_max_workers', type=int, default=None)
parser.add_argument('--parallel_q_matrices', type=str, default=None)

# Q-matrix choice
parser.add_argument('--q_matrix', type=str, default='bull_bear',
                    choices=['bull_bear', 'neutral_bear', 'bull_neutral'])

# shocks (kept for compat)
parser.add_argument('--shared_shocks', type=str2bool, default=False)

# -------- NEW: path lengths + testing seed list --------
parser.add_argument('--train_T_days', type=int, default=2048, help="Training path length (business days)")
parser.add_argument('--train_burn_in_months', type=int, default=50)
parser.add_argument('--train_start_date', type=str, default="2000-01-03")

parser.add_argument('--test_T_days', type=int, default=252 * 20, help="Test path length (business days)")
parser.add_argument('--test_burn_in_months', type=int, default=0)
parser.add_argument('--test_start_date', type=str, default="2000-01-03")
parser.add_argument('--test_seeds', type=str, default="", help="Comma-separated list of test seeds (e.g. 1000,2000,3000)")
parser.add_argument('--test_use_mean_action', type=str2bool, default=True)
parser.add_argument('--test_store_daily_weights', type=str2bool, default=True)
parser.add_argument('--test_tag', type=str, default="test")


def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=0):
    if seed is None:
        seed = 0
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def train(env_model, agent,
          epochs=30, batch_size=128,
          rng=None, log_every=1, monitoring=None):

    if rng is None:
        rng = np.random.default_rng(0)

    N = int(env_model.N)

    hist = {
        "actor_loss": [], "critic_loss": [], "reward": [], "turnover": [], "fee": [],
        "n_transitions": []
    }

    for ep in range(1, epochs + 1):
        transitions = env_model.make_batch(agent.actor_network, rng_eps=rng)
        n_trans = len(transitions)
        LOG.info("[epoch %03d] built %d transitions (PATH)", ep, n_trans)
        hist["n_transitions"].append(n_trans)

        if n_trans == 0:
            hist["actor_loss"].append(np.nan)
            hist["critic_loss"].append(np.nan)
            hist["reward"].append(np.nan)
            hist["turnover"].append(np.nan)
            hist["fee"].append(np.nan)
            continue
        # keep random shuffling
        idx = rng.permutation(n_trans)
        transitions = [transitions[i] for i in idx]

        a_loss_acc = 0.0
        c_loss_acc = 0.0
        nb = 0

        bs = n_trans if batch_size <= 0 else int(batch_size)

        for start in range(0, n_trans, bs):
            end = min(start + bs, n_trans)
            batch = transitions[start:end]
            if not batch:
                break

            a_l, c_l = agent.learn(batch)
            a_loss_acc += float(a_l)
            c_loss_acc += float(c_l)
            nb += 1

        rs = np.array([t[3] for t in transitions], dtype=float)
        r_mean = float(np.mean(rs)) if rs.size else np.nan

        taus = []
        for t in transitions:
            # Path sampler returns (s_obs, s_next, a_full, rwd)
            s_obs = np.asarray(t[0], float)
            a_full = t[2]
            w_prev = np.asarray(s_obs[:N], dtype=float)
            r_now = np.asarray(s_obs[N:N + N], dtype=float)
            w_full = np.asarray(a_full, float).reshape(-1)  # (N,) risky-only
            tau = float(env_model.cost(w_target_full=w_full, w_prev_risky=w_prev, r_now=r_now)) / max(env_model.c, EPS)
            taus.append(tau)

        tc_avg = float(np.mean(taus)) if taus else 0.0
        fee_avg = float(env_model.c * tc_avg)

        a_mean = (a_loss_acc / nb) if nb > 0 else 0.0
        c_mean = (c_loss_acc / nb) if nb > 0 else 0.0

        hist["actor_loss"].append(a_mean)
        hist["critic_loss"].append(c_mean)
        hist["reward"].append(r_mean)
        hist["turnover"].append(tc_avg)
        hist["fee"].append(fee_avg)

        if monitoring is not None:
            monitoring.update_metric("Actor loss", a_mean)
            monitoring.update_metric("Critic loss", c_mean)
            monitoring.update_metric("Avg reward (path avg)", r_mean)
            monitoring.update_metric("Avg turnover (no cost)", tc_avg)
            monitoring.update_metric("Avg fee (with cost)", fee_avg)

        if ep % log_every == 0:
            print(f"[epoch {ep:03d}] "
                  f"actor_loss={a_mean:.4f} | critic_loss={c_mean:.4f} | "
                  f"r_mean={r_mean:.6f} | tau={tc_avg:.6f} | fee={fee_avg:.6f}")

    return a_mean, c_mean, r_mean, tc_avg, fee_avg, hist


def _build_train_args_for_parallel(tau, seed, q_name, base_args):
    tau_code = str(tau).replace('.', '')
    file_name = f"{base_args.save_as_file}_{q_name}_{seed}_{tau_code}"
    return [
        "--learning_tau", str(tau),
        "--gamma", str(base_args.gamma),
        "--save_as_file", file_name,
        "--episodes", str(base_args.episodes),
        "--add_tic_date", str(base_args.add_tic_date),
        "--critic_lr_start", str(base_args.critic_lr_start),
        "--critic_lr_end", str(base_args.critic_lr_end),
        "--critic_type", str(base_args.critic_type),
        "--rho", str(base_args.rho),
        "--mode", "train",
        "--transaction_cost", str(base_args.transaction_cost),
        "--gamma_crra", str(base_args.gamma_crra),
        "--batch_size", str(base_args.batch_size),
        "--seed", str(seed),
        "--actor_lr_start", str(base_args.actor_lr_start),
        "--actor_lr_end", str(base_args.actor_lr_end),
        "--actor_loss", str(base_args.actor_loss),
        "--entropy_reg", str(base_args.entropy_reg),
        "--q_matrix", q_name,
        "--parallel_batch", "False",
        "--parallel_taus", "",
        "--parallel_seeds", "",
        "--parallel_q_matrices", "",

        # pass through train/test args too
        "--train_T_days", str(base_args.train_T_days),
        "--train_burn_in_months", str(base_args.train_burn_in_months),
        "--train_start_date", str(base_args.train_start_date),
        "--test_T_days", str(base_args.test_T_days),
        "--test_burn_in_months", str(base_args.test_burn_in_months),
        "--test_start_date", str(base_args.test_start_date),
        "--test_seeds", str(base_args.test_seeds),
        "--test_use_mean_action", str(base_args.test_use_mean_action),
        "--test_store_daily_weights", str(base_args.test_store_daily_weights),
        "--test_tag", str(base_args.test_tag),
        "--evaluate", str(base_args.evaluate),
    ]


def _run_agent_subprocess(tau, seed, q_name, base_args):
    script_path = os.path.abspath(__file__)
    cmd = [sys.executable, script_path] + _build_train_args_for_parallel(tau, seed, q_name, base_args)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}")
    return result.stdout


def run_qac_batch_parallel(taus, seeds, q_names, base_args, max_workers=None):
    jobs = [(tau, seed, q_name) for tau in taus for seed in seeds for q_name in q_names]
    outputs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_agent_subprocess, tau, seed, q_name, base_args): (tau, seed, q_name)
            for tau, seed, q_name in jobs
        }
        for future in as_completed(future_map):
            tau, seed, q_name = future_map[future]
            try:
                outputs.append((tau, seed, q_name, future.result()))
            except Exception as exc:
                outputs.append((tau, seed, q_name, f"FAILED: {exc}"))
    return outputs


def main(env_model, agent, monitoring, args):
    save_root = os.path.join(args.checkpoints_dir, args.save_as_file)
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    rng = np.random.default_rng(int(getattr(args, "seed", 0) if args.seed is not None else 0))

    LOG.info("state shape: %s", agent.actor_network.model.input_shape)
    LOG.info("args:\n%s", json.dumps(vars(args), indent=2))

    if getattr(args, "load_weights", False):
        load_root = getattr(monitoring, "load_path_root", save_root)
        LOG.info("loading weights from %s", load_root)
        agent.load_weights(load_root)
    else:
        LOG.info("starting new training")

    a_mean, c_mean, r_mean, tc_avg, fee_avg, history = train(
        env_model=env_model,
        agent=agent,
        epochs=args.episodes,
        batch_size=args.batch_size,
        rng=rng,
        log_every=max(1, getattr(args, "log_every", 1)),
        monitoring=monitoring
    )

    if getattr(args, "save_weights", True):
        LOG.info("saving weights to %s", save_root)
        agent.save_weights(save_root + os.sep)
        if hasattr(monitoring, "save_brain_weights"):
            monitoring.save_brain_weights()

    if monitoring is not None:
        monitoring.save_metrics()

    if getattr(args, "evaluate", False):
        test_seeds = parse_int_list(getattr(args, "test_seeds", "")) or []
        if len(test_seeds) == 0:
            LOG.info("evaluate=True but test_seeds is empty -> skipping test_on_seed_list()")
        else:
            LOG.info("Running test_on_seed_list: seeds=%s", test_seeds)
            monitoring.test_on_seed_list(
                test_seeds=test_seeds,
                T_days=int(args.test_T_days),
                burn_in_months=int(args.test_burn_in_months),
                start_date=str(args.test_start_date),
                use_mean_action=bool(args.test_use_mean_action),
                store_daily_weights=bool(args.test_store_daily_weights),
                tag=str(args.test_tag),
                save=True,
            )
            LOG.info("Saved per-seed test outputs under: %s", save_root)

    with open(os.path.join(save_root, "last_run.txt"), "w") as f:
        f.write("=== Last Run ===\n")
        f.write(f"timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"episodes: {args.episodes}\n")
        f.write(f"learning_tau: {getattr(args, 'learning_tau', None)}\n")
        f.write(f"avg_actor_loss: {a_mean:.6f}\n")
        f.write(f"avg_critic_loss: {c_mean:.6f}\n")
        f.write(f"avg_reward_path: {r_mean:.8f}\n")
        f.write(f"avg_turnover: {tc_avg:.8f}\n")
        f.write(f"avg_fee: {fee_avg:.8f}\n")
        f.write(f"evaluate: {getattr(args, 'evaluate', False)}\n")
        f.write(f"test_T_days: {getattr(args, 'test_T_days', None)}\n")
        f.write(f"test_seeds: {getattr(args, 'test_seeds', '')}\n")

    if getattr(args, "evaluate", False):
        try:
            import matplotlib.pyplot as plt
            plt.figure(); plt.plot(history["actor_loss"]); plt.title("Actor loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
            plt.figure(); plt.plot(history["critic_loss"]); plt.title("Critic loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
        except Exception as e:
            LOG.warning("Plotting failed: %s", e)


def _build_default_env_agent_monitor(args):
    '''

        Phi_fixed = np.array([[0.15, 0.10],
                            [0.10, 0.15]], dtype=float)
        Phi_k = np.tile(Phi_fixed, (3, 1, 1))

        const_k = np.array([
            [ 0.0040,  0.0030],   # Bull
            [ 0.0030,  0.0028],   # Neutral
            [-0.0090,  0.0030],   # Bear
        ], dtype=float)

        Sigma_k = np.array([
            [[0.0005,  0.00010],
            [0.00010, 0.00045]],   # Bull
            [[0.0018,  0.00000],
            [0.00000, 0.00140]],   # Neutral
            [[0.0050, -0.00300],
            [-0.00300, 0.00200]],  # Bear
        ], dtype=float)

        # SV/t parameters
        df_list = np.array([20.0, 10.0, 5.0], dtype=float)
        logh_mu_list = np.array([-1.6, -1.3, -0.7], dtype=float)
        sv_rho, sv_sigma = 0.97, 0.20

        Q_bull_bear = np.array([
            [0.74, 0.02, 0.24],
            [0.10, 0.82, 0.08],
            [0.30, 0.02, 0.68],
        ], dtype=float)

        Q_neutral_bear = np.array([
            [0.82, 0.08, 0.10],
            [0.02, 0.68, 0.30],
            [0.02, 0.24, 0.74],
        ], dtype=float)

        Q_bull_neutral = np.array([
            [0.74, 0.24, 0.02],
            [0.30, 0.68, 0.02],
            [0.10, 0.08, 0.82],
        ], dtype=float)
    '''
    # Drifts: Bull strongly favors asset1; Bear punishes asset1; asset2 stays slightly + / flat
    const_k = np.array([
        [ 0.00075,  0.00025],   # Bull
        [ 0.00030,  0.00020],   # Neutral
        [-0.00110,  0.00005],   # Bear
    ], dtype=float)

    Phi_fixed = np.array([[0.15, 0.10],
                        [0.10, 0.15]], dtype=float)
    Phi_k = np.tile(Phi_fixed, (3, 1, 1))

    # Covariances: Bear still riskier, but not "10x", and with negative correlation (asset2 hedges)
    Sigma_k = np.array([
        # Bull
        [[0.000196,  0.000050],
        [0.000050,  0.000064]],

        # Neutral
        [[0.000225,  0.000030],
        [0.000030,  0.000081]],

        # Bear  (risk up + hedge effect)
        [[0.000625, -0.000140],
        [-0.000140, 0.000196]],
    ], dtype=float)

    # SV/t: tails do the heavy lifting in Bear (so volatility timing alone isn't enough)
    df_list = np.array([40.0, 12.0, 4.0], dtype=float)
    logh_mu_list = np.array([-2.1, -1.5, -0.7], dtype=float)
    sv_rho, sv_sigma = 0.97, 0.23
    Q_bull_bear = np.array([
        [0.74, 0.02, 0.24],
        [0.10, 0.82, 0.08],
        [0.30, 0.02, 0.68],
    ], dtype=float)

    Q_neutral_bear = np.array([
        [0.82, 0.08, 0.10],
        [0.02, 0.68, 0.30],
        [0.02, 0.24, 0.74],
    ], dtype=float)

    Q_bull_neutral = np.array([
        [0.74, 0.24, 0.02],
        [0.30, 0.68, 0.02],
        [0.10, 0.08, 0.82],
    ], dtype=float)
    q_map = {
        "bull_bear": Q_bull_bear,
        "neutral_bear": Q_neutral_bear,
        "bull_neutral": Q_bull_neutral,
    }
    Q = q_map.get(getattr(args, "q_matrix", "bull_bear"), Q_bull_bear)

    K, N = const_k.shape

    env = RSVARPathSampler(
        c_list=const_k,
        Phi=Phi_k if Phi_k.ndim == 2 else Phi_k[0],
        Sigma_list=Sigma_k,
        Q=Q,
        c=args.transaction_cost,
        gamma_crra=args.gamma_crra,
        T_days=int(getattr(args, 'train_T_days', 2048)),
        burn_in_months=int(getattr(args, 'train_burn_in_months', 50)),
        seed=(args.seed if args.seed is not None else 0),

        # SV/t params
        df_list=df_list,
        sv_rho=sv_rho,
        sv_sigma=sv_sigma,
        logh_mu_list=logh_mu_list,
        k0=0,
    )

    state_shape = (2 * N + K + 1,)

    agent = QACDirichletAgent(state_shape=state_shape, stock_dimension=N, args=args)
    monitoring = Monitoring(agent=agent, env=env, args=args)
    return env, agent, monitoring


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    set_global_determinism(seed=args.seed)

    if getattr(args, "parallel_batch", False):
        taus = parse_float_list(args.parallel_taus) or [args.learning_tau]
        seeds = parse_int_list(args.parallel_seeds) or ([args.seed] if args.seed is not None else [0])
        q_names = parse_str_list(args.parallel_q_matrices) or [args.q_matrix]

        LOG.info("Launching parallel runs: taus=%s seeds=%s q_matrices=%s", taus, seeds, q_names)
        results = run_qac_batch_parallel(taus, seeds, q_names, args, max_workers=args.parallel_max_workers)

        for tau, seed, q_name, out in results:
            print(f"=== q={q_name}, tau={tau}, seed={seed} ===")
            print(out)
    else:
        env, agent, monitoring = _build_default_env_agent_monitor(args)
        main(env, agent, monitoring, args)
