# run_model_based.py
from model_dirichlet import QACDirichletAgent
from modelbased_env import RSVARBatchSampler
from monitoring_model_based import Monitoring

import itertools
import os, json, logging, random, warnings, datetime, math
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
matplotlib.use('module://ipykernel.pylab.backend_inline')

LOG = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)

EPS = 1e-12

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ('yes','true','t','y','1'): return True
    if v in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--render_each", default=100, type=int)
parser.add_argument("--evaluate", default=True, type=str2bool)
parser.add_argument("--seed", default=None, type=int)

parser.add_argument("--episodes", default=30, type=int)
parser.add_argument("--gamma", default=0.96, type=float) # Same as in the expected case but different than model free
parser.add_argument("--learning_tau", type=float, default=0.2)
parser.add_argument('--tau_levels', type=int, default=10)

parser.add_argument('--critic_lr_start', type=float, default=0.01)
parser.add_argument('--critic_lr_end', type=float, default=0.001)
parser.add_argument('--actor_lr_start', type=float, default=0.001)
parser.add_argument('--actor_lr_end', type=float, default=0.0001)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--entropy_reg', type=float, default=0.01)

parser.add_argument('--transaction_cost', type=float, default=1e-3) # 1e-5 small
parser.add_argument('--initial_wealth', type=float, default=1.0)
parser.add_argument('--r_f', type=float, default=1.001)
parser.add_argument('--gamma_crra', type=float, default=3.0) # Quantile function is invariant to any monotonic transformation, but lets align with Expected case
parser.add_argument('--n_scen', type=int, default=64)

# minibatch
parser.add_argument('--batch_size', type=int, default=512, help="Minibatch size for training; <=0 uses full batch")

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
    choices=['advantage', 'is_negative', 'weighted_quantile','original'])
parser.add_argument('--critic_type', type=str, nargs='?', default='standard',
    choices=['standard', 'monte_carlo_dropout', 'bayesian'])

# for later, we may need. Not sure that our weight grid is sufficient....
def make_weight_grid_simplex(N, levels=(0.0, 0.5, 1.0)):
    cand = np.array(list(itertools.product(*([levels]*N))), dtype=float)
    mask = cand.sum(axis=1) <= 1.0
    Wgrid = cand[mask]
    Wgrid = np.unique(np.round(Wgrid, 6), axis=0)
    return Wgrid

def make_logreturn_grid(axes_list):
    meshes = np.meshgrid(*axes_list, indexing="ij")
    M = meshes[0].size
    N = len(meshes)
    Rgrid = np.zeros((M, N), float)
    for n, g in enumerate(meshes):
        Rgrid[:, n] = g.reshape(-1)
    return Rgrid

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def train(env_model, agent,
          epochs=50, batches_per_epoch=200, batch_size=512,
          n_scen=16, rng=None, log_every=1, monitoring=None,
          eval_tag_prefix="eval"):

    if rng is None:
        rng = np.random.default_rng(0)

    N = int(env_model.N)
    K = int(env_model.K)

    hist = {
        "actor_loss": [], "critic_loss": [], "reward": [], "turnover": [], "fee": [],
        "eval_reward": [], "eval_tau": [], "eval_fee": [],
        "n_transitions": []
    }

    for ep in range(1, epochs + 1):
        transitions = env_model.make_batch(agent.actor_network, rng_eps=rng) # we have model based set up, we need to create transitions
        n_trans = len(transitions)
        LOG.info("[epoch %03d] built %d transitions", ep, n_trans)
        hist["n_transitions"].append(n_trans)


        # shuffle transitions: we used a structured loop to create
        # transitions, i think we need to shuffle for safety
        idx = rng.permutation(n_trans)
        transitions = [transitions[i] for i in idx]

        a_loss_acc = 0.0
        c_loss_acc = 0.0
        nb = 0

        # better pass..
        n_trans = len(transitions)
        bs = n_trans if batch_size <= 0 else batch_size


        for start in range(0, n_trans, bs):
            end = min(start + bs, n_trans)
            batch = transitions[start:end]
            if not batch:
                break
           
           
            a_l, c_l = agent.learn(batch)   # NOTE: agent.learn must accept q-weights
            a_loss_acc += float(a_l)
            c_loss_acc += float(c_l)
            nb += 1

        # Q-weighted average reward
        rs = np.array([t[3] for t in transitions], dtype=float)
        qs = np.array([t[4] for t in transitions], dtype=float)
        r_mean = float(np.sum(qs * rs) / max(np.sum(qs), 1e-12))

        # state structure (weights(2),returns(2),onehot(i.r 0,1,0)). One hot indicates which regime we are
        tcs = []
        for (s_obs, _s_next, a_full, _r, _q) in transitions:
            w_prev = np.asarray(s_obs[:N], dtype=float)                   # risky in state
            r_now  = np.asarray(s_obs[N:N+N], dtype=float)                # log-returns
            w_full = np.asarray(a_full, float).reshape(-1)                # (N+1), Risky weights + Risk free
            tau    = float(env_model.cost(w_target_full=w_full, w_prev_risky=w_prev, r_now=r_now)) / max(env_model.c, 1e-12) # without c the cost, turnover
            tcs.append(tau)
        tc_avg  = float(np.mean(tcs)) if tcs else 0.0
        fee_avg = float(env_model.c * tc_avg) # super primitie calculation but ok

        a_mean = (a_loss_acc / nb) if nb > 0 else 0.0 # average a mean, c mean over
        c_mean = (c_loss_acc / nb) if nb > 0 else 0.0

        hist["actor_loss"].append(a_mean)
        hist["critic_loss"].append(c_mean)
        hist["reward"].append(r_mean)
        hist["turnover"].append(tc_avg)
        hist["fee"].append(fee_avg)

        if monitoring is not None:
            monitoring.update_metric("Actor loss", a_mean)
            monitoring.update_metric("Critic loss", c_mean)
            monitoring.update_metric("Avg reward (Q-weighted)", r_mean)
            monitoring.update_metric("Avg turnover (no cost)", tc_avg)
            monitoring.update_metric("Avg fee (with cost)", fee_avg)

        if ep % log_every == 0:
            print(f"[epoch {ep:03d}] "
                  f"actor_loss={a_mean:.4f} | critic_loss={c_mean:.4f} | "
                  f"r_mean={r_mean:.6f} | tau={tc_avg:.6f} | fee={fee_avg:.6f}")
    
    # finally evaulate
    if monitoring is not None:

        summary = monitoring.evaluate_on_training_testing_data_stepwise(save=True, tag="train_end")
        
        hist["eval_reward"].append(summary.get("avg_reward", np.nan))
        hist["eval_tau"].append(summary.get("avg_tau", np.nan))
        hist["eval_fee"].append(summary.get("avg_fee", np.nan))

    # return last-epoch metrics + full history
    return a_mean, c_mean, r_mean, tc_avg, fee_avg, hist
def main(env_model, agent, monitoring, args):
    save_root = os.path.join(args.checkpoints_dir, args.save_as_file)
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    rng = np.random.default_rng(int(getattr(args, "seed", 0)))

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
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        n_scen=args.n_scen,
        rng=rng,
        log_every=max(1, getattr(args, "log_every", 1)),
        monitoring=monitoring,
        eval_tag_prefix="train_end"
    )

    if getattr(args, "save_weights", True):
        LOG.info("saving weights to %s", save_root)
        agent.save_weights(save_root + os.sep)
        if hasattr(monitoring, "save_brain_weights"):
            monitoring.save_brain_weights()

    if monitoring is not None:
        monitoring.save_metrics()


    with open(os.path.join(save_root, "last_run.txt"), "w") as f:
        f.write("=== Last Run ===\n")
        f.write(f"timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"episodes: {args.episodes}\n")
        f.write(f"learning_tau: {getattr(args, 'learning_tau', None)}\n")
        f.write(f"avg_actor_loss: {a_mean:.6f}\n")
        f.write(f"avg_critic_loss: {c_mean:.6f}\n")
        f.write(f"avg_reward_Qweighted: {r_mean:.8f}\n")
        f.write(f"avg_turnover: {tc_avg:.8f}\n")
        f.write(f"avg_fee: {fee_avg:.8f}\n")


    if getattr(args, "evaluate", False):
        try:
            import matplotlib.pyplot as plt
            plt.figure(); plt.plot(history["actor_loss"]); plt.title("Actor loss (avg/epoch)"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
            plt.figure(); plt.plot(history["critic_loss"]); plt.title("Critic loss (avg/epoch)"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
        except Exception as e:
            LOG.warning("Plotting failed: %s", e)

def _build_default_env_agent_monitor(args):
    # RS-VAR parameters
    '''
    '''
    Phi_k = np.tile(np.array([[0.15, 0.10],
                          [0.10, 0.15]], dtype=float), (3, 1, 1))
    ''' Const difff par
        const_k = np.array([
        [ 0.0060,  0.0050],  # Bull: lift both assets
        [ 0.0045,  0.0045],  # Neutral: make both clearly positive
        [-0.0010,  0.0045],  # Bear: keep asset 2 solidly positive
    ], dtype=float)
    '''
   
    
    '''
    #original base
    const_k = np.array([
        [ 0.0040,  0.0030],   # Bull
        [ 0.0030,  0.0028],   # Neutral
        [-0.0090,  0.0030],   # Bear
    ], dtype=float)
    
    '''
    # more cash incentive: good example, tau 9 put into cash. not in this set up, there is a code parameter problem
    const_k = np.array([
        [ 0.0015,  0.0010],   # Bull (down from 0.0020/0.0015)
        [ 0.0010,  0.0010],   # Neutral (down from 0.0015/0.0014)
        [-0.0120,  0.0010],   # Bear (more negative A1, smaller A2)
    ], dtype=float)

    
    # more cash incentive good example, tau 9 put into cash
    Sigma_k = np.array([
        [[0.00040,  0.00008],
        [0.00008,  0.00036]],         # Bull
        [[0.00140,  0.00000],
        [0.00000,  0.00110]],         # Neutral (slightly calmer than before)
        [[0.00650, -0.00400],
        [-0.00400, 0.00250]],         # Bear: fatter vol, stronger neg corr
    ], dtype=float)
   
    '''
   

    #original base
    Sigma_k = np.array([
        [[0.0005,  0.00010],
        [0.00010, 0.00045]],   # Bull
        [[0.0018,  0.00000],
        [0.00000, 0.00140]],   # Neutral
        [[0.0050, -0.00300],
        [-0.00300, 0.00200]],  # Bear
    ], dtype=float)
    '''


    Q_bull_bear = np.array([
    [0.74, 0.02, 0.24],
    [0.10, 0.82, 0.08],
    [0.30, 0.02, 0.68]],dtype=float)  

    Q_neutral_bear = np.array([
    [0.82, 0.08, 0.10],
    [0.02, 0.68, 0.30],
    [0.02, 0.24, 0.74]],dtype=float)   

    Q_bull_neutral = np.array([
    [0.74, 0.24, 0.02],
    [0.30, 0.68, 0.02],
    [0.10, 0.08, 0.82]],dtype=float)    

    K, N = const_k.shape

    # previous risky weight grid
    Wgrid = np.array([
        [0.00, 0.00],
        [1.00, 0.00],
        [0.00, 1.00],
        [0.50, 0.50],
        [0.20, 0.20],
        [0.35, 0.35],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.60, 0.20],
        [0.20, 0.60],
        [0.00, 0.10],
        [0.10, 0.00],
        [0.30, 0.00],
        [0.00, 0.30],
        [0.60, 0.00],
        [0.00, 0.60],
    ], dtype=float)

    # log-return grid
    axes_list = [np.round(np.linspace(-0.04, 0.04, 7, dtype=float), 3) for _ in range(N)]
    Rgrid = make_logreturn_grid(axes_list)

    env = RSVARBatchSampler(
        const_k=const_k, Phi_k=Phi_k, Sigma_k=Sigma_k, Q=Q_bull_bear,
        Wgrid=Wgrid, Rgrid=Rgrid,
        r_f=args.r_f, c=args.transaction_cost,
        gamma_crra=args.gamma_crra,
        n_scen=args.n_scen,
        seed=(args.seed if args.seed is not None else 0)
    )
    # state = [w_prev(N), r_now(N), onehot(K)]
    state_shape = (2 * N + K,)
    # IMPORTANT: Dirichlet actor outputs N+1 weights (last = cash)
    agent = QACDirichletAgent(state_shape=state_shape, stock_dimension=N + 1, args=args)
    monitoring = Monitoring(agent=agent, env=env, args=args)
    return env, agent, monitoring

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    set_global_determinism(seed=args.seed)

    env, agent, monitoring = _build_default_env_agent_monitor(args)
    main(env, agent, monitoring, args)

    # TODO:
    # - Verify DP consistency: reward, cost, state
