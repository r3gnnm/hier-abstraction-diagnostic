"""
  python compare_memory.py --env egocentric --episodes 150 --epochs 20
  python compare_memory.py --env base --episodes 150 --epochs 20
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from env_variants import VARIANTS
from models import Encoder
from models_recurrent import RecurrentPredictor, MLPPredictorSeq
from losses import vicreg_loss


def collect_sequences(env_cls, n_episodes, ep_len, seed=0):
    env = env_cls(seed=seed)
    rng = np.random.default_rng(seed)
    O, A, S = [], [], []
    for _ in range(n_episodes):
        o = env.reset()
        obs_ep, act_ep, st_ep = [o], [], [env.state.copy()]
        a = rng.uniform(-1, 1, 2)
        for _ in range(ep_len):
            a = np.clip(0.7 * a + 0.5 * rng.normal(size=2), -1, 1)
            o = env.step(a)
            obs_ep.append(o); act_ep.append(a.astype(np.float32))
            st_ep.append(env.state.copy())
        O.append(np.stack(obs_ep)); A.append(np.stack(act_ep)); S.append(np.stack(st_ep))
    return np.stack(O), np.stack(A), np.stack(S)


def train(pred_cls, obs, acts, epochs, latent_dim, device, bs=32, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    enc = Encoder(latent_dim).to(device)
    pred = pred_cls(latent_dim).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=lr, weight_decay=1e-5)
    obs_t = torch.from_numpy(obs)
    acts_t = torch.from_numpy(acts)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_ep)
        tot_gap = tot_loss = n_batches = 0
        for i in range(0, n_ep, bs):
            idx = perm[i:i + bs]
            o = obs_t[idx].to(device)            # (B, T+1, 1, 64, 64)
            a = acts_t[idx].to(device)           # (B, T, 2)
            B = o.shape[0]
            z_all = enc(o.view(B * (T + 1), *o.shape[2:])).view(B, T + 1, -1)
            h = pred.init_hidden(B, device)
            loss = 0.0
            gap_num = gap_den = 0.0
            for t in range(T):
                z_hat, h = pred(z_all[:, t], a[:, t], h)
                l, _ = vicreg_loss(z_hat, z_all[:, t + 1])
                loss = loss + l
                with torch.no_grad():
                    z_rand, _ = pred(z_all[:, t], a[torch.randperm(B), t], h)
                    e_rand = F.mse_loss(z_rand, z_all[:, t + 1]).item()
                    e_true = F.mse_loss(z_hat, z_all[:, t + 1]).item()
                    gap_num += e_rand - e_true; gap_den += e_rand
            loss = loss / T
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 5.0)
            opt.step()
            tot_loss += loss.item()
            tot_gap += gap_num / max(gap_den, 1e-8)   # НОРМИРОВАННЫЙ gap
            n_batches += 1
        if ep % 5 == 0 or ep == epochs:
            print(f"    epoch {ep:3d} | loss {tot_loss/n_batches:.4f} "
                  f"| action_gap_norm {tot_gap/n_batches:.4f}", flush=True)
    return enc, pred, tot_gap / n_batches


@torch.no_grad()
def probe(enc, pred, obs, acts, states, device, use_memory: bool):
    n_ep, T1 = obs.shape[0], obs.shape[1]
    T = T1 - 1
    feats, targets = [], []
    for i in range(0, n_ep, 16):
        o = torch.from_numpy(obs[i:i + 16]).to(device)
        a = torch.from_numpy(acts[i:i + 16]).to(device)
        B = o.shape[0]
        z_all = enc(o.view(B * (T + 1), *o.shape[2:])).view(B, T + 1, -1)
        h = pred.init_hidden(B, device)
        for t in range(T):
            f = torch.cat([z_all[:, t], h], -1) if use_memory else z_all[:, t]
            feats.append(f.cpu())
            targets.append(torch.from_numpy(states[i:i + 16, t]))
            _, h = pred(z_all[:, t], a[:, t], h)
    X = torch.cat(feats).to(device)
    Y = torch.cat(targets).to(device)
    n_tr = int(0.8 * len(X))
    Xtr = torch.cat([X[:n_tr], torch.ones(n_tr, 1, device=device)], 1)
    Xte = torch.cat([X[n_tr:], torch.ones(len(X) - n_tr, 1, device=device)], 1)
    reg = 1e-3 * torch.eye(Xtr.shape[1], device=device)
    w = torch.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Y[:n_tr])
    P = Xte @ w
    ss_res = ((Y[n_tr:] - P) ** 2).sum(0)
    ss_tot = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).cpu().numpy()
    per_point = ((Y[n_tr:] - P) ** 2).sum(-1).sqrt().cpu().numpy()
    return r2, Y[n_tr:].cpu().numpy(), per_point


def probe_error_map(states, errors, out, title, bins=14):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.zeros((bins, bins)); cnt = np.zeros((bins, bins))
    for (x, y), e in zip(states, errors):
        i, j = min(int(y * bins), bins - 1), min(int(x * bins), bins - 1)
        grid[i, j] += e; cnt[i, j] += 1
    grid = np.where(cnt > 0, grid / np.maximum(cnt, 1), np.nan)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(grid, cmap="inferno", origin="upper", extent=[0, 64, 64, 0])
    fig.colorbar(im, label="ошибка probe (позиция)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"    сохранено: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="egocentric")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--ep-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=str, default=None,
                   help="default: results/table2_memory_<env>.json")
    p.add_argument("--maps", action="store_true", help="also save per-position error maps")
    args = p.parse_args()
    if args.out is None:
        args.out = f"results/table2_memory_{args.env}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"device: {device} | env: {args.env} | seeds: {args.seeds}")

    runs = []
    for seed in args.seeds:
        obs, acts, states = collect_sequences(VARIANTS[args.env], args.episodes,
                                              args.ep_len, seed=seed)
        row = {"seed": seed}
        for key, cls in [("mlp", MLPPredictorSeq), ("gru", RecurrentPredictor)]:
            print(f"\n--- seed {seed} | {key} ---")
            enc, pred, gap = train(cls, obs, acts, args.epochs, args.latent_dim, device, seed=seed)
            enc.eval(); pred.eval()
            r2_z, st, err_z = probe(enc, pred, obs, acts, states, device, use_memory=False)
            r2_zh, _, err_zh = probe(enc, pred, obs, acts, states, device, use_memory=True)
            row[key] = {"action_gap_norm": round(float(gap), 4),
                        "probe_r2_from_z": [round(float(v), 4) for v in r2_z],
                        "probe_r2_from_z_and_h": [round(float(v), 4) for v in r2_zh],
                        "probe_r2_from_z_mean": round(float(np.mean(r2_z)), 4),
                        "probe_r2_from_z_and_h_mean": round(float(np.mean(r2_zh)), 4)}
            print(f"    probe R2 from z: {np.mean(r2_z):.3f} | from (z,h): {np.mean(r2_zh):.3f}")
            if args.maps:
                probe_error_map(st, err_zh, f"results/probe_map_{args.env}_{key}_s{seed}.png",
                                f"{key}: where the model knows where it is")
        runs.append(row)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "runs": runs}, f, indent=2)

    # Table 2: memoryless = MLP predictor probed from z; with memory = GRU probed from (z, h)
    no_mem = np.array([r["mlp"]["probe_r2_from_z_mean"] for r in runs])
    mem = np.array([r["gru"]["probe_r2_from_z_and_h_mean"] for r in runs])
    summary = {"no_memory_mean": round(float(no_mem.mean()), 4), "no_memory_std": round(float(no_mem.std()), 4),
               "with_memory_mean": round(float(mem.mean()), 4), "with_memory_std": round(float(mem.std()), 4)}
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "runs": runs, "summary": summary}, f, indent=2)
    print(f"\nno memory: {summary['no_memory_mean']:.3f}±{summary['no_memory_std']:.3f} | "
          f"with memory: {summary['with_memory_mean']:.3f}±{summary['with_memory_std']:.3f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
