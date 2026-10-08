"""Table 1 of the paper: the 2x2 design (observability x abstractor type),
trained top level vs. random-abstractor control, over several seeds.

Conditions:
  full + instantaneous   full + recurrent
  ego  + instantaneous   ego  + recurrent

For every (condition, seed) we record:
  z2_trained  - linear-probe accuracy of room identity from the trained top level
  z2_random   - same probe on an identically-shaped UNTRAINED abstractor
  gap_pp      - difference in percentage points
  z1_acc      - same probe on the fast level z1 (is the slow variable already there?)
  level_use_gap - increase of level-1 prediction error when z2 is shuffled (training diagnostic)

Results are written after every run to results/table1_2x2.json; finished runs
are skipped on restart, so the sweep can be interrupted and resumed.

Usage:
  python run_2x2_sweep.py --seeds 0 1 2 --episodes 200 --ep-len 48 --epochs 40 --k 8
"""
import argparse
import json
import os
import numpy as np
import torch

from train_hier import collect, encode_all, room_accuracy, train
from models_hier import make_abstractor
from env_building import GRID

CONDITIONS = [("full", "instantaneous"), ("full", "recurrent"),
              ("ego", "instantaneous"), ("ego", "recurrent")]


def run_one(env_name, kind, seed, episodes, ep_len, epochs, k, device):
    obs, acts, local_pos, rooms = collect(episodes, ep_len, seed=seed, env_name=env_name)
    enc, abst, p1, p2, gap = train(obs, acts, epochs, k, device, flat=False, seed=seed,
                                   abstractor=kind)
    enc.eval(); abst.eval()
    Z1, Z2 = encode_all(enc, abst, obs, device)
    rnd_abst = make_abstractor(kind, 128, 32).to(device).eval()
    _, Z2rnd = encode_all(enc, rnd_abst, obs, device)
    RID = torch.from_numpy(rooms.reshape(-1)).long()
    n_cls = GRID * GRID
    z1 = room_accuracy(Z1.reshape(-1, Z1.shape[-1]), RID, device, n_cls)
    z2 = room_accuracy(Z2.reshape(-1, Z2.shape[-1]), RID, device, n_cls)
    zr = room_accuracy(Z2rnd.reshape(-1, Z2rnd.shape[-1]), RID, device, n_cls)
    return {"seed": seed, "z1_acc": round(float(z1), 4), "z2_trained": round(float(z2), 4),
            "z2_random": round(float(zr), 4), "gap_pp": round(100 * (z2 - zr), 2),
            "level_use_gap": round(float(gap), 4)}


def summarize(rows):
    out = {"rows": rows}
    for f in ["z1_acc", "z2_trained", "z2_random", "gap_pp", "level_use_gap"]:
        v = np.array([r[f] for r in rows], float)
        out[f + "_mean"] = round(float(v.mean()), 4)
        out[f + "_std"] = round(float(v.std()), 4)     # population std over seeds, as in the paper
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--ep-len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--out", type=str, default="results/table1_2x2.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = {"config": vars(args), "conditions": {}}
    if os.path.exists(args.out):
        with open(args.out) as f:
            res = json.load(f)
    print(f"device: {device} | seeds: {args.seeds}")

    for env_name, kind in CONDITIONS:
        key = f"{env_name}_{kind}"
        cond = res["conditions"].setdefault(key, {"rows": []})
        done = {r["seed"] for r in cond["rows"]}
        for s in args.seeds:
            if s in done:
                print(f"[{key} seed {s}] already done, skipping"); continue
            print(f"[{key} seed {s}] training...", flush=True)
            r = run_one(env_name, kind, s, args.episodes, args.ep_len, args.epochs, args.k, device)
            cond["rows"].append(r)
            res["conditions"][key] = summarize(sorted(cond["rows"], key=lambda x: x["seed"]))
            cond = res["conditions"][key]
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"[{key} seed {s}] z1={r['z1_acc']:.3f} trained={r['z2_trained']:.3f} "
                  f"random={r['z2_random']:.3f} gap={r['gap_pp']:+.2f}pp "
                  f"level_use={r['level_use_gap']:.4f}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'condition':<22}{'z1':>8}{'trained':>16}{'random':>16}{'gap (pp)':>16}")
    for key, c in res["conditions"].items():
        print(f"{key:<22}{c['z1_acc_mean']:>8.3f}{c['z2_trained_mean']:>9.3f}±{c['z2_trained_std']:<6.3f}"
              f"{c['z2_random_mean']:>9.3f}±{c['z2_random_std']:<6.3f}"
              f"{c['gap_pp_mean']:>+9.2f}±{c['gap_pp_std']:<6.2f}")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
