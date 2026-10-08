"""
Writes results/env_stats.json.
Usage:  python env_stats.py --episodes 500 --ep-len 75
"""
import argparse
import json
import os
import numpy as np

from env_building import BuildingEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--ep-len", type=int, default=75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/env_stats.json")
    args = p.parse_args()
    env = BuildingEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    trans = []
    for _ in range(args.episodes):
        env.reset()
        prev = env.room_id
        a = rng.uniform(-1, 1, 2)
        n = 0
        for _ in range(args.ep_len):
            a = np.clip(0.8 * a + 0.5 * rng.normal(size=2), -1, 1)
            env.step(a)
            if env.room_id != prev:
                n += 1
                prev = env.room_id
        trans.append(n)
    trans = np.array(trans)
    res = {"config": vars(args),
           "room_transitions_per_episode_mean": round(float(trans.mean()), 3),
           "room_transitions_per_episode_std": round(float(trans.std()), 3),
           "share_episodes_without_transition": round(float((trans == 0).mean()), 3)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(res)


if __name__ == "__main__":
    main()
