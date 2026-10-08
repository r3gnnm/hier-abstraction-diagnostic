"""Builds every empirical number in the paper from the committed result files.

Reads   results/table1_2x2.json, results/table2_memory_<env>.json,
        results/table3_landmarks.json, results/env_stats.json
Writes  paper/generated/numbers.tex   - one LaTeX macro per number used in the text
        paper/generated/table1.tex    - Table 1 body
        paper/generated/table2.tex    - Table 2 body
        paper/generated/table3.tex    - Table 3 body
        results/paper_numbers.json    - the same numbers, machine-readable

The paper \\input{}s these files, so the PDF cannot drift from the results.
Usage:  python make_tables.py
"""
import json
import os

R = "results"
G = os.path.join("paper", "generated")


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def main():
    os.makedirs(G, exist_ok=True)
    num = {}          # macro name -> formatted string
    raw = {}          # for results/paper_numbers.json

    def put(name, value, fmt):
        num[name] = format(value, fmt)
        raw[name] = value

    # ---------------- Table 1 ----------------
    t1 = load("table1_2x2.json")["conditions"]
    rows = [("full_instantaneous", "Full", "Instantaneous"),
            ("full_recurrent", "Full", "Recurrent"),
            ("ego_instantaneous", "Egocentric", "Instantaneous"),
            ("ego_recurrent", "Egocentric", "Recurrent")]
    body = []
    for key, obs, ab in rows:
        c = t1[key]
        tag = key.replace("_", "").replace("full", "Full").replace("ego", "Ego") \
                 .replace("instantaneous", "Inst").replace("recurrent", "Rec")
        put(f"T{tag}Trained", c["z2_trained_mean"], ".3f")
        put(f"T{tag}TrainedSd", c["z2_trained_std"], ".3f")
        put(f"T{tag}Random", c["z2_random_mean"], ".3f")
        put(f"T{tag}RandomSd", c["z2_random_std"], ".3f")
        put(f"T{tag}Gap", c["gap_pp_mean"], "+.2f")
        put(f"T{tag}GapSd", c["gap_pp_std"], ".2f")
        put(f"T{tag}Zone", c["z1_acc_mean"], ".3f")
        put(f"T{tag}LevelUse", c["level_use_gap_mean"], ".4f")
        gaps = [r["gap_pp"] for r in c["rows"]]
        put(f"T{tag}GapMin", min(gaps), ".1f")
        put(f"T{tag}GapMax", max(gaps), ".1f")
        put(f"T{tag}N", len(c["rows"]), "d")
        bold = key == "ego_recurrent"
        b = (lambda s: r"\mathbf{" + s + "}") if bold else (lambda s: s)
        body.append(f"{obs} & {ab} & ${num[f'T{tag}Zone']}$ & "
                    f"${b(num[f'T{tag}Trained'] + r'\pm' + num[f'T{tag}TrainedSd'])}$ & "
                    f"${b(num[f'T{tag}Random'] + r'\pm' + num[f'T{tag}RandomSd'])}$ & "
                    f"${b(num[f'T{tag}Gap'] + r'\pm' + num[f'T{tag}GapSd'])}$pp \\\\")
    # ratio of level-use between recurrent and instantaneous (egocentric)
    lu_ratio = t1["ego_recurrent"]["level_use_gap_mean"] / max(t1["ego_instantaneous"]["level_use_gap_mean"], 1e-9)
    put("TLevelUseRatio", lu_ratio, ".1f")
    with open(os.path.join(G, "table1.tex"), "w") as f:
        f.write("\n".join(body) + "\n")

    # ---------------- Table 2 ----------------
    body = []
    for env, label in [("egocentric", "Egocentric"), ("base", "Full observability (control)")]:
        s = load(f"table2_memory_{env}.json")["summary"]
        tag = "Ego" if env == "egocentric" else "Full"
        put(f"M{tag}NoMem", s["no_memory_mean"], ".3f")
        put(f"M{tag}NoMemSd", s["no_memory_std"], ".3f" if s["no_memory_std"] >= 0.0005 else ".4f")
        put(f"M{tag}Mem", s["with_memory_mean"], ".3f")
        put(f"M{tag}MemSd", s["with_memory_std"], ".3f" if s["with_memory_std"] >= 0.0005 else ".4f")
        body.append(f"{label} & ${num[f'M{tag}NoMem']}\\pm{num[f'M{tag}NoMemSd']}$ & "
                    f"${num[f'M{tag}Mem']}\\pm{num[f'M{tag}MemSd']}$ \\\\")
    with open(os.path.join(G, "table2.tex"), "w") as f:
        f.write("\n".join(body) + "\n")

    # ---------------- Table 3 ----------------
    t3 = load("table3_landmarks.json")
    summ = sorted(t3["summary"], key=lambda x: x["n_landmarks"])
    head = " & ".join(str(s["n_landmarks"]) for s in summ)
    means = " & ".join(format(s["gap_mean_pp"], ".2f") for s in summ)
    stds = " & ".join(format(s["gap_std_pp"], ".2f") for s in summ)
    with open(os.path.join(G, "table3.tex"), "w") as f:
        f.write(f"$n$ landmarks & {head} \\\\\n\\midrule\nGap, mean (pp) & {means} \\\\\n"
                f"Std (pp) & {stds} \\\\\n")
    for s in summ:
        n = s["n_landmarks"]
        word = {0: "Zero", 3: "Three", 6: "Six", 10: "Ten", 15: "Fifteen"}.get(n, f"N{n}")
        put(f"L{word}Mean", s["gap_mean_pp"], ".2f")
        put(f"L{word}Std", s["gap_std_pp"], ".2f")
    # seed-0 single runs at n = 0, 3, 6 (the "single-run trend" of Sec. 4.3)
    seed0 = {r["n_landmarks"]: r["gap_pp"] for r in t3["runs"] if r["seed"] == min(x["seed"] for x in t3["runs"])}
    for n, word in [(0, "Zero"), (3, "Three"), (6, "Six")]:
        if n in seed0:
            put(f"L{word}SeedZero", seed0[n], ".1f")
    raw["landmark_seed0_monotonic_0_3_6"] = bool(seed0.get(0, 0) < seed0.get(3, 0) < seed0.get(6, 0))

    # ---------------- environment statistics ----------------
    e = load("env_stats.json")
    put("EnvTransitions", e["room_transitions_per_episode_mean"], ".1f")
    put("EnvEpLen", e["config"]["ep_len"], "d")
    put("EnvNoTransitionPct", 100 * e["share_episodes_without_transition"], ".0f")

    with open(os.path.join(G, "numbers.tex"), "w") as f:
        f.write("% generated by make_tables.py from results/*.json -- do not edit by hand\n")
        for k, v in num.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(R, "paper_numbers.json"), "w") as f:
        json.dump(raw, f, indent=2)
    print(f"{len(num)} numbers -> {G}/numbers.tex, tables -> {G}/table*.tex, results/paper_numbers.json")
    if not raw["landmark_seed0_monotonic_0_3_6"]:
        print("NOTE: seed-0 gaps at n=0,3,6 are NOT monotonic -- rewrite the 'single-run trend' "
              "sentence in Sec. 4.3 accordingly.")


if __name__ == "__main__":
    main()
