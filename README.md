# When Does Hierarchical Abstraction in World Models Actually Learn Anything?

Code, data and result files for the paper
**"When Does Hierarchical Abstraction in World Models Actually Learn Anything? A Controlled Diagnostic Study"**
(Regina Nam). PDF: [`paper/arxiv_paper.pdf`](paper/arxiv_paper.pdf).

The paper introduces a **random-abstractor control** for hierarchical world models: compare a trained
top level against an identically-shaped, untrained one. If the two score the same on a probe, the top
level has preserved information from the level below, not learned it.

## Reproduce everything

```bash
pip install -r requirements.txt
./reproduce.sh          # Linux / macOS
reproduce.bat           # Windows
```

This regenerates every file in `results/`, then `make_tables.py` turns them into
`paper/generated/*.tex`, which the paper includes directly — so the numbers in the PDF are produced
from the committed result files, not typed by hand. CPU only; the full run takes several hours.
All scripts fix their random seeds; finished runs are skipped on restart.

## Where every number in the paper comes from

| Paper | Number(s) | Result file | Script |
|---|---|---|---|
| Sec. 3.1 | room transitions per episode, share of episodes without a transition | [`results/env_stats.json`](results/env_stats.json) | `env_stats.py` |
| Table 1, Abstract, Sec. 4.1 | z1 / trained / random probe accuracy, gaps, seed range, level-use ablation, all four conditions | [`results/table1_2x2.json`](results/table1_2x2.json) | `run_2x2_sweep.py` |
| Table 2 | position probe R², memoryless vs. recurrent predictor, egocentric and full observability | [`results/table2_memory_egocentric.json`](results/table2_memory_egocentric.json), [`results/table2_memory_base.json`](results/table2_memory_base.json) | `compare_memory.py` |
| Table 3, Sec. 4.3 | random-control gap by landmark count (mean, std, per seed) | [`results/table3_landmarks.json`](results/table3_landmarks.json) | `run_landmark_sweep.py` |
| all of the above | the exact values printed in the paper, as LaTeX macros and JSON | [`paper/generated/numbers.tex`](paper/generated/numbers.tex), [`results/paper_numbers.json`](results/paper_numbers.json) | `make_tables.py` |

Architectural constants quoted in the paper (64×64 observations, 3×3 rooms, 128-d fast latent, 32-d
abstract latent, 128×128 open world, 4×4 probing zones, k-step level-2 predictor) are defined in
`env_building.py`, `env_openworld.py`, `models.py`, `models_hier.py` and `train_hier.py`.

## Files

| File | Purpose |
|---|---|
| `env_building.py` | 3×3 rooms; `BuildingEnv` (full view) and `EgocentricBuildingEnv` (agent-centred crop) |
| `env_openworld.py` | open 128×128 world with obstacles; fixed, reshuffled, and reshuffled + persistent landmarks |
| `env.py`, `env_variants.py` | two-room environment and its egocentric variant (memory ablation, Table 2) |
| `models.py` | convolutional encoder |
| `models_hier.py` | recurrent (`Abstractor`) and instantaneous (`InstantAbstractor`) top levels, level-1/level-2 predictors |
| `models_recurrent.py` | GRU vs. memoryless predictors with the same interface |
| `losses.py` | VICReg |
| `train_hier.py` | training of the two-level model and all diagnostics for a single run |
| `run_2x2_sweep.py` | Table 1 |
| `compare_memory.py` | Table 2 |
| `run_landmark_sweep.py` | Table 3 |
| `env_stats.py` | environment statistic of Sec. 3.1 |
| `make_tables.py` | result files → numbers and tables in the paper |

Some code comments are in Russian; all outputs, result files and this README are in English.

## License

MIT, see [`LICENSE`](LICENSE).
