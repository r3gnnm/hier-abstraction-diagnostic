@echo off
rem Regenerates every result file and every number in the paper. CPU, several hours.
python env_stats.py --episodes 500 --ep-len 75 || exit /b 1
python run_2x2_sweep.py --seeds 0 1 2 --episodes 200 --ep-len 48 --epochs 40 --k 8 || exit /b 1
python compare_memory.py --env egocentric --episodes 150 --ep-len 32 --epochs 20 --seeds 0 1 2 || exit /b 1
python compare_memory.py --env base --episodes 150 --ep-len 32 --epochs 20 --seeds 0 1 2 || exit /b 1
python run_landmark_sweep.py --landmarks 0 3 6 10 15 --seeds 0 1 2 --episodes 200 --ep-len 48 --epochs 40 --k 8 || exit /b 1
python make_tables.py || exit /b 1
