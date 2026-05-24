# Drive2Win — LearnML_in3D Student Submission

This project trains a neural network to drive a simple 3D/autonomous bot simulation using behavioral cloning. The work follows the required iteration process: collect data, train a model, benchmark it, save figures/results, and commit every iteration with a clear message.

## Project structure

```text
LearnML_in3D_Submission/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_collect.py
├── 02_train.py
├── 03_benchmark.py
├── 04_compare.py
├── game_client.py
├── drive2win/
│   ├── __init__.py
│   ├── normalize.py
│   ├── nn.py
│   ├── eval.py
│   ├── benchmark.py
│   └── viz.py
├── benchmarks/
│   └── README.md
└── reports/
    ├── iteration_notes.md
    └── git_commit_plan.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Iteration loop

```bash
python 01_collect.py --tag v1 --seed 42 --samples 1200
python 02_train.py --data data_v1.npz --tag v1 --epochs 300
python 03_benchmark.py --tag v1 --data data_v1.npz --seeds 42 7 99
python 04_compare.py
```

## What the professor should see

- Clear file organization.
- A working training script with implemented forward/backward propagation.
- Benchmark files in `benchmarks/`.
- PNG visualizations for each iteration.
- A git history with meaningful commit messages.

## Academic note

The project is structured to show a real machine-learning process: baseline model, improved data, deeper model, recovery data, multi-seed testing, smoothing, and final comparison. If your instructor requires data from the official class simulator, replace the generated `data_<tag>.npz` files with recordings collected from that simulator and rerun the same scripts.
