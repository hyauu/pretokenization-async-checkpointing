# S09 Experiment Code

Python code for single-GPU fine-tuning experiments on pretokenized inputs and asynchronous checkpointing.

## Source layout

- `replication/host/` coordinates runs on a Windows host with WSL2.
- `replication/src/` contains the guest training, checkpointing, placement, and timing code.
- `replication/setup/` prepares and checks the experiment environment.
- `analysis/rebuild_replication.py` recomputes statistics, tables, and figures from completed run records.
- `artifact/analysis/` contains the analysis and validation scripts for the preceding experiments.

## Running the code

The replication scripts use PyTorch, TorchTune, and a GPU-enabled WSL2 environment. Review the paths in `replication/host/runtime.py`, the setup scripts, and `replication/protocol.json` for your machine. Provide the model, datasets, and starting checkpoints in the expected locations. The checked-in execution window records the original run. Set up a fresh window for a new run.

For CPU analysis, install SciPy and Matplotlib. Put `replication_runs.csv`, `historical_main_runs.csv`, and `historical_noise_reference.json` under `data/`, and put `replication_review.json` under `evidence/`. Then run:

~~~sh
mkdir -p tables figures
python analysis/rebuild_replication.py
~~~

The historical scripts use experiment inputs under `artifact/`, as specified by their relative paths.
