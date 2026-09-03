## Robust Tagging Challenge

Train a model whose anomaly-detection performance stays robust as more of the detector drops out. See `hackathon_playground.ipynb` for the full challenge writeup, scoring details, and what you're expected to edit.

Built by Arianna Cox and Ellison Scheuller.

Special thanks to Roy Cruz Candelaria, Maciej Glowacki, and Mehrnoosh Moallemi for the contrastive model baseline used in this challenge.

### Setup in JupyterHub

1. Launch a server in JupyterHub (with the GPU attached) and open a terminal.
2. Clone this repo and `cd` into it:
   ```bash
   git clone https://github.com/ellisonscheuller/robust_tagging_fastml_hackathon.git robust-tagging
   cd robust-tagging
   ```
3. Install the project (installs `embedding` in editable mode plus its dependencies):
   ```bash
   pip install -e .
   ```
4. Open `hackathon_playground.ipynb` in JupyterLab and run the cells top to bottom.
   - The notebook must be run from the root of your repo copy (where this README lives) — that's the default JupyterLab working directory when you open a notebook there.
   - Training/eval data is read directly from the shared PVC at `/hackathon-data/C9_robust_tagging/{train,eval}` — you don't need to download or convert anything.
   - Your own checkpoints and plots are written locally into `./checkpoints` and `./eval_plots` inside your repo copy, so they never collide with other participants sharing the same PVC.

### What you'll edit

- **`src/embedding/degradation.py`** — `Degradation.forward()` is a stub. This simulates detector dropout: `train.py` calls it with `severity=None` (implement your own randomized augmentation), `eval.py` calls it with a fixed `severity` for each step of the AUC-vs-severity sweep. Keep the class signature so both call sites keep working.
- **`src/embedding/models.py`** — the baseline architecture (`TransformerEncoder`, `Projector`, ...). Change layers, swap the encoder, add heads — anything, as long as it still produces a latent embedding.
- **`configs/train_config.yaml`** — hyperparameters (lr, embed size, loss weights, etc.).

### Scoring

At eval time the model sees both background and signal events and outputs a per-event anomaly score, scored via AUC vs. degradation severity. A robust model keeps a high AUC as more of the detector goes dark; your score is the area under that curve. `eval.py` overlays a red "(No degradation)" reference curve against your model's ("Your solution") on the same plot.

`eval.py` here uses **your own** `degradation.py` to simulate severity locally — it's for testing your own approach, not the official scoring run. For judging, we'll degrade the eval set ourselves with a method we're not disclosing in advance (conceptually it kills off geometric η–φ regions, similar to real detector dead zones), so solutions aren't tuned to the exact grading procedure.
