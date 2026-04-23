# Custom LSTM Cell Sentiment Demo

This is deep understanding level project for understanding how an LSTM works from the inside.

I have built a custom LSTM Cell in this for better understanding.
Instead of using PyTorch's built-in LSTM layer, it builds a custom LSTM cell manually. The model is then trained on a tiny sentiment dataset using 50-dimensional GloVe word embeddings. After training, it saves logs and plots so you can inspect what the forget, input, candidate, and output gates are doing.

The goal is not to build a production sentiment classifier. The goal is to make the LSTM gates easier to see, debug, and understand.

## What This Project Does

- Loads GloVe word vectors.
- Converts simple movie-review sentences into embedding tensors.
- Trains a custom LSTM-based sentiment classifier.
- Tracks loss, accuracy, gradients, gate activations, weight magnitudes, and weight updates.
- Tests a few example sentences after training.
- Saves a summary, detailed log, and plots for each run.

## Project Layout

```text
app/
  config.py       training settings and paths
  data.py         toy dataset, tokenization, and GloVe loading
  model.py        custom LSTM cell and sentiment model
  training.py     training loop and learning diagnostics
  testing.py      sentence inspection after training
  plotting.py     training and gate visualizations
  main.py         full demo pipeline

docs/
  project_structure.md
  training_notes.md

requirements.txt
```

Generated files are written to `runs/`. The downloaded GloVe file is stored under `data_store/`. Both folders are ignored by Git because they can become large or machine-specific.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running The Demo

Run this from the project root:

```bash
python3 app/main.py
```

On the first run, the project will download `glove.6B.50d.txt` through `kagglehub` and save it locally at:

```text
data_store/glove/glove.6B.50d.txt
```

After that, future runs reuse the local file.

## Output

Each run creates a timestamped folder like this:

```text
runs/run_YYYYMMDD_HHMMSS/
```

Inside that folder you will find:

- A short run summary.
- A detailed training and testing log.
- Plots for training behavior.
- Gate, cell-state, and hidden-state plots for test sentences.

The console also prints the active device, run folder, and phase progress.

## Notes

This project uses a very small hand-written dataset, so the results are mostly useful for learning and visualization. If you want a stronger sentiment model, the dataset would need to be much larger and split into proper train/validation/test sets.

For more detail about the LSTM equations and what the gates mean, see `docs/training_notes.md`.

Project By :  [Satyam Shah](https://github.com/satyam-shah-gt)