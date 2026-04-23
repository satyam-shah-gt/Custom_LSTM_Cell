# Project Structure

The code is focused on learning the working of LSTM by creating custom LSTM cell and viewing the indept of LSTM training and decision making, working of different LSTM gates.

## Main folder

- `app/config.py`
  Stores device and training hyperparameters.
- `app/data.py`
  Keeps the toy sentiment dataset, tokenization, GloVe loading, and sentence-to-tensor conversion.
- `app/model.py`
  Contains the manual `CustomLSTMCell` and the sentiment classifier built on top of it.
- `app/training.py`
  Contains the training loop, gradient tracking, gate activation tracking, and weight update tracking.
- `app/testing.py`
  Contains sentence inspection and token-level gate interpretation after training.
- `app/plotting.py`
  Contains all plotting so visualization is separated from model code.
- `app/main.py`
  Runs the full demo in order.

## Entry point

- `app/main.py`
  Simple runner that imports and calls `app.main`.

## Documentation

- `docs/project_structure.md`
  Explains how the project is split.
- `docs/training_notes.md`
  Explains how the custom LSTM cell trains and what to watch during learning.
