# Training Notes

## What happens at each token

For each word embedding `x_t`, the custom cell concatenates the previous hidden state `h_prev` with the current word vector:

`combined = [h_prev ; x_t]`

That combined vector is sent through four learned linear transforms:

- Forget gate: `f_t = sigmoid(Wf * combined + bf)`
- Input gate: `i_t = sigmoid(Wi * combined + bi)`
- Candidate memory: `g_t = tanh(Wg * combined + bg)`
- Output gate: `o_t = sigmoid(Wo * combined + bo)`

Then the memory and hidden state are updated:

- Cell state: `c_t = f_t * c_prev + i_t * g_t`
- Hidden state: `h_t = o_t * tanh(c_t)`

## What each gate means

- Forget gate:
  Decides how much old memory should stay.
- Input gate:
  Decides how much new information should enter memory.
- Candidate memory:
  Proposes new content that could be written into memory.
- Output gate:
  Decides how much of the current memory becomes visible as the hidden state.

## How training changes the weights

For one sentence:

1. The model runs token by token and produces a final logit.
2. `BCEWithLogitsLoss` compares that logit with the target label.
3. `loss.backward()` computes gradients for all gate weights and biases.
4. `optimizer.step()` updates weights using those gradients.

In this project the training code now tracks:

- Gradient magnitude per gate
- Weight magnitude per gate
- Average weight change after each optimizer step
- Average gate activation through the epoch

## How to read the training output

- Large gradient magnitude:
  That gate is receiving a stronger correction signal.
- Large average weight change:
  That gate's parameters are moving more during optimization.
- Forget gate near `1`:
  Old memory is being kept.
- Forget gate near `0`:
  Old memory is being dropped.
- Input gate near `1`:
  New information is being written strongly.
- Output gate near `1`:
  The hidden state exposes more of the cell memory.

## Learning helpers added

- A single-step walkthrough before full training
  This shows one forward pass, one backward pass, and one optimizer update on a sample sentence.
- Epoch-level training history
  This is used by the plotting module so you can see how learning evolves across epochs.
