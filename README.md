# Curiosity-driven exploration (intrinsic motivation) in PyTorch

An implementation of curiosity as an intrinsic reward, in the style of the Intrinsic Curiosity Module, written from scratch in PyTorch for a grid world.

- **Inverse dynamics model** (classification): given a state and the next state, predict the action taken (one-hot actions, softmax output).
- **Forward dynamics model** (regression): given a state and an action, predict the next state.
- **Intrinsic reward** = η × the forward model's prediction error, so the agent is drawn to states it can't yet predict.
- Experiments compare agents trained **with and without** the intrinsic reward. Trained weights for both are saved as `.pth` files.

The notebook (`intrinsic_rl.ipynb`) keeps my working notes between cells: design decisions, things that didn't work, and why.

Related: [Intrinsically-Motivated-Agents](https://github.com/mahen2-cmd/Intrinsically-Motivated-Agents), a two-agent version with no task reward at all.

Stack: Python, PyTorch, NumPy, pandas.
