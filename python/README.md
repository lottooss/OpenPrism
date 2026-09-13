# OpenPrism Python tooling

Offline training, export, evaluation and deterministic testing utilities. Python is not in the production control loop. See [the project README](../README.md) and [training guide](../docs/TRAINING.md).

Use `uv sync --frozen --project python` for base tooling, or add `--extra ml` for optional model training and export dependencies. The internal Python import namespace remains `aim` for contract compatibility.
