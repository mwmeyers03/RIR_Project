# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
Physics-Informed Room Impulse Response (RIR) Generation Framework — a deep-learning research project. Two source files: `RIR_Project.ipynb` (main notebook, 28 cells) and `dataset.py` (standalone data pipeline module).

### Dependencies
All Python dependencies are installed via pip at the user level:
```
pip install datasets huggingface_hub torch pandas numpy scipy matplotlib jupyter ipykernel soundfile torchcodec flake8
```
`libsndfile1` is required as a system package for audio decoding.

### Running the code

- **Dataset unit test:** `python dataset.py` — downloads HuggingFace `mandipgoswami/rirmega` dataset (~4000 RIRs) on first run (cached afterwards at `~/.cache/huggingface/`), then runs assertions on single-sample and batch shapes.
- **Notebook:** `jupyter nbconvert --to notebook --execute RIR_Project.ipynb` to execute all cells headless, or launch `jupyter lab` for interactive use.
- **Lint:** `flake8 dataset.py --max-line-length=120` — pre-existing style warnings (E127, E272, F401, F541) are in the original code.

### Gotchas
- The `datasets` library (v4.6+) requires `torchcodec` for audio encoding/decoding. Without it, `load_dataset` fails with `ImportError: To support encoding audio data, please install 'torchcodec'`.
- The notebook's install cell uses `!pip install -q datasets huggingface_hub torch pandas numpy scipy` — this is safe to re-run but doesn't cover `torchcodec`, `soundfile`, or `matplotlib` which are also needed.
- Cell 10 (training cell) raises a `ValueError` about BatchNorm with batch_size=1. This is a pre-existing code issue, not an environment problem.
- Dataset download is cached after first run; subsequent `load_dataset` calls use the local cache and are fast.
- No GPU is required; code auto-detects CUDA and falls back to CPU.
