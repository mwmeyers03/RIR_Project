# Baseline Notebook

The file `RIR_Project_baseline.ipynb` is the pre-refactor version of the project.
Use it as a reference and to re-run the original training pipeline for comparison
metrics.

## How to use

1. Open the notebook in Colab or locally. Install dependencies as before with
   `!pip install -q datasets huggingface_hub torch pandas numpy scipy`.
2. Run the notebook from top to bottom with a fixed random seed, e.g.: 

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

3. Save the resulting history and example predictions to a JSON file:

```python
import json
with open('baseline_history.json','w') as f:
    json.dump(history, f)
# optionally export one predicted and reference RIR
```

4. Commit `baseline_history.json` and the notebook to the repository to lock in
the baseline.

The metrics and visualizations produced by the baseline notebook provide the
"point zero" against which the refactored, modular code can be judged.
