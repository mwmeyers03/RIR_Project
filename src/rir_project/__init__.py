"""Top-level package for the Physics-Informed RIR project.

This package mirrors the six-phase architecture described in ARCHITECTURE.md but
split into pure-Python modules so the notebook can remain thin.  Designed for
use in Google Colab and local dev environments alike.

Usage from a notebook:

```python
# install editable package when running on Colab
!pip install -q -e .

from rir_project.data import get_dataloader, RIRMegaDataset
from rir_project.models import MultibandEDCPredictor, DifferentiableFDN
from rir_project.loss import PhysicsInformedRIRLoss
from rir_project.trainer import TrainingConfig, RIRTrainer
from rir_project.synthesis import RIRSynthesiser
```

"""

__version__ = "0.1.0"

# expose convenient shortcuts if needed
from .data import (
	DEFAULT_MAX_RIR_LEN,
	DEVICE,
	INPUT_DIM,
	METRICS_DIM,
	MODAL_FEAT_DIM,
	OCTAVE_BANDS,
	CachedRIRDataset,
	RIRMegaDataset,
	compute_edc,
	compute_multiband_edc,
	compute_room_modes,
	downsample_edc_tensor,
	get_dataloader,
	rir_collate_fn,
)
from .models import (
	ConvBlock1D,
	DecoderBlock,
	DifferentiableFDN,
	EarlyReflectionNet,
	EncoderBlock,
	MultiHeadAttentionBottleneck,
	MultibandEDCPredictor,
	SirenLayer,
	SinusoidalPosEncoding,
	UNetRefiner,
)
from .loss import (
	EDCReconstructionLoss,
	MultiResolutionSTFTLoss,
	PhysicsInformedRIRLoss,
	continuity_residual,
	momentum_residual,
)
from .trainer import TrainingConfig, RIRTrainer
from .synthesis import (
	ConditionedFDN,
	EDCToFDNMapper,
	EarlyReflections,
	RIRSynthesiser,
	SignStickyPhaseReconstructor,
)
from .utils import (
	backup_notebook,
	compute_drr,
	demo_inference,
	edc_rmse_db,
	evaluate_on_test_set,
	estimate_rt60,
	generate_rir_from_params,
	load_synthesiser,
	log_spectral_distance,
	plot_edc_with_rt60,
	plot_multiband_edc,
	plot_per_band_rt60,
	plot_results_table,
	plot_rir_waveform,
	plot_spectrogram_comparison,
	plot_training_curves,
	save_checkpoint,
	save_figure,
	save_history,
	save_metrics,
	save_rir_audio,
	seed,
	set_seed,
	visualise_demo,
)

