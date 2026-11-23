# PASS: Progress-Aware Step-Size Scheduling for Transfer-Based Adversarial Security Evaluation

Core algorithm implementation of **PASS** (Progress-aware Adaptive Step-size Strategy) for zero-query transferable adversarial attacks.

## Repository Structure

```text
PASS_Code/
|-- pass_attack/
|   |-- attack.py      # Core PASS scheduler (PASSScheduler) and transfer attack backbones
|   |-- models.py      # Standard and diverse ImageNet model wrappers and feature hooks
|   `-- data.py        # NIPS 2017 ImageNet-compatible dataset loader
|-- run_attack.py      # Minimal CLI entry point for running an attack
|-- config.json        # Supported models and core attack configurations
`-- requirements.txt   # Core Python dependencies
```

## Setup

Python 3.10 or later and a CUDA-capable GPU are recommended. Install a PyTorch build matching your local CUDA runtime, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Pretrained model weights are downloaded automatically by `torchvision` and `timm` at runtime and are not included in this repository.

## Data

This repository provides the core algorithm only and does not include dataset files. To run `run_attack.py`, place the 1,000-image NIPS 2017 ImageNet-compatible benchmark under `data/`:

```text
data/
|-- images.csv
`-- images/
    `-- *.png
```

## Usage

Run commands from the repository root. Default PASS hyperparameters match the paper: `steps = 300`, `epsilon = 16/255`, `alpha = 2/255`, `eta = 0.75`, `tau = 0.01`, and non-uniform checkpoint schedule `c2`.

Single-source `DTNI-SU-PASS` (targeted, CE):

```bash
python run_attack.py --attack dtni-su-pass --mode targeted --loss ce
```

Single-source `DTMI-PASS` (untargeted, Logit):

```bash
python run_attack.py --attack dtmi-pass --mode untargeted --loss logit
```

Evaluate on diverse target architectures:

```bash
python run_attack.py --source-models resnet50 --target-group diverse --attack dtni-su-pass --mode targeted --loss ce
```

Ensemble-source attack:

```bash
python run_attack.py --source-models resnet50 densenet121 vgg16 --target-models inception_v3 --attack dtni-su-pass --mode targeted --loss logit
```

Checkpoint schedule and rollback options (`c1`, `c2`, `c3`, and `--no-rollback`):

```bash
python run_attack.py --attack dtni-su-pass --mode targeted --loss logit --checkpoint-schedule c1
python run_attack.py --attack dtni-su-pass --mode targeted --loss logit --checkpoint-schedule c3
python run_attack.py --attack dtni-su-pass --mode targeted --loss logit --no-rollback
```
