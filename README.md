# MorphoCLIP

**Cross-modal contrastive learning aligning molecular graphs with Cell Painting morphological profiles for zero-shot mechanism-of-action retrieval.**

## Overview

MorphoCLIP trains a CLIP-style dual encoder that maps molecular structures and Cell Painting CellProfiler feature vectors into a shared embedding space. Once trained, you can query the model with any SMILES string to retrieve phenotypically similar compounds — without any MoA label at inference time.

### Key novelty
- First CLIP-style alignment between molecular graph encodings and Cell Painting profiles
- Hard negative mining strategy using known off-target MoA pairs
- Zero-shot MoA retrieval benchmark on JUMP-CP × ChEMBL

## Installation
```bash
cd morphoclip
pip install -r requirements.txt
```

## Data
```bash
python scripts/download_data.py
python scripts/preprocess.py
```

## Training
```bash
python scripts/train.py --config configs/default.yaml
```

## Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Tests
```bash
pytest tests/ -v
```

## Architecture
```
SMILES ──► GATv2 (4 layers) ──► mean pool ──► projection ──► L2-norm ──► 256-d embedding
                                                                                │
                                                                    cosine similarity + InfoNCE
                                                                                │
812-d CellProfiler ──► Residual MLP ──► projection ──► L2-norm ──► 256-d embedding
```