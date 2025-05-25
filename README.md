This repo contains an extention to the implementation based on the following paper:

## Context-aware Pedestrian Trajectory Prediction with Multimodal Transformer

Haleh Damirchi, Michael Greenspan, Ali Etemad

**ICIP 2023**  
[[paper](https://arxiv.org/pdf/2307.03786)]

---

# Context-Aware Trajectory Prediction + VIT

## Overview
This project implements a context-aware trajectory prediction framework for pedestrians, inspired by the work of Damirchi et al. (ICIP 2023). **This implementation extends the original approach by incorporating a Vision Transformer (ViT) for visual context encoding, enabling richer scene understanding from images.** The core model, `Trajnet`, fuses trajectory, speed, and visual context to predict future pedestrian trajectories in dynamic environments. The framework is modular, extensible, and built with PyTorch.

## Citation
If you use this code or ideas from this repository, please cite the following paper by Damirchi et al.:

```
@inproceedings{damirchi2023context,
  title={Context-aware Pedestrian Trajectory Prediction with Multimodal Transformer},
  author={Damirchi, Haleh and Greenspan, Michael and Etemad, Ali},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  year={2023},
  organization={IEEE},
  url={https://arxiv.org/pdf/2307.03786}
}
```

## Features
- **Multimodal Fusion:** Combines trajectory, speed, and visual context for robust prediction.
- **Vision Transformer Integration:** Uses a ViT-based encoder for extracting visual features from scene images.
- **Flexible Training & Evaluation:** Scripts for training (`train_deterministic.py`) and evaluation (`eval_deterministic.py`).
- **Custom Loss Function:** Uses RMSE for trajectory prediction accuracy.
- **Configurable:** All hyperparameters and paths are set via command-line arguments.

## Installation
1. **Clone the repository:**
   ```bash
   git clone 
   cd Context-Aware
   ```
2. **Set up the Python environment:**
   - (Recommended) Use a virtual environment or conda.
   - Install dependencies (PyTorch, torchvision, timm, tensorboard, etc.):
     ```bash
     pip install torch torchvision timm tensorboard
     ```
   - Additional dependencies may be required depending on your dataset and environment.

## Usage
### Training
Run the training script with the recommended settings:
```bash
python train_deterministic.py --batch_size 32 --version_name Model --hidden_size_traj 256 --hidden_size_sp 128 --d_model_traj 256 --d_model_sp 128 --d_inner 1024 --d_k 32 --d_v 32 --n_head 16 --epochs 10 --patience 10
```

### Evaluation
Evaluate a trained model checkpoint (replace `<path-to-checkpoint>` and `<path-to-dataset>` as needed):
```bash
python eval_deterministic.py --data_root <path-to-dataset> --checkpoint <path-to-checkpoint> --batch_size 32 --hidden_size_traj 256 --hidden_size_sp 128 --d_model_traj 256 --d_model_sp 128 --d_inner 1024 --d_k 32 --d_v 32 --n_head 16
```

### Common Arguments
- `--data_root`: Path to the dataset root directory.
- `--checkpoint`: Path to a model checkpoint (for evaluation or resuming training).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training/evaluation.
- `--gpu`: GPU device ID to use.
- `--version_name`: Name for experiment versioning (used for logs/checkpoints).
- See `configs/pie/pie.py` for all available arguments and their defaults.

## Directory Structure
```
Context-Aware/
├── train_deterministic.py      # Training script
├── eval_deterministic.py       # Evaluation script
├── configs/                    # Configuration files
│   └── pie/pie.py              # Argument parser and defaults
├── lib/                        # Core library
│   ├── models/                 # Model definitions (Trajnet, encoders, etc.)
│   ├── utils/                  # Data loading, training, evaluation utilities
│   └── losses/                 # Loss functions (e.g., RMSE)
├── runs/                       # TensorBoard logs
└── README.md                   # Project documentation
```


## Credits
- Built with [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), and [TensorBoard](https://www.tensorflow.org/tensorboard).
- Model and code structure inspired by recent research in context-aware trajectory prediction.

## License

## Dataset Preparation

This repository provides utilities to help you prepare your dataset for training and evaluation:

- **`lib/utils/data_utils.py`**: Contains functions for data loading, normalization, and batching. Key utilities include:
  - `build_data_loader`: Constructs PyTorch DataLoaders for train/val/test splits, handling batching and shuffling.
  - `bbox_normalize` / `bbox_denormalize`: Normalize and denormalize bounding box coordinates for consistent model input.
  - `cxcywh_to_x1y1x2y2`: Converts bounding box formats as needed.
  - `set_seed`: Ensures reproducibility by setting random seeds.

- **`lib/utils/extract_set_frames.py`**: Script for extracting specific frames from video clips based on annotation CSV files. This is essential for preparing the visual context (images) required by the Vision Transformer (ViT) in the model. Usage example:
  ```bash
  python lib/utils/extract_set_frames.py
  ```
  - The script reads annotation CSVs, extracts the specified frames from each video, and saves them as images in an organized directory structure.

Make sure to run these utilities as needed to prepare your dataset before training the model.

---
For questions or contributions, please open an issue or pull request.
