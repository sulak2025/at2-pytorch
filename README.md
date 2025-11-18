## Project: Distributed CIFAR-10 Training with PyTorch DDP (Kaggle GPU)

This repository contains a fully functional example of **Distributed Data Parallel (DDP)** training using PyTorch on the CIFAR-10 dataset, implemented and tested in the **Kaggle GPU environment**.
It includes training scripts, documentation, environment setup instructions, troubleshooting, expected outputs, and a mapping to relevant assessment (UoC) requirements.

## Repository Structure
project-root/
│
├── model.py             # Single-GPU CNN training and evaluation
├── train_ddp.py         # Multi-GPU DDP training script
├── train.sh             # Shell script to launch DDP training
├── requirements.txt     # Python dependencies for easy environment setup
├── README.md            # Project documentation
└── data/                # CIFAR-10 dataset folder (auto-downloaded)

## Project Overview
This project demonstrates:
- Training of a classification model on CIFAR-10
- Scaling of the training using Distributed Data Parallel (DDP)
- Running code in the Kaggle GPU environment
- Include automation via shell script
- Package a reproducible training pipeline
- Document the process for assessment and reuse

## Environment Setup(Kaggle)
1. Enable GPU in the Kagge notebook: Settings -> Accelerator  -> GPU
2. Verify GPU availability

## Features

- **CNN architecture** with 2 convolutional layers, 3 fully connected layers, and ReLU activations.  
- **CIFAR-10 dataset support** with automatic download, train/test split, and normalization.  
- **Single-GPU training loop** with mini-batch loss computation, optimizer updates, and real-time loss tracking.  
- **Distributed Data Parallel (DDP)** support for multi-GPU training using `torch.multiprocessing` and the NCCL backend.  
- **Checkpointing system**: saves model weights every N epochs during DDP training and the final trained model at the end.  
- **Evaluation metrics**: computes overall accuracy and per-class accuracy on the test set.  
- **Reproducibility** ensured via fixed random seeds and deterministic CuDNN settings.  
- **Visualization**: displays sample images from training and test datasets for inspection.  
- **Configurable optimizer**: SGD with adjustable learning rate and momentum.  

## Running Instructions

### Single-GPU Training
!python model.py
#### Expected behaviour
- Training loss displayed periodically
- Final accuracy metrics on the test set, including per-class accuracy.

### Multi-GPU DDP Training
!chmod +x train.sh
!./train.sh
#### Expected behaviour
- GPU-specific logs per epoch
- Checkpoints saved at specified intervals.
- Final confirmation message indicating training completion.
## Final output:
[GPU0] Epoch 0 | Batchsize: 32 | Steps: 782
[GPU1] Epoch 0 | Batchsize: 32 | Steps: 782
Epoch 0 | Training checkpoint saved at checkpoint.pt
[GPU1] Epoch 1 | Batchsize: 32 | Steps: 782
[GPU0] Epoch 1 | Batchsize: 32 | Steps: 782

## Troubleshooting
- Permission denied for train.sh -> chmod+x train.sh
- No GPU detected -> Enable GPU in Kaggle noteebook settings.
- DDP hangs or NCCL runtime errors → Restart the notebook or adjust the MASTER_PORT in train_ddp.py.
- Only one GPU used → Kaggle free tier typically provides a single GPU.
- No output during training → Restart the kernel and rerun all cells.

## Environment requirements and dependencies
- Python 3.8+
- PyTorch ≥ 2.0
- Torchvision
- CUDA-enabled GPU (required for DDP)
- NCCL backend (provided by PyTorch in Kaggle)
- Matplotlib (for visualization)
- All required Python packages can be installed using 'requirements.txt' using:
```bash
pip install -r requirements.txt
This ensures that the exact versions of packages used in the project are installed.

## Notes
- CIFAR-10 dataset downloads automatically at runtime.
- Checkpoints allow resuming multi-GPU training from intermediate states (however, my code does not save checkpoints.)
- Optimized for Kaggle GPU; CPU-only DDP training not supported.
- This README serves as the complete guide for setup, execution, and troubleshooting.
 
