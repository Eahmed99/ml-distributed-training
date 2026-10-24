# CIFAR-10 ResNet-18 training benchmark

This repository compares a CPU training baseline with synchronous data-parallel training on NVIDIA GPUs. A one-GPU run can be used as the controlled reference for measuring scaling from one to two GPU workers.

## What is trained

- Standard CIFAR-10: 50,000 training images and 10,000 test images, 10 classes, 32 x 32 RGB.
- ResNet-18, randomly initialized and adapted to CIFAR-10: a 3 x 3 stride-one input convolution, no initial max-pooling, and a ten-class output layer.
- Cross-entropy loss, SGD without momentum, default learning rate 0.01.
- The global batch size is 128 in every configuration (local batch size 64 on two GPUs).
- No augmentation or pretrained weights. This experiment is about training-system performance, not state-of-the-art image classification.

## Code

- `mlso/data.py` loads the standard CIFAR-10 Python archive using TorchVision and prepares the DataLoaders.
- `mlso/model.py` defines the CIFAR-10 ResNet-18.
- `mlso/train.py` is the CPU or single-GPU entry point and contains the shared training loop.
- `mlso/distributed.py` launches one training process per GPU via torchrun, uses DistributedSampler and PyTorch DDP, and synchronizes metrics.
- `mlso/evaluate.py` evaluates classification accuracy over the 10,000-image official test set.
- `analyze_results.py` prints comparisons and creates figures from recorded CSVs.

## Installation and dataset

Use a Python environment with compatible `torch` and `torchvision` installations. The pinned versions are listed in `requirements.txt`; for GPU machines, ensure that PyTorch was installed with CUDA/NCCL support.

The dataset must exist at `data/cifar-10-batches-py/`, including `data_batch_1` through `data_batch_5` and `test_batch`. Source dataset: https://www.cs.toronto.edu/~kriz/cifar.html . It is deliberately not included in Git.

All commands must run from the repository root.

## Run

CPU (four CPU computation threads):

```bash
python -m mlso.train --device cpu --cpu-threads 4 --epochs 5
```

Single GPU baseline for GPU scaling:

```bash
python -m mlso.train --device cuda --epochs 5
```

Two-GPU synchronous training:

```bash
torchrun --standalone --nproc-per-node=2 -m mlso.distributed --epochs 5
```

Short sanity tests, which **must not** be used for the final benchmark:

```bash
python -m mlso.train --device cpu --epochs 1 --max-batches 3
torchrun --standalone --nproc-per-node=2 -m mlso.distributed --epochs 1 --max-batches 3
```

The default is one full epoch. Omitting `--max-batches` processes every batch. Use identical `--epochs`, `--batch-size`, `--learning-rate` and `--seed` for comparison. Use separate working directories or save the output after each experiment if your execution environment resets between accelerator configurations.

Each experiment writes `benchmarks/{cpu,gpu_1,gpu_2}_metrics.csv` and a matching `*_latest.pt` checkpoint. Metrics record the global loss, full test accuracy, samples processed, and two different timing measurements.

Analyze once you have the CSVs together:

```bash
python analyze_results.py --dir benchmarks
```

The analysis creates `training_loss.png`, `accuracy_vs_training_time.png` and `epoch_times.png`. Neither the plots nor the report should contain fabricated measurements.

## Metrics and interpretation

- Training time excludes evaluation and checkpoint writing but includes batch loading, forward/backward passes and gradient synchronization.
- Epoch response time includes the training loop and test evaluation, but deliberately excludes checkpoint and CSV disk writes. The precise measurement boundary is documented so experiments can be compared fairly.
- Loss is an image-count-weighted average. Distributed ranks sum their loss totals and image counts with NCCL All-Reduce.
- The two-GPU epoch time uses the slower worker's measured training duration. The CPU and one-GPU cases use the same training-loop timing boundaries.
- DDP synchronizes gradients, not scalar losses. Rank zero evaluates the *unwrapped* model so that other ranks are not required to participate in an evaluation forward pass.
- GPU scaling speedup is `T_1GPU/T_2GPU` and parallel efficiency is `speedup/2 * 100%`.
- CPU-to-two-GPU acceleration is a heterogeneous hardware comparison. It should **not** be reported as two-GPU parallel efficiency.
- Accuracy gap is final test accuracy on one GPU minus test accuracy on two GPUs, in percentage points.
- Setting `--profile` exports a first-epoch PyTorch trace for each worker. The analysis script reports a sum of recorded NCCL All-Reduce event durations when present. This is a **collective-work measurement**, not a rigorous non-overlapped communication-stall estimate, as backward compute and NCCL can overlap.
- CPU and DDP can show different learning curves despite a matching global batch size because ResNet-18 has BatchNorm layers that use per-worker batch statistics under ordinary DDP.

## Reproducibility / remaining validation

All runs default to random seed 42. Dataset shuffling uses a seeded generator for the single-worker loader and `DistributedSampler(seed=42)` with `set_epoch(epoch)` for DDP. Shuffled sample orders are not identical between the one-worker and two-worker configurations. Seeds improve repeatability but exact numerical equality is not guaranteed across CPU/GPU hardware.

Run both quick tests before full training. Save the version of PyTorch, CUDA and GPU driver used. Run at least two full configurations with the same workload and preferably repeat trials to show timing variability.

No actual final benchmark table has been committed to this repository yet; it must be populated from measured outputs.
