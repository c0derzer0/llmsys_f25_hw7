# Training Process Documentation

This document summarizes the training experiments conducted for Assignment 7: RLHF Training.

## Overview

I conducted multiple training runs to improve the reward model accuracy and subsequently trained RLHF models using different reward models. This document outlines the experiments, hyperparameters, and results.

---

## Reward Model Training

### Experiment 1: Default Hyperparameters
- **Accuracy**: ~60%
- **Hyperparameters**: Default settings from config
- **Data samples**: Default (~9,000)
- **Batch size**: Default (4)

### Experiment 2: Improved Reward Model (Final)
- **Accuracy**: ~66%
- **Hyperparameters**:
  - Data samples: **20,000**
  - Batch size: **64**
- **Result**: Improved accuracy from 60% to 66%

---

## RLHF Training

### Run 1: Using 60% Accuracy Reward Model
- **Reward Model**: First reward model (~60% accuracy)
- **Batch size**: Default (4)
- **Reward Normalization**: ON (forgot to disable)
- **Evaluation Results**: Saved in `evaluation_results_60_accuracy_reward_model/`
- **Improvement**: 95% of samples showed improvement
- **Note**: Training plots not saved for this run

### Run 2: Using 66% Accuracy Reward Model (Final)
- **Reward Model**: Improved reward model (~66% accuracy)
- **Batch size**: **8** (changed from default)
- **Reward Normalization**: **OFF** (disabled for this run)
- **Evaluation Results**: Saved in `evaluation_results/`
- **Improvement**: 99% of samples showed improvement
- **Note**: The change in batch size affected the training curves appearance due to different batch size scaling

---

## Implementation Notes

### True GAE Implementation
I implemented the true Generalized Advantage Estimation (GAE) algorithm as discussed in the course forum:
- Reference: [Ed Discussion Thread #7420638](https://edstem.org/us/courses/81633/discussion/7420638)

---

## Files for Grading

### Evaluation Results
| Directory | Description |
|-----------|-------------|
| `evaluation_results_60_accuracy_reward_model/` | Results from RLHF model trained with 60% accuracy reward model |
| `evaluation_results/` | Results from RLHF model trained with improved 66% accuracy reward model |

### Key Files
- `evaluation_results_60_accuracy_reward_model/evaluation_summary.json` - Summary metrics (60% RM)
- `evaluation_results_60_accuracy_reward_model/rlhf_model_outputs.json` - Model outputs (60% RM)
- `evaluation_results_60_accuracy_reward_model/detailed_comparison.json` - Detailed comparison (60% RM)
- `evaluation_results/evaluation_summary.json` - Summary metrics (66% RM)
- `evaluation_results/rlhf_model_outputs.json` - Model outputs (66% RM)
- `evaluation_results/detailed_comparison.json` - Detailed comparison (66% RM)

### Training Logs
- All reward model training logs are in `logs/`
- All RLHF training logs are in `logs/`
- Training curves are in `plots/`

---

## Summary of Results

| Metric | 60% RM Run | 66% RM Run |
|--------|------------|------------|
| Samples Improved | 95% | 99% |
| Reward Model Accuracy | ~60% | ~66% |
| RLHF Batch Size | 4 | 8 |
| Reward Normalization | ON | OFF |

Both RLHF runs showed improvement over the base model. The first run (60% RM) showed larger per-sample improvement on 95% of samples, while the second run (66% RM) achieved improvement on 99% of samples with a more accurate reward model.

---

## Code Implementation

### Reward Model (`src/reward_model.py`)
- Implemented `compute_loss()` function using `MarginRankingLoss`
- Correctly computes rewards for chosen and rejected responses

### RLHF Trainer (`src/rlhf_trainer.py`)
- Implemented `VERLTrainer` class
- Implemented true GAE for advantage estimation

