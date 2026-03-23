# Assignment 4: Optimizing Transformer Translation with Ray Tune and Optuna

## Author Details
- Name: Jyoti Dwivedi
- Roll No: M25CSA010
- Repository Branch: Assignment-4

## Overview
This repository contains my Assignment 4 submission for optimizing a custom PyTorch Transformer model for English-to-Hindi translation using Ray Tune with OptunaSearch and ASHAScheduler.

## Submitted Files
- Tuned notebook: M25CSA010_ass_4_tuned_en_to_hi.ipynb
- Best tuned model: M25CSA010_ass_4_best_model.pth
- Report: M25CSA010_ass_4_report.pdf

## Baseline Metrics (Original 100-Epoch Training)
- Total epochs: 100
- Training time: 111.53 minutes
- Final training loss: 0.0970
- BLEU score (NLTK): 47.41 (0.4741)

## Hyperparameter Tuning Setup
- Framework: Ray Tune
- Search algorithm: OptunaSearch
- Scheduler: ASHAScheduler
- Objective: Minimize validation loss
- Number of trials: 12
- GPU usage: 1 GPU per trial when available

### Hyperparameters Tuned
- Learning rate : loguniform(1e-5,1e-3)
- Batch size : {16,32,64}
- Number of attention heads :{4,8} 
- Feedforward dimension :  {1024,2048}
- Dropout : uniform(0.1,0.4)
- Number of encoder/decoder layers : {4,6}
- Epoch budget per trial :  {20,25,30}

## Best Configuration Found
- lr: 2.95244e-05
- batch_size: 64
- num_heads: 4
- d_ff: 2048
- dropout: 0.10734
- num_layers: 6
- d_model: 512
- grad_clip: 1.0
- epochs: 30
- Best validation loss: 2.6154

## Final Tuned Model Metrics
- Tuning time: 136.45 minutes
- Final retraining time: 32.28 minutes
- Final retraining loss (epoch 30): 1.3345
- BLEU score (NLTK): 64.99 (0.6499)

## Baseline vs Tuned (Summary)
- Baseline BLEU: 0.4741 at 100 epochs
- Tuned BLEU: 0.6499 at 30 epochs

Result: The tuned model exceeds BLEU 0.50 and outperforms the baseline in significantly fewer epochs.

## How to Run
1. Open M25CSA010_ass_4_tuned_en_to_hi.ipynb.
2. Ensure dataset path is correct for your environment.
3. Run all notebook cells in order:
   - Data loading and preprocessing
   - Baseline training and evaluation
   - Ray Tune + Optuna search
   - Final retraining with best config
   - BLEU evaluation and model saving

## Notes
- The final tuned weights are saved as M25CSA010_ass_4_best_model.pth.

## Assignment Outcome
This submission satisfies the Assignment 4 requirements:
- Baseline metrics documented
- Training loop refactored for Ray Tune
- 4 or more hyperparameters tuned
- OptunaSearch and ASHAScheduler used
- Tuned model beats baseline BLEU with fewer epochs
- Report and model artifacts provided
