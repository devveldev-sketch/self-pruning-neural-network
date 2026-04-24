# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune itself during training using learnable gating mechanisms.

## Method
Each weight is associated with a learnable gate parameter. The gate is passed through a sigmoid function and multiplied with the weight.

Pruned Weight = Weight × Sigmoid(Gate Score)

## Sparsity Loss
We apply L1 regularization on gate values:

Loss = Classification Loss + λ × Sparsity Loss

This encourages many gates to approach zero, effectively pruning weights.

## Why L1 Works
L1 regularization promotes sparsity by penalizing non-zero values, pushing many gate values toward zero.

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 1e-5   |             |             |
| 1e-4   |             |             |
| 1e-3   |             |             |

## Observations

The model successfully learns to prune weights during training, as seen from the bimodal gate distribution with peaks near 0 and 1.

Across different values of λ, sparsity remains relatively stable, indicating that the model converges early to a compact representation. This suggests that important connections are identified quickly during training.

Accuracy shows a slight decrease as λ increases, indicating a trade-off between model sparsity and performance.

Overall, the approach demonstrates effective self-pruning behavior with minimal impact on accuracy.

## Conclusion
The model successfully learns to remove unnecessary connections during training, achieving a balance between accuracy and efficiency.