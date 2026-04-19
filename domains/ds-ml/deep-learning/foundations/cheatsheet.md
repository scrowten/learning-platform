# MLP & Backpropagation — Cheatsheet

## Activation Functions

| Name | Formula | Derivative | Typical Use |
|---|---|---|---|
| Sigmoid | `1 / (1 + exp(-z))` | `sigma(z) * (1 - sigma(z))` | Binary output |
| Tanh | `(exp(z) - exp(-z)) / (exp(z) + exp(-z))` | `1 - tanh(z)^2` | Hidden (old); RNNs |
| ReLU | `max(0, z)` | `1 if z > 0 else 0` | Default hidden layer |
| GELU | `z * Phi(z)` (Gaussian CDF) | (computed numerically) | Transformers, BERT |
| Softmax | `exp(z_k) / sum(exp(z_j))` | (vector operation) | Multi-class output |

## Loss Functions

| Loss | Formula | Gradient w.r.t. output |
|---|---|---|
| MSE | `(1/N) sum(y_hat - y)^2` | `(2/N) * (y_hat - y)` |
| Cross-entropy + softmax | `-(1/N) sum(y * log(p))` | `p - y` (combined) |
| Binary CE + sigmoid | `-(1/N) sum(y*log(p) + (1-y)*log(1-p))` | `p - y` (combined) |

## Backpropagation (2-layer MLP)

```
# Forward
z1 = W1 @ x + b1
a1 = ReLU(z1)
z2 = W2 @ a1 + b2
p  = softmax(z2)

# Backward
delta2 = p - y                        # dL/dz2 (CE + softmax, combined)
dW2    = (1/N) * delta2.T @ a1        # shape: (n2, n1)
db2    = (1/N) * delta2.sum(axis=0)   # shape: (n2,)

da1    = delta2 @ W2                  # shape: (N, n1)
delta1 = da1 * (z1 > 0)              # through ReLU: zero where z1 <= 0
dW1    = (1/N) * delta1.T @ x        # shape: (n1, n0)
db1    = (1/N) * delta1.sum(axis=0)  # shape: (n1,)
```

## Optimizer Update Rules

```
# SGD
# formula: theta = theta - lr * grad
theta = theta - lr * grad

# SGD + Momentum
# formula: v = momentum * v - lr * grad; theta = theta + v
v     = momentum * v - lr * grad
theta = theta + v

# Adam
# formula: m_hat = m / (1 - beta1^t); v_hat = v / (1 - beta2^t)
#          theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
m     = beta1 * m + (1 - beta1) * grad              # 1st moment
v     = beta2 * v + (1 - beta2) * grad**2           # 2nd moment
m_hat = m / (1 - beta1**t)                          # bias corrected
v_hat = v / (1 - beta2**t)                          # bias corrected
theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
# typical: beta1=0.9, beta2=0.999, eps=1e-8, lr=1e-3
```

## Weight Initialization

| Method | Formula | Use When |
|---|---|---|
| Xavier (Glorot) | `std = sqrt(2 / (n_in + n_out))` | Sigmoid / tanh / linear activations |
| He (Kaiming) | `std = sqrt(2 / n_in)` | ReLU / Leaky ReLU activations |
| Zeros | `W = 0` | NEVER — symmetry breaking fails |

## Regularization Summary

| Method | Applied Where | Effect |
|---|---|---|
| L2 (weight decay) | Loss term: `+ (lambda/2) * sum(W^2)` | Shrinks weights toward 0 |
| L1 | Loss term: `+ lambda * sum(|W|)` | Pushes weights to exactly 0 (sparse) |
| Dropout | During training: zero neurons with prob p | Ensemble of sub-networks |
| Batch Norm | After linear, before activation | Stabilizes activations, mild regularization |
| Early stopping | Training loop: stop at best val checkpoint | Implicit regularization |

## Training Checklist

1. Normalize input data (zero mean, unit variance)
2. Choose initialization: He for ReLU layers, Xavier otherwise
3. Start with Adam (lr=1e-3) as the default optimizer
4. Add dropout (0.1–0.3) and L2 weight decay (1e-4) to fight overfitting
5. Monitor both train and val loss every epoch — check generalization gap
6. Use a learning rate schedule (cosine annealing or step decay)
7. Run numerical gradient check on a small batch before full training
8. Save the model checkpoint at the best validation loss
9. Plot loss curves — identify if underfit or overfit and adjust accordingly
10. Inspect misclassified examples to understand model failure modes
