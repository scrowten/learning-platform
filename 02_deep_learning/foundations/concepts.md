# MLP & Backpropagation — Theory

---

## 1. The Perceptron

The perceptron is the single neuron — the basic unit of every neural network.

Given an input vector **x** ∈ ℝⁿ, a weight vector **w** ∈ ℝⁿ, and a bias b ∈ ℝ:

```
z = w^T x + b          # linear combination (pre-activation)
a = f(z)               # activation (post-activation, output)
```

`z` is the **pre-activation** (or **logit**). `a` is the **activation**.

### Decision Boundary

For binary classification, the decision boundary is where `z = 0`:

```
w^T x + b = 0
```

This is a hyperplane in n-dimensional space. The perceptron can only separate linearly
separable classes. XOR is the classic failure case — it requires at least one hidden layer.

---

### Activation Functions

Activation functions introduce non-linearity, which is what allows stacked layers to learn
non-trivial functions. Without non-linearity, composing linear layers is still linear.

**Sigmoid**

```
sigma(z) = 1 / (1 + exp(-z))

# derivative (important for backprop):
sigma'(z) = sigma(z) * (1 - sigma(z))
```

Output range: (0, 1). Historically popular for binary classification output.
Problem: **vanishing gradients** — for large |z|, the derivative approaches 0, starving
earlier layers of gradient signal.

**Tanh**

```
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))

# derivative:
tanh'(z) = 1 - tanh(z)^2
```

Output range: (-1, 1). Zero-centered (better than sigmoid). Still saturates.

**ReLU** (Rectified Linear Unit)

```
ReLU(z) = max(0, z)

# derivative (subgradient):
ReLU'(z) = 1 if z > 0 else 0
```

The default choice for hidden layers. Does not saturate for positive values; very cheap
to compute. Problem: **dying ReLU** — neurons that consistently receive negative pre-
activations get zero gradient and never update.

**GELU** (Gaussian Error Linear Unit)

```
GELU(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))

# In practice, often approximated or computed exactly via the Gaussian CDF.
```

Used in BERT, GPT-2, and most modern Transformers. Smoother than ReLU, performs better
in deep models. The output is z weighted by its probability under a Gaussian.

**Softmax** (for multi-class output layer)

```
softmax(z)_k = exp(z_k) / sum_j(exp(z_j))
```

Converts a vector of logits into a probability distribution. Note: softmax is applied
to an entire vector, not element-wise. Use the numerically stable form:

```
softmax(z)_k = exp(z_k - max(z)) / sum_j(exp(z_j - max(z)))
```

---

## 2. Multi-Layer Perceptron (MLP)

An MLP stacks multiple layers of neurons. Each layer applies a linear transformation
followed by an activation function.

### Layer Notation

For layer `l` (1-indexed), with input `a^(l-1)` (where `a^(0) = x`, the input):

```
z^(l) = W^(l) a^(l-1) + b^(l)      # linear step
a^(l) = f^(l)(z^(l))                # activation step
```

Where:
- `W^(l)` ∈ ℝ^(n_l × n_{l-1}) — weight matrix for layer l
- `b^(l)` ∈ ℝ^(n_l)            — bias vector for layer l
- `f^(l)` — activation function for layer l (usually ReLU for hidden, softmax for output)

### Forward Pass (2-layer MLP example)

```
Input: x ∈ R^(n_0)

Layer 1 (hidden):
  z^(1) = W^(1) x + b^(1)      shape: (n_1,)
  a^(1) = ReLU(z^(1))           shape: (n_1,)

Layer 2 (output):
  z^(2) = W^(2) a^(1) + b^(2)  shape: (n_2,)
  a^(2) = softmax(z^(2))        shape: (n_2,)  ← predicted class probabilities
```

For a batch of N examples, all operations are batched: x ∈ ℝ^(N × n_0), W^(1) x^T gives
(n_1 × N), etc. In practice we usually write it as (N × n_1) with x on the left.

### Universal Approximation Theorem (Intuition)

A feedforward network with a single hidden layer of sufficient width can approximate any
continuous function on a compact subset of ℝⁿ to arbitrary precision.

This tells us neural networks are expressive enough — but it says nothing about whether
gradient descent can find the approximating weights in practice. In practice, depth
(more layers) is more efficient than width (more neurons per layer) for many problems.

---

## 3. Loss Functions

The loss function measures how wrong the model's predictions are. Backpropagation computes
the gradient of the loss with respect to every parameter.

### Mean Squared Error (MSE) — Regression

```
L_MSE = (1/N) * sum_i (y_hat_i - y_i)^2

# Gradient with respect to y_hat:
dL/d(y_hat_i) = (2/N) * (y_hat_i - y_i)
```

Penalizes large errors heavily (quadratic). Appropriate when output is a real number.

### Cross-Entropy Loss — Classification

For true label y (one-hot) and predicted probability p = softmax(z):

```
L_CE = -(1/N) * sum_i sum_k y_{i,k} * log(p_{i,k})
```

For binary classification (single output, sigmoid activation):

```
L_BCE = -(1/N) * sum_i [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

**Gradient of cross-entropy + softmax (combined):**

The combined gradient has a beautifully clean form:

```
dL/dz^(L) = p - y        # predicted probability minus true one-hot label
```

This is one of the most important results in neural network training. The gradient
of the output layer is simply the error between predicted and true probabilities.

**Why cross-entropy, not MSE, for classification?**

MSE on probabilities creates vanishing gradients near 0 and 1 (sigmoid saturation).
Cross-entropy with softmax has constant gradient magnitude — learning is much faster.

---

## 4. Backpropagation

Backpropagation is the algorithm for computing ∂L/∂W and ∂L/∂b for every layer,
using the chain rule on the computational graph.

### The Chain Rule

If L depends on z, which depends on W, then:

```
dL/dW = (dL/dz) * (dz/dW)
```

For vectors/matrices, these are Jacobians, but in practice we work with shapes directly.

### Computational Graph View

Forward pass builds a graph of operations. Backprop traverses it in reverse, multiplying
local gradients at each node.

```
x → [W^(1), b^(1)] → z^(1) → ReLU → a^(1) → [W^(2), b^(2)] → z^(2) → softmax → L
```

Each arrow represents an operation. Backward pass flows gradients from L back to x.

### Full Derivation: 2-Layer MLP with Cross-Entropy

**Setup:**
- Input: x ∈ ℝ^(n_0)
- Hidden: z^(1) = W^(1)x + b^(1), a^(1) = ReLU(z^(1))
- Output: z^(2) = W^(2)a^(1) + b^(2), p = softmax(z^(2))
- Loss: L = cross_entropy(p, y)

**Step 1 — Gradient at output layer (combined softmax + cross-entropy):**

```
delta^(2) = dL/dz^(2) = p - y                  # shape: (n_2,)
```

**Step 2 — Gradients for W^(2) and b^(2):**

```
dL/dW^(2) = delta^(2) ⊗ a^(1)                  # outer product
           = (a^(1))^T @ delta^(2) if batched   # shape: (n_2, n_1) — need to transpose for W^(2) shape (n_2, n_1)
dL/db^(2) = delta^(2)                            # shape: (n_2,)
```

For batched case (N examples), sum over the batch:

```
dL/dW^(2) = (1/N) * delta^(2).T @ a^(1)        # shape: (n_2, n_1)
dL/db^(2) = (1/N) * sum over batch of delta^(2) # shape: (n_2,)
```

**Step 3 — Propagate error back through W^(2):**

```
da^(1) = (W^(2))^T @ delta^(2)                  # shape: (n_1,)
```

**Step 4 — Gradient through ReLU:**

ReLU derivative is 1 where z^(1) > 0, else 0. This is the Hadamard (element-wise) product:

```
delta^(1) = da^(1) * ReLU'(z^(1))
           = da^(1) * (z^(1) > 0)               # shape: (n_1,)
```

This step is the **local gradient** of the ReLU. Neurons with z^(1) <= 0 receive zero
gradient — they do not update their upstream weights.

**Step 5 — Gradients for W^(1) and b^(1):**

```
dL/dW^(1) = (1/N) * delta^(1).T @ x            # shape: (n_1, n_0)
dL/db^(1) = (1/N) * sum over batch of delta^(1) # shape: (n_1,)
```

### Gradient Flow Intuition

- Gradients flow from the loss backward through every operation.
- At each linear layer: multiply by the transposed weight matrix.
- At each activation: multiply element-wise by the activation derivative.
- Vanishing gradients occur when activation derivatives are consistently small (<1).
- Exploding gradients occur when weight matrices have large singular values.
- Residual connections (as in ResNets and Transformers) allow gradient to flow directly
  back without passing through activations — this is why deep networks became trainable.

---

## 5. Gradient Descent Variants

All optimizers minimize L by iteratively updating parameters θ using gradient information.

### SGD (Stochastic Gradient Descent)

```
# formula: theta <- theta - lr * grad
theta = theta - lr * dL/dtheta
```

Stochastic: use a mini-batch rather than the full dataset to estimate the gradient.
Computationally efficient; gradient estimates have variance that can help escape local minima.

**Problem**: sensitive to learning rate. Slow convergence, especially in ill-conditioned
problems (long valleys in loss landscape).

### SGD + Momentum

```
# formula: v <- momentum * v - lr * grad; theta <- theta + v
v = momentum * v - lr * dL/dtheta
theta = theta + v
```

Accumulates velocity in directions of consistent gradient. Dampens oscillations.
Typical momentum: 0.9.

### RMSProp

```
# formula: s <- rho * s + (1 - rho) * grad^2; theta <- theta - lr * grad / sqrt(s + eps)
s = rho * s + (1 - rho) * (dL/dtheta)^2       # running mean of squared gradients
theta = theta - lr * dL/dtheta / sqrt(s + eps)
```

Adapts learning rate per parameter: parameters with large gradients get smaller updates.
Good for non-stationary problems (recurrent networks). Typical rho: 0.99.

### Adam (Adaptive Moment Estimation)

Adam combines momentum and RMSProp with bias correction:

```
# formula:
m = beta1 * m + (1 - beta1) * grad             # 1st moment (mean)
v = beta2 * v + (1 - beta2) * grad^2           # 2nd moment (uncentered variance)
m_hat = m / (1 - beta1^t)                       # bias correction
v_hat = v / (1 - beta2^t)                       # bias correction
theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

Typical hyperparameters: `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `lr=1e-3`.

Bias correction compensates for the fact that m and v are initialized to 0 and are
biased toward 0 in early iterations.

Adam is the **default optimizer** in most deep learning today. It is robust to learning
rate choice and works well across a wide range of architectures.

---

## 6. Weight Initialization

Weights cannot be initialized to zero — all neurons in a layer would compute the same
output and receive the same gradient (symmetry breaking problem). Random initialization
breaks symmetry. The scale matters:

- Too small: activations shrink to zero with each layer (vanishing activations)
- Too large: activations explode, saturating activations and causing vanishing gradients

### Xavier / Glorot Initialization

Designed for sigmoid/tanh activations. Keeps variance of activations constant across layers:

```
W ~ Uniform(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))
# equivalently, normal:
W ~ Normal(0, sqrt(2 / (n_in + n_out)))

# formula: std = sqrt(2 / (n_in + n_out))
```

Where `n_in` = number of input neurons, `n_out` = number of output neurons.

### He / Kaiming Initialization

Designed for ReLU activations. Accounts for the fact that ReLU zeros half of inputs:

```
W ~ Normal(0, sqrt(2 / n_in))

# formula: std = sqrt(2 / n_in)
```

The factor of 2 compensates for ReLU discarding half the distribution.

**Rule of thumb**: use He initialization for ReLU, Xavier for tanh/sigmoid/linear.

---

## 7. Regularization

Regularization techniques reduce overfitting by constraining the model.

### L2 Regularization (Ridge / Weight Decay)

Adds a penalty on the squared magnitude of weights to the loss:

```
L_total = L_data + (lambda/2) * sum(W^2)

# gradient addition:
dL_total/dW = dL_data/dW + lambda * W
```

Effect in SGD update: `W = W - lr * (grad + lambda * W)` — weights are "decayed" toward 0.
Encourages small, distributed weights. Most commonly used regularizer in deep learning.

### L1 Regularization (Lasso)

Adds a penalty on the absolute magnitude of weights:

```
L_total = L_data + lambda * sum(|W|)

# gradient addition:
dL_total/dW = dL_data/dW + lambda * sign(W)
```

Encourages sparsity — some weights are driven exactly to 0. Less common in deep learning
than L2, but useful when sparsity is desired.

### Dropout

During training, randomly zero out each neuron's activation with probability p:

```
# Training:
mask = Bernoulli(1 - p)     # 1 with prob (1-p), 0 with prob p
a_dropped = a * mask / (1 - p)   # scale to keep expected value the same

# Inference:
a_inference = a             # no dropout; no scaling needed (already compensated above)
```

The `/ (1 - p)` scaling (inverted dropout) is the standard. Without it, activations
at inference would be (1-p) times smaller than during training.

Dropout effectively trains an ensemble of 2^n different sub-networks (exponential in
number of neurons). At inference, we use the full network as an approximation to the
ensemble average.

Typical rates: 0.1–0.5. Larger models can tolerate higher dropout.

### Batch Normalization

Normalize each mini-batch's activations to zero mean and unit variance, then scale and shift:

```
mu_B    = (1/m) * sum_i(x_i)                   # batch mean
sigma_B = sqrt((1/m) * sum_i((x_i - mu_B)^2))  # batch std
x_hat   = (x - mu_B) / (sigma_B + eps)          # normalize
y       = gamma * x_hat + beta                  # scale and shift (learned)
```

At inference: use running statistics accumulated during training (not the current batch).

Benefits:
- Reduces internal covariate shift (activations don't shift as weights update)
- Enables higher learning rates
- Acts as a mild regularizer (due to batch statistics noise)

---

## 8. Training Dynamics

### Learning Rate Schedules

A fixed learning rate is rarely optimal:
- Too large: training is unstable, loss oscillates
- Too small: training is slow; may get stuck

**Step decay**: reduce lr by a factor every k epochs:

```
lr(epoch) = lr_0 * decay_rate ^ floor(epoch / drop_every)
```

**Cosine annealing**: lr follows a cosine curve from `lr_max` to `lr_min`:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

Smoother than step decay. Can be combined with warm restarts (SGDR).

**Warmup**: start with a very small lr and ramp up for the first few thousand steps,
then decay. Used in Transformer training (as in "Attention Is All You Need").

### Overfitting and Underfitting

- **Underfitting**: model is too simple; both train and val loss are high. Fix: larger
  model, more capacity, better features, train longer.
- **Overfitting**: model memorizes training data; train loss low but val loss high. Fix:
  regularization (L2, dropout), more data, data augmentation, early stopping.
- **Generalization gap** = val_loss - train_loss. Monitor this during training.

### Early Stopping

Stop training when validation loss stops improving. Keep a copy of the model at the
best validation checkpoint. This is a form of implicit regularization.

```
best_val_loss = infinity
patience_counter = 0
PATIENCE = 10  # epochs to wait without improvement

for epoch in range(MAX_EPOCHS):
    train(...)
    val_loss = evaluate(...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        break  # stop training
```

---

## Key Equations Summary

| Concept | Formula |
|---|---|
| Forward pass | `z^l = W^l a^(l-1) + b^l`, `a^l = f(z^l)` |
| Softmax + CE gradient | `delta^L = p - y` |
| Backprop through linear | `delta^(l) = (W^(l+1))^T delta^(l+1) * f'(z^l)` |
| Weight gradient | `dL/dW^l = delta^l (a^(l-1))^T` |
| He initialization | `std = sqrt(2 / n_in)` |
| Adam update | `theta -= lr * m_hat / (sqrt(v_hat) + eps)` |

---

## Further Reading

- [LeCun et al., "Efficient BackProp" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) — practical backprop tips
- [Goodfellow, Bengio, Courville, "Deep Learning" (2016)](https://www.deeplearningbook.org/) — chapters 6–8 cover this module
- [He et al., "Delving Deep into Rectifiers" (2015)](https://arxiv.org/abs/1502.01852) — He initialization paper
- [Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010)](http://proceedings.mlr.press/v9/glorot10a.html) — Xavier initialization paper
- [Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)](https://arxiv.org/abs/1412.6980) — Adam paper
- [Ioffe & Szegedy, "Batch Normalization" (2015)](https://arxiv.org/abs/1502.03167) — batch norm paper
- [Srivastava et al., "Dropout" (2014)](https://jmlr.org/papers/v15/srivastava14a.html) — dropout paper
- [Michael Nielsen, "Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) — free online book, very accessible
