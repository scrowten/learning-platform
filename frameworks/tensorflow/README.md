# Framework Track — TensorFlow / Keras

TensorFlow/Keras implementations of core modules.

Note: As of 2024-2025, PyTorch has overtaken TensorFlow in research and is rapidly growing in industry. However, TF/Keras remains widely deployed in production (especially Google infrastructure) and has a strong ecosystem (TF Serving, TFX, TF Lite).

## Contents (planned)

| File | Topic |
|---|---|
| `attention_keras.ipynb` | Transformer with Keras functional API |
| `transfer_learning_tf.ipynb` | Fine-tuning with TF Hub models |

## Key Keras Patterns

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Subclassing API (mirrors PyTorch nn.Module)
class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, d_model // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model),
        ])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x, training=False):
        attn_out = self.attn(x, x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x, training=training)
        return self.ln2(x + ffn_out)
```
