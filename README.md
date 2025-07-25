# Dot-Product Attention Mechanism with PyTorch

This project demonstrates the fundamental **dot-product attention mechanism** using PyTorch tensors. It illustrates how queries, keys, and values are used to compute attention weights and subsequently derive a context vector, along with a simple visualization of the attention weights. This is a core component found in advanced neural network architectures, most notably the Transformer.

## What is Attention?

The attention mechanism in neural networks enables a model to selectively focus on relevant parts of an input sequence when processing another part. Instead of relying on a fixed-size representation of an entire sequence, attention dynamically weighs the importance of different input elements based on their relevance to a given query.

The process typically involves:

* **Queries (Q)**: Represents the element for which we want to find relevant information.
* **Keys (K)**: Represents the elements available to be attended to.
* **Values (V)**: The actual information associated with the keys, which will be weighted and aggregated.

The steps are:

1.  **Calculate Scores**: A measure of similarity or relevance between each Query and all Keys (e.g., using dot product).
2.  **Normalize Scores**: Applying a softmax function to convert these raw scores into attention weights. These weights sum up to 1 for each query, indicating the "importance" or "focus" on each key's corresponding value.
3.  **Compute Context Vector**: A weighted sum of the Values, where the weights are the calculated attention weights. This aggregated vector is the "context" that the model uses, having selectively emphasized the most pertinent information.

## Project Overview

The Python script `pytorch_attention_example.py` (assuming you save the code as such) illustrates the dot-product attention mechanism using PyTorch:

1.  **Define Inputs**: Initializes sample `queries`, `keys`, and `values` as PyTorch tensors.
2.  **Compute Attention Scores**: Performs a matrix multiplication (`torch.matmul`) between `queries` and the transpose of `keys` (`keys.T`) to get the raw attention scores.
3.  **Apply Softmax**: Uses `torch.nn.functional.softmax` to normalize the scores across the last dimension (`dim=-1`), converting them into `attention_weights`.
4.  **Compute Context Vector**: Calculates the weighted sum of `values` by performing another matrix multiplication with the `attention_weights`.
5.  **Print Results**: Displays the computed `attention_weights` and the final `context` vector.
6.  **Visualize Weights**: Generates a heatmap (`matplotlib.pyplot.matshow`) of the `attention_weights`, providing a visual representation of how each query attends to different keys.

## Mathematical Formulation (Dot-Product Attention in PyTorch)

Given:
* Queries $Q$ (shape: $N_Q \times D_K$)
* Keys $K$ (shape: $N_K \times D_K$)
* Values $V$ (shape: $N_K \times D_V$)

Where:
* $N_Q$ is the number of queries.
* $N_K$ is the number of keys (and values).
* $D_K$ is the dimensionality of keys and queries.
* $D_V$ is the dimensionality of values.

1.  **Attention Scores**:
    $Scores = Q K^T$
    (computed via `torch.matmul(queries, keys.T)`)
    (shape: $N_Q \times N_K$)

2.  **Attention Weights**:
    $AttentionWeights = \text{softmax}(Scores)$
    (computed via `F.softmax(scores, dim=-1)`)
    (shape: $N_Q \times N_K$)

3.  **Context Vector**:
    $Context = AttentionWeights V$
    (computed via `torch.matmul(attention_weights, values)`)
    (shape: $N_Q \times D_V$)

## Code Walkthrough

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define queries, keys and values as PyTorch tensors
# Queries: What we are looking for (e.g., current word in decoder)
# Keys: What is available to be matched (e.g., all words in encoder output)
# Values: The information associated with the keys
queries = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
keys = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
values = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

# Compute attention scores using matrix multiplication
# scores[i, j] represents the relevance of key[j] to query[i]
scores = torch.matmul(queries, keys.T)
# scores will be:
# tensor([[2., 1., 1.],
#         [1., 1., 2.]])

# Apply softmax to normalize scores
# softmax(dim=-1) ensures that the sum of weights for each query (row) is 1.
attention_weights = F.softmax(scores, dim=-1)
# Example attention_weights (approximate after softmax):
# tensor([[0.7071, 0.1464, 0.1464],
#         [0.1464, 0.1464, 0.7071]])

# Compute weighted sum of values
# This combines the information from 'values' according to the 'attention_weights'.
# context[i] is the aggregated context vector for query[i].
context = torch.matmul(attention_weights, values)

print("Attention Weights: \n", attention_weights)
print("Context Vector:\n", context)

# Visualize attention weights as a heatmap
# Each row represents a query, and each column represents a key.
# The intensity of the color shows the weight assigned.
plt.matshow(attention_weights.numpy()) # Convert to NumPy for matplotlib
plt.colorbar()
plt.title("Attention Weights")
plt.xlabel("Keys")
plt.ylabel("Queries")
plt.show()