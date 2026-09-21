import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Read input sentence
with open("dataset/sample_sentence.txt", "r", encoding="utf-8") as file:
    text = file.read().strip()

words = text.split()

print("Input Sentence:")
print(text)

print("\nWords:")
print(words)

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(words)

print("\nEmbedding Shape:")
print(embeddings.shape)

# Normalize embeddings
normalized_embeddings = embeddings / (
    np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
)

# Calculate attention scores
scores = np.dot(
    normalized_embeddings,
    normalized_embeddings.T
)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

attention_weights = np.array([
    softmax(row) for row in scores
])

print("\nAttention Weights Shape:")
print(attention_weights.shape)

# Attention output
attention_output = np.dot(
    attention_weights,
    embeddings
)

print("\nAttention Output Shape:")
print(attention_output.shape)

# Save embeddings
np.save("dataset/embeddings.npy", embeddings)

# Save attention weights
attention_df = pd.DataFrame(
    attention_weights,
    index=words,
    columns=words
)

attention_df.to_csv("dataset/attention_weights.csv")

# Save attention output
output_df = pd.DataFrame(attention_output)

output_df.to_csv(
    "dataset/attention_output.csv",
    index=False
)

# Create heatmap
plt.figure(figsize=(12, 8))

plt.imshow(
    attention_weights,
    aspect="auto"
)

plt.xticks(
    range(len(words)),
    words,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(words)),
    words
)

plt.xlabel("Key Words")
plt.ylabel("Query Words")
plt.title("Attention Mechanism Heatmap")

plt.colorbar(label="Attention Weight")

plt.tight_layout()

plt.savefig(
    "dataset/attention_heatmap.png",
    dpi=300
)

plt.close()

print("\nProject completed successfully!")