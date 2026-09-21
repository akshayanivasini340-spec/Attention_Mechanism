# Attention Mechanism

A Python project demonstrating the working of an Attention Mechanism using sentence embeddings, attention weights, and an attention heatmap.

## Project Overview

This project demonstrates how an attention mechanism can identify relationships between words in a sentence.

The project:

- Reads a sample sentence from the dataset
- Generates word embeddings
- Calculates attention weights
- Produces attention output
- Saves the results as CSV and NumPy files
- Generates an attention heatmap

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Sentence Transformers

## Project Structure

```text
Attention_Mechanism/
│
├── attention.py
├── requirements.txt
├── README.md
│
└── dataset/
    ├── sample_sentence.txt
    ├── embeddings.npy
    ├── attention_weights.csv
    ├── attention_output.csv
    └── attention_heatmap.png
