# Rumi-Inspired Mini GPT

A small decoder-only Transformer language model built from scratch using **Python and PyTorch**, designed to generate English poetry inspired by the themes and language of Rumi's poetry.

This project is an educational exploration of how GPT-style language models work internally, from tokenization and self-attention to training and text generation.

## Project Overview

The model is trained from randomly initialized weights on an English versified translation of Rumi's *Masnavi*, Book I.

The objective is to understand and implement the key components of a GPT architecture rather than use a pretrained language model or an external AI API.

**Current project status:** Experimental prototype. The model can generate text, but its output is not yet consistently grammatical or coherent.

## Architecture

This project implements a decoder-only Transformer with:

- Character-level tokenization
- Token embeddings
- Causal multi-head self-attention
- Rotary Positional Embeddings (RoPE)
- Feed-forward neural networks
- Residual connections
- Layer normalization
- Cross-entropy loss
- AdamW optimization
- Autoregressive text generation

### Model Configuration

| Component | Configuration |
|---|---:|
| Architecture | GPT-style decoder-only |
| Transformer layers | 2 |
| Attention heads | 4 |
| Embedding dimension | 64 |
| Head dimension | 16 |
| Feed-forward dimension | 256 |
| Context length | 128 characters |
| Batch size | 16 |
| Optimizer | AdamW |
| Positional encoding | RoPE |
| Framework | PyTorch |
| Hardware | CPU |

### Parameter Count

The model uses weight tying between the token embedding and language-model output layer.

Its trainable parameter count is:

\[
P = 100,096 + 64V
\]

where \(V\) is the number of unique characters in the vocabulary.

The exact count depends on the processed dataset.

## Dataset

**Source:** [The Mesnevi — Project Gutenberg, eBook #61724](https://www.gutenberg.org/ebooks/61724)

- Original poet: Jalaluddin Rumi
- Translator: Sir James W. Redhouse
- Translation published: 1881
- Language: English
- Material used: Poetry from Book I

The project uses a historical English versification of Rumi's Persian poetry. The model therefore learns patterns from this English translation, not from original English writings by Rumi.

The source is identified as public domain in the United States by Project Gutenberg. Users outside the United States should check applicable local copyright laws.

The source text and extracted training corpus are not included in this repository. Follow the dataset preparation instructions to obtain and process the source.

## Project Structure

```text
rumi-inspired-mini-gpt/
│
├── prepare_rumi.py
├── tokenizer.py
├── batch_generator.py
├── attention.py
├── model.py
├── train.py
├── generate.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

The project was developed on Windows 11 using an HP EliteBook 840 G5 with 16 GB RAM and CPU-only processing.

### 1. Clone the repository

```bash
git clone https://github.com/talhasaleem24/rumi-inspired-mini-gpt.git

cd rumi-inspired-mini-gpt
```

### 2. Create a virtual environment

```bash
python -m venv .venv

.venv\Scripts\activate
```

### 3. Install PyTorch

For CPU-only installation:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the plain-text source from Project Gutenberg:

[Download The Mesnevi](https://www.gutenberg.org/cache/epub/61724/pg61724.txt)

Save the file as `source.txt` in the project directory.

Run:

```bash
python prepare_rumi.py
```

This creates:

```text
rumi_english.txt
```

Then create the character vocabulary:

```bash
python tokenizer.py
```

This creates:

```text
vocab.json
```

## Training

Train the model on your CPU:

```bash
python train.py
```

The training script uses:

- Forward propagation
- Cross-entropy loss
- Backpropagation
- Gradient clipping
- AdamW optimization
- Training and validation loss evaluation
- Model checkpoint saving

The default experiment uses 1,000 training iterations. Results depend on the training data and hardware.

## Text Generation

After training, run:

```bash
python generate.py
```

Enter an English prompt to generate text character by character.

Example prompts:

```text
The heart

Love is

The soul is
```

Generation uses temperature and top-k sampling to control the next-character selection.

## Limitations

This is a small educational language model, not a production-grade conversational AI.

Current limitations include:

- Limited training data
- Character-level tokenization
- Small model capacity
- Limited context length
- Inconsistent grammar and spelling
- Occasional repetition and incoherent output

The generated text should be considered experimental poetry inspired by patterns in the source translation, not authentic poetry written by Rumi.

## Learning Objectives

Through this project, I am exploring:

1. Neural networks and embedding layers
2. Linear algebra and matrix operations
3. Query, Key and Value representations
4. Scaled dot-product attention
5. Causal masking
6. Multi-head self-attention
7. Rotary Positional Embeddings
8. Transformer decoder blocks
9. Backpropagation and gradient descent
10. Language-model training and text generation

## Future Improvements

- Improve dataset preparation and quality
- Experiment with larger context lengths
- Compare character-level and subword tokenization
- Experiment with additional licensed English translations
- Evaluate different model sizes
- Compare RoPE with learned positional embeddings
- Improve generation quality and reduce repetition
- Visualize training and validation loss

## Acknowledgements

- Jalaluddin Rumi, for the original poetic work.
- Sir James W. Redhouse, for the historical English versification.
- Project Gutenberg, for making the historical source available.
- PyTorch, for the deep learning framework.

---

**Built as a hands-on Generative AI learning project to understand and implement a GPT-style Transformer from scratch.**