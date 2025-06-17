# 🧠 MicroGPT – Minimal GPT Training from Scratch

This repository demonstrates how to train a tiny GPT-style language model on any text dataset, following the minimal and educational approach by [Andrej Karpathy](https://github.com/karpathy). The focus is on understanding and experimenting with the core ideas behind transformer-based models.

---

## 📜 1. Select Your Training Text

Download and load your dataset:

```python
wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

> ✅ **Tip:**
> You can replace `input.txt` with any text corpus (e.g., all works of Shakespeare, a book, or your own data).
> Larger datasets lead to significantly better model performance.

---

## ⚙️ 2. Use GPU if Available

Let PyTorch automatically choose the best hardware:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

If your system has an NVIDIA GPU with CUDA support, tensors and models will be moved to GPU memory. This enables massively parallel operations and faster training.

---

## 🔧 3. Set Hyperparameters

| Variable        | Karpathy (A100 GPU) | Your Setup (e.g. M1 Pro) | Description                                                        |
| --------------- | ------------------- | ------------------------ | ------------------------------------------------------------------ |
| `batch_size`    | 64                  | 8                        | Number of sequences processed in parallel during training          |
| `block_size`    | 256                 | 12                       | Maximum context length (how far back the model looks)              |
| `max_iters`     | —                   | 3000                     | Total number of training iterations                                |
| `eval_interval` | 500                 | 100                      | Frequency (in iterations) to evaluate training and validation loss |
| `learning_rate` | 3e-4                | 0.01                     | Optimization step size; affects training stability                 |
| `device`        | 'cuda'              | auto                     | Hardware used for training (GPU/CPU)                               |
| `eval_iters`    | 200                 | 100                      | Number of batches used to estimate average loss during evaluation  |
| `n_embd`        | 384                 | 10                       | Embedding dimension (model width)                                  |
| `n_head`        | 6                   | 2                        | Number of attention heads in the Transformer                       |
| `n_layer`       | 6                   | 2                        | Number of Transformer blocks (depth of the model)                  |
| `dropout`       | 0.2                 | 0.2                      | Dropout probability for regularization                             |

> ⚠️ **Note on Scaling:**
> Karpathy runs his model on an NVIDIA A100 — an enterprise-grade GPU with vast compute power. This allows him to dramatically scale up the model:

* **Larger batches** (`batch_size = 64`) process more data in parallel.
* **Longer context** (`block_size = 256`) allows the model to attend to more previous tokens when predicting the next one.
* **More attention heads** and **more layers** enable deeper, more expressive networks that capture complex token dependencies.

These enhancements contribute to better generalization and text generation quality — but they require significantly more memory and compute.

---

## ▶️ 4. Run the Model

Install required packages in your virtual environment:

```bash
pip install torch wget
```

| Module in Code                             | Install via `pip` |
| ------------------------------------------ | ----------------- |
| `torch`, `torch.nn`, `torch.nn.functional` | `torch`           |
| `wget`                                     | `wget`            |

Then run the training script:

```bash
python bigram.py
```

During training, the model will periodically print training and validation loss. Once training is complete, you can sample text from the model:

```python
print(decode(model.generate(context, 500)[0].tolist()))
```

This generates 500 characters of model output based on your training data.

---

## 💡 Summary

MicroGPT is a great learning project to:

* Understand transformer models from scratch
* Explore token embeddings, attention, and autoregressive generation
* Experiment with architecture and hyperparameters depending on your hardware

---

**Happy building! 🚀**
Feel free to extend the model with attention masks, positional encodings, or deeper networks as your next step.

---





Here’s a unified, polished README for **MicroGPT**, aligned with your bio/landing style and including a clean structure and tone:

---

# MicroGPT – Minimal GPT Training from Scratch 🧠✨

A minimal, educational implementation of a GPT-style language model based on [Andrej Karpathy’s approach](https://github.com/karpathy). This project lets you train a tiny transformer from scratch on any text dataset to explore core ideas behind modern language models.

---

## 💡 About This Project

MicroGPT demystifies transformer models by building one step-by-step and training it on character-level text data. It’s perfect for learning fundamentals like embeddings, attention, autoregressive generation, and the impact of hyperparameters — all while experimenting on your own data.

---

## 🚀 Getting Started

### 1. Download Your Training Text

Grab a text corpus (e.g., Shakespeare or your own data):

```python
wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

*Tip:* Larger datasets yield better results but require more compute.

---

### 2. Enable GPU Training (If Available)

PyTorch will automatically use CUDA-enabled GPUs for faster training:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

### 3. Set Hyperparameters

| Parameter       | Karpathy (A100 GPU) | Example Setup (M1 Pro) | Description                              |
| --------------- | ------------------- | ---------------------- | ---------------------------------------- |
| `batch_size`    | 64                  | 8                      | Sequences per training batch             |
| `block_size`    | 256                 | 12                     | Context length (tokens model attends to) |
| `max_iters`     | —                   | 3000                   | Number of training iterations            |
| `eval_interval` | 500                 | 100                    | Steps between validation evaluations     |
| `learning_rate` | 3e-4                | 0.01                   | Optimizer step size                      |
| `device`        | 'cuda'              | auto                   | Training hardware (GPU/CPU)              |
| `eval_iters`    | 200                 | 100                    | Batches used to estimate average loss    |
| `n_embd`        | 384                 | 10                     | Embedding dimension                      |
| `n_head`        | 6                   | 2                      | Attention heads                          |
| `n_layer`       | 6                   | 2                      | Transformer layers                       |
| `dropout`       | 0.2                 | 0.2                    | Dropout rate                             |

*Note:* Karpathy’s large-scale model leverages powerful GPUs, enabling deeper and wider networks for better text generation.

---

### 4. Install Dependencies and Run Training

```bash
pip install torch wget
python bigram.py
```

Training progress will print loss metrics. After training, generate sample text like so:

```python
print(decode(model.generate(context, 500)[0].tolist()))
```

---

## 🎯 What You’ll Learn

* Fundamentals of transformer architectures
* Token embeddings and attention mechanisms
* Autoregressive text generation
* Impact of hyperparameters on training and output quality

---

## 🔜 Next Steps

Consider extending this minimal GPT by adding:

* Attention masking
* Positional encodings
* More layers or heads
* Larger datasets

---

**Happy learning and building! 🚀**

---

If you want me to unify or polish other README files in this style, just share them!

