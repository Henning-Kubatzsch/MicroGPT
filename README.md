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

