# 🧠 MicroGPT

A minimal character-level Transformer model implemented from scratch, inspired by nanoGPT. This notebook demonstrates the core principles behind training language models using PyTorch, with a focus on clean, readable code and educational clarity.

---

## ⚙️ Device Selection (CPU/GPU)

When running the model locally, it's essential to ensure that both the model and data are moved to the appropriate device (e.g. GPU if available):

```python
x, y = x.to(device), y.to(device)  # Move input and labels to device
model = model.to(device)           # Move model parameters to device
```

PyTorch modules like `nn.Embedding` store tensors such as `.weight`, which also need to be transferred to the device:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # (B=1, T=1)
```

➡️ Always create new tensors directly on the device to avoid errors.

---

## 📊 Introducing `eval_iters` – More Stable Loss Estimates

In early stages, we printed the loss directly during training. However, this can be **noisy** since batch-to-batch variation is high.

We now introduce a `estimate_loss()` function that averages loss over multiple batches for both training and validation:

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

📌 Note:

* `model.eval()` switches to evaluation mode (important for layers like BatchNorm or Dropout).
* `model.train()` re-enables training behavior.
* `@torch.no_grad()` disables gradient tracking, reducing memory usage and speeding up evaluation.

---

## 🔁 Why Switching Modes Matters

Certain layers like **BatchNorm** behave differently during training vs. evaluation. Specifically:

* **Training mode**: Uses batch statistics.
* **Evaluation mode**: Uses running averages (global stats).

Switching modes correctly ensures consistent behavior and prevents bugs.

---

## 🧠 Reminder: No Gradients During Evaluation

Everything inside `@torch.no_grad()` is **excluded from the computation graph**. This means:

* No gradients are calculated.
* No `.backward()` is called.
* It's safe and efficient for validation.

---

Let me know if you'd like to add contact links or a project summary at the top!



