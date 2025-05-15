# MicroGPT
MikroGPT

## Select Device Cuda/GPU
- when we run the code on a local mashine we can select now select device it is running on
- the data we move:
```
x, y = x.to(device), y.to(device) 
```
...when we load the data we make sure to load it on the device
```
m = model.to(device)
```
...when we create a model we want to make sure to move the parameters to the device

For example the model contains the Embedding that contains a .weight which stores the lookup table, which will be moved to the device.
```
context = torch.zeros((1, 1), dtype=torch.long, device=device) # (B=1, T=1)

```
...also if we are now creating the context that feeds into generate we also have to the device

## Introducing eval_iters

We use eval_iters in the training loop. Before we printed the Loss in the training loop. But this measuring could be very noisy as some batches are more or less lucky. Now we use an estimate_loss() function that avera ges the loss over multiple batches.

## Introducing modes for the model

```
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
...switching between modes becomes important when we use Batch_Normalization (shifting between local and global mean/std)


...and remember, everything that happens after '@torch.no_grad' will not be include in the backward pass (no gradients get calculated)


