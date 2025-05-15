import torch
import torch.nn as nn
from torch.nn import functional as F

import wget

#hyperparameters

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # number of embedding dimension = channels
# -----------------
 
torch.manual_seed(1337)

# wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping form characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, ouput a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype= torch.long)
#print(data.shape)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # this is new and interestin, as i now am working on local devices -> working with cuda (nvidia) would now be possible
    x, y = x.to(device), y.to(device) 
    return x, y

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

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head -> language model head
    
    def forward(self, idx, targets=None):
    
        # idx and targets are both (B, T) tensor of integers
        # logits should have a normal distribution
        tok_emb = self.token_embedding_table(idx) # (B, T, C) = (batch = batch_size, time = block_size, channel = vocab_size)
        logits = self.lm_head(tok_emb) # (B, T, vocab_size) 
        

        if targets == None:
            loss = None
        else:
            # the right dimension should have a very high number

            # this doesn't work -> look into pytorch library
            #loss = F.cross_entropy(logits, targets)

            # dimension are (B,T,..) but we want them to be (B*T,...) to be in the right cross_entropy shape
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # generate form the model
    # idx: context of characters in some batch
    # the job generate: take idx (B, T) and generate (B, T+1), (B, T+1), .... (B, T+ max_new_tokens )
    def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx) # the loss is gonna be ignored as we are not sending any targets
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C), take emission probabilites of the last tokens in the sample sequence
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = 1) # (B, C)
            # sample from the distribution/ probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # (B=1, T=1)
print(decode(model.generate(context, 500)[0].tolist())) # generate 1000 tokens
