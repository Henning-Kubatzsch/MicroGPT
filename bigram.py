import torch
import torch.nn as nn
from torch.nn import functional as F

import wget

#hyperparameters

batch_size = 8  # 64 
block_size =  12 # 256 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 100 # 500
learning_rate = 0.01  # 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100 # 200
n_embd = 10 # 384 number of embedding dimension = channels
n_head  = 2 # 6
n_layer = 2 # 6
dropout = 0.2
# -----------------
 
torch.manual_seed(1337)

wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

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
    # this is new and interesting, as i now am working on local devices -> working with cuda (nvidia) would now be possible
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

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # in sort of pytorch naming conventions tril is called a buffer not a parameter -> assign it to the model via register_buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ('affinities')
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C **-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # performan the weighted aggregatoin of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
 

class MultiHeadAttention(nn.Module):
    """" multi-heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) 
        self.dropout = nn.Dropout(dropout) # dropout is a regularization technique to prevent overfitting
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.dropout(self.proj(out)) 
        return  out # we are concatenating over the channel dimension

class FeedForward(nn.Module):
    """" a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: comminication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # Normalize the features and make them unit gaussian at initialization
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # fork off, do some communication (self-attention), then come back
        x = x + self.ffwd(self.ln2(x))    # fork off, do some computation (feed-forward), then come back

        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        #self.blocks = nn.Sequential(
        #    Block(n_embd, n_head=4,),
        #    Block(n_embd, n_head=4,),
        #    Block(n_embd, n_head=4,),
        #    nn.LayerNorm(n_embd)
        #)
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head -> language model head
    
    def forward(self, idx, targets=None):
        B, T = idx.shape    
        # idx and targets are both (B, T) tensor of integers
        # logits should have a normal distribution
        tok_emb = self.token_embedding_table(idx) # (B, T, C) = (batch = batch_size, time = block_size, channel = vocab_size) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C), integers form 0 to T-1
         # encoding the position of the token in the sequence
        x = tok_emb + pos_emb # (B, T, C) + (T, C) -> broadcating (B, T, C) + (B, T, C)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size) 
        

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
            # crop idx to be the last bloxk_size tikens
            idx_cond = idx[:, -block_size:] 
            # get the predictions
            logits, loss = self(idx_cond) # the loss is gonna be ignored as we are not sending any targets
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
