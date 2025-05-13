import torch
import torch.nn as nn
from torch.nn import functional as F

import wget

#hyperparameters

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -----------------
 
torch.manual_seed(1337)

# wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

