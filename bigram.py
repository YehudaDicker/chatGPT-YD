import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters 
batch_size = 32 # how many independent sequences we will process in parallel 
block_size = 8 # context length for prediction 
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# --------------

torch.manual_seed(1337)

# download tiny shakespeare dataset to train on
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# load data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # random offset to grab chunk from data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # context from chunk
    x = torch.stack([data[i:i+block_size] for i in ix])
    # target from chunk
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # 
    x,y = x.to(device), y.to(device)
    return x, y


@torch.no_grad() # tells pytorch we're never calling .backward(), so more efficient
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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # self.token_embedding_table(idx) means every integer in input to transformer will refer to embedding table made in __init__ 
        # and will pluck out row in embedding table corresponding to index
        # we pluck out all rows, arrange in B x T x C
        # logits is score for next character in sequence based on individual identity of token in sequence
        logits = self.token_embedding_table(idx) # (B,T,C (C is channel which is vocab size))
        
        if targets is None:
            loss = None
        else:
            # reshape arrays to 2d and 1d to conform to cross_entropy docs
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # loss function measures quality of predictions made by logits
            # measures quality of logits with respect to targets
            # correct dimension of logits should be very high number
            # we're expecting loss of -ln(1/65)  = 4.17...
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions of current indexes
            logits, loss = self(idx)
            # focus only on the last time step cuz those predict what comes next
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
#logits, loss = m(xb, yb)
#print("logits shape::", logits.shape)
#print("loss:", loss)

# create 1x1 tensor which holds a 0, datatype int
# 0 stand for new line char so reasonable start
# 0 kicks off generation 
# ask for 100 tokens
# generate works on level of batches, so we index into 0th row to unpluck single batch dimension which gives of 1d array of indexes
# convert array to list to feed into decode 

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# train loop
for iter in range(max_iters): # increase number of steps for good results... 
    
    # every once in a while eval the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, cal loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    # zero out previous gradients
    optimizer.zero_grad(set_to_none=True)
    # get gradients for all parameters
    loss.backward()
    # use gradients to update parameters
    optimizer.step()

# generate from the model
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))

