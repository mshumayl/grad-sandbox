#%%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# %%

with open(r"../../data/names.txt") as f:
    words = f.read().lower().splitlines()


SEED = 987654321
# %%
chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# %%
block_size = 3 # context size

def build_dataset(words):
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size
            
        for ch in w + '.': # only stop .
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            
            context = context[1:] + [ix] # moving char window
                                         # starts with <.> <.> <.> <char>
            
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    
    return X, Y

# %%

def split_dataset():
    random.seed(SEED)
    random.shuffle(words)

    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))
    
    xtr, ytr = build_dataset(words[:n1])
    xdev, ydev = build_dataset(words[n1:n2])
    xts, yts = build_dataset(words[n2:])
    
    return xtr, ytr, xdev, ydev, xts, yts

    
# %%
def init_model():
    g = torch.Generator().manual_seed(SEED)
    C = torch.randn((27, 10), generator=g, requires_grad=True)
    
    W1 = torch.randn((30, 200), generator=g, requires_grad=True)  
    b1 = torch.randn(200, generator=g, requires_grad=True)        
    
    W2 = torch.randn((200, 27), generator=g, requires_grad=True)  
    b2 = torch.randn(27, generator=g, requires_grad=True)
    
    parameters = [C, W1, b1, W2, b2]
    
    return parameters

# %%
def train_model(data: tuple, parameters: list, steps: int = 1, batch_size: int = 32):
    x, y = data
    C, W1, b1, W2, b2 = parameters
    
    
    for i in range(steps):
        # get random batch
        ix = torch.randint(0, x.shape[0], (batch_size,)) # default batch size 32
        
        # forward pass
        embs = C[x[ix]] # [32, 3, 2]
        h = torch.tanh(embs.view(-1, 30) @ W1 + b1) # [32, 100]
        logits = h @ W2 + b2 # [32, 27]
        loss = F.cross_entropy(logits, y[ix])
        
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # update
        lr = 0.1 if i <100000 else 0.01 # decay
        for p in parameters:
            p.data += -lr * p.grad
            
        # learning stats
        stepi.append(i)
        lossi.append(loss.log10().item()) # neg log-
        
    print(f"{loss=}")
    plt.plot(stepi, lossi)
            
# %%
build_dataset(words)
xtr, ytr, xdev, ydev, xts, yts = split_dataset()

# %%
model = init_model()
lri = []
lossi = []
stepi = []

#%%
train_model(
    data=(xtr, ytr),
    parameters=model, 
    steps=100000,
    batch_size=32
    )
# %%
