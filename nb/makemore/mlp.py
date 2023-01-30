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
def build_dataset(words, block_size: int = 3):
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


def split_dataset():
    random.seed(SEED)
    random.shuffle(words)

    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))
    
    xtr, ytr = build_dataset(words[:n1])
    xdev, ydev = build_dataset(words[n1:n2])
    xts, yts = build_dataset(words[n2:])
    
    return xtr, ytr, xdev, ydev, xts, yts

    
def init_model(hl_size: int = 200, lookup_size: int = 30):
    g = torch.Generator().manual_seed(SEED)
    C = torch.randn((27, 10), generator=g, requires_grad=True)
    
    W1 = torch.randn((lookup_size, hl_size), generator=g, requires_grad=True)  
    b1 = torch.randn(hl_size, generator=g, requires_grad=True)        
    
    W2 = torch.randn((hl_size, 27), generator=g, requires_grad=True)  
    b2 = torch.randn(27, generator=g, requires_grad=True)
    
    model_params = [C, W1, b1, W2, b2]
    
    return model_params


def train_model(data: tuple, model_params: list, steps: int = 1, batch_size: int = 32):
    x, y = data
    C, W1, b1, W2, b2 = model_params
    
    for i in range(steps):
        # get random batch
        ix = torch.randint(0, x.shape[0], (batch_size,)) # default batch size 32
        
        # forward pass
        embs = C[x[ix]] # [32, 3, 2]
        h = torch.tanh(embs.view(-1, 30) @ W1 + b1) # [32, 100]
        logits = h @ W2 + b2 # [32, 27]
        loss = F.cross_entropy(logits, y[ix])
        
        # backward pass
        for p in model_params:
            p.grad = None
        loss.backward()
        
        # update
        lr = 0.1 if i <100000 else 0.01 # decay
        for p in model_params:
            p.data += -lr * p.grad
            
        # learning stats
        stepi.append(i)
        lossi.append(loss.log10().item()) # neg log-
        
    print(f"{loss=}")
    plt.plot(stepi, lossi)


def generate_samples(model_params: list, num_samples: int = 10, block_size: int = 3):
    g = torch.Generator().manual_seed(SEED)
    C, W1, b1, W2, b2 = model_params
    
    for _ in range(num_samples):
        out = []
        context = [0] * block_size
        
        while True:
            emb = C[torch.tensor([context])] # [1, block_size, d]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            
            context = context[1:] + [ix]
            
            out.append(ix)
            
            if ix==0:
                break
            
        print(''.join(itos[i] for i in out))
            

def validate_model(model_params: list, data: tuple):
    
    C, W1, b1, W2, b2 = model_params
    x, y = data

    emb = C[x]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    
    print(f"{loss=}")
        

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
    model_params=model, 
    steps=100000,
    batch_size=128
    )
# %%
generate_samples(model)
# %%
datatr = (xtr, ytr)
datadev = (xdev, ydev)
datats = (xts, yts)

validate_model(model, datatr)
validate_model(model, datadev)
# validate_model(model, datats)

# %%
