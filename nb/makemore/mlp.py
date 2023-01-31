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

    
def init_model(hl_size: int = 200, lookup_size: int = 10, context_length: int = 3):
    g = torch.Generator().manual_seed(SEED)
    C = torch.randn((27, lookup_size), generator=g)
    
    W1 = torch.randn((lookup_size*context_length, hl_size), generator=g) * 0.1 # to avoid tanh from outputting ~1.0
    b1 = torch.randn(hl_size, generator=g) * 0.01                              # this could cause grad to be ~0, resulting in dead neurons
    
    W2 = torch.randn((hl_size, 27), generator=g) * 0.01 # so that first step is not too far away
    b2 = torch.randn(27, generator=g) * 0.0 # zero bias on initialization
    
    model_params = [C, W1, b1, W2, b2]
    model_hyperparams = [hl_size, lookup_size, context_length]
    
    for p in model_params:
        p.requires_grad = True 
    
    return (model_params, model_hyperparams)


def init_uniform_model(hl_size: int = 200, lookup_size: int = 10, context_length: int = 3):
    C = torch.ones((27, lookup_size), requires_grad=True)
    
    W1 = torch.ones((lookup_size*context_length, hl_size), requires_grad=True)
    b1 = torch.ones(hl_size, requires_grad=True)
    
    W2 = torch.ones((hl_size, 27), requires_grad=True)
    b2 = torch.ones(27, requires_grad=True)
    
    model_params = [C, W1, b1, W2, b2]
    model_hyperparams = [hl_size, lookup_size, context_length]
    
    return (model_params, model_hyperparams)



def train_model(data: tuple, model, steps: int = 1, batch_size: int = 32):
    x, y = data
    
    model_params, model_hyperparams = model
    C, W1, b1, W2, b2 = model_params
    hl_size, lookup_size, context_length = model_hyperparams
    
    for i in range(steps):
        # get random batch
        ix = torch.randint(0, x.shape[0], (batch_size,))
        
        # forward pass
        embs = C[x[ix]] # [32, 3, 2]
        h = torch.tanh(embs.view(-1, lookup_size*context_length) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, y[ix])
        
        # backward pass
        for p in model_params:
            p.grad = None
        loss.backward()
        
        # update
        lr = 0.1 if i <20000 else 0.01 # decay
        for p in model_params:
            p.data += -lr * p.grad
            
        # learning stats
        stepi.append(i)
        lossi.append(loss.log10().item()) # neg log-
        
    print(f"{loss=}")
    plt.plot(stepi, lossi)


def generate_samples(model, num_samples: int = 10, block_size: int = 3):
    g = torch.Generator().manual_seed(SEED)
    
    model_params, _ = model
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
            
@torch.no_grad()
def validate_model(model, data: tuple):
    
    model_params, model_hyperparams = model
    C, W1, b1, W2, b2 = model_params
    hl_size, lookup_size, context_length = model_hyperparams
    
    x, y = data

    emb = C[x]
    h = torch.tanh(emb.view(-1, lookup_size*context_length) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    
    print(f"{loss=}")
        

# %%
build_dataset(words)
xtr, ytr, xdev, ydev, xts, yts = split_dataset()

# %%
model = init_model(hl_size=500, lookup_size=100)
# model = init_uniform_model(hl_size=500, lookup_size=100)

lri = []
lossi = []
stepi = []

#%%
train_model(
    data=(xtr, ytr),
    model=model, 
    steps=15000,
    batch_size=64
    )
# %%
generate_samples(model)
# %%
datatr = (xtr, ytr)
datadev = (xdev, ydev)
datats = (xts, yts)

validate_model(model, datatr)
validate_model(model, datadev)
validate_model(model, datats)

# %%
"""
Outputs:

farristepha.
jene.
sten.
malah.
kyn.
jeffie.
kahlanita.
namdira.
ged.
amaud.
"""