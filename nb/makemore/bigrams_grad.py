# %%
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# %%
with open(r"../../data/names.txt") as f:
    words = f.read().lower().splitlines()

# %%
"""
Implement bigrams model through gradient optimization. 
"""

# %%
chars = sorted(list(set(''.join(words)))) # Get all chars
stoi = {s:i+1 for i, s in enumerate(chars)} # Get integer mapping of chars
stoi['.'] = 0 
itos = {i:s for s, i in stoi.items()} # Reverse mapping -- from int to char

# %%
xs, ys = [], []

b = {}
for w in words[:1000]: 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): # Shift by one to get second char of bigram
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement() # Number of elements in xs
        
# %%

# Initialize weights randomly
# Weights need to be a 2D matrix of size (27, 27) 
#   corresponding to each alphabet combination in the bigram

g = torch.Generator().manual_seed(9876543210)
W = torch.randn((27, 27), generator=g, requires_grad=True) # Keep grads


# %%
# Gradient optimization
for epoch in range(500):
    
    # Forward pass
    xenc = F.one_hot(xs, num_classes=27).float()    # Cast int to float
    logits = xenc @ W   # Matmul between encoded input and random weight
                        # Logits are log of 'counts'
    # Softmax
    counts = logits.exp() # equivalent to 'counts' in statistical bigram
    probs = counts / counts.sum(1, keepdims=True) # Probs for next char
                                                  # Probs.shape = torch.Size([[N, 27]]), N=number of inputs
                                                  # Hence, probs[1] is getting the prob. dist. of chars that come after 'A'
                                                  #     and probs[27] is getting the prob. dist. of chars that come after 'Z'
    
    # Calculate loss for the whole matrix at this epoch
    # torch.arange(num) is a vector containing the indices of all xs, equivalent to the number of inputs
    # ys is the indices of the actual second chars
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print(f"{epoch=},{loss.item()=}")
    
    # Backward pass
    W.grad = None # Set all grad to zero at each epoch
    loss.backward()
    
    # Update
    W.data += -30 * W.grad # -ve to reduce nll
    
# %%
# Sample from the trained model

for i in range(10):
    out = []
    ix = 0
    
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W 
        counts = logits.exp()
        p = counts/counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        
        out.append(itos[ix])
        
        if ix==0:
            break
        
    print(''.join(out))

# %%
