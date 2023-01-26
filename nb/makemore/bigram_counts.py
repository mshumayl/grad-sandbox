# %%
import torch
import matplotlib.pyplot as plt

# %%
with open(r"../../data/names.txt") as f:
    words = f.read().lower().splitlines()

# %%
"""
Implement bigram statistical model by getting probability distribution via counts
"""
N = torch.zeros((27,27), dtype=torch.int32)

# %%
chars = sorted(list(set(''.join(words)))) #get all chars
stoi = {s:i+1 for i, s in enumerate(chars)} #get integer mapping of chars
stoi['.'] = 0 
itos = {i:s for s, i in stoi.items()} #reverse mapping -- from int to char

# %%
b = {}
for w in words: 
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]): #shift by one to get second char of bigram
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
        
# %%

# Visualize counts
plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

# %%

# Get probability distribution
P = (N+1).float() # +1 smoothing
P /= P.sum(1, keepdims=True) #keepdims ensures no spooky broadcasting bug

# %%

g = torch.Generator().manual_seed(9876543210) #seed for consistency

# Make predictions using prob. dist.
for i in range(5):
  out = []
  ix = 0
  while True:
    prob = P[ix] # get prob. dist of row ix (i.e. prob. dist. of char2)

    # sample an int from the prob. dist.
    ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0: # if index of stopchar, stop
      break
  print(''.join(out))
  
# %%

# Evaluate model using log likelihood
log_likelihood = 0.0
n = 0

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2] #index the matrix to get prob of that specific bigram
    logprob = torch.log(prob) 
    log_likelihood += logprob # ll is accumulative
    n += 1
    print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood #to make more similar to loss function
print(f'{nll=}')
print(f'{nll/n}')