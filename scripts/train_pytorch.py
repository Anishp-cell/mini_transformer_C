import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time

# ==========================================
# 1. Config exactly matching model_config.h
# ==========================================
vocab_size = 65
embed_dim = 128
seq_len = 64
ff_dim = 512
num_layers = 3

# Pytorch specific training parameters
batch_size = 32
learning_rate = 1e-3
train_steps = 3000  # Will drop loss to ~1.0 rapidly

# Force CPU because RTX 50-series (Blackwell) requires PyTorch Nightly builds.
# PyTorch's CPU math is still heavily multi-threaded and uses Intel MKL, 
# so it will still be immensely faster than pure single-threaded C!
device = 'cpu'
print(f"[{device.upper()}] Initializing PyTorch optimized engine for i7...")

# ==========================================
# 2. PyTorch Architecture Mapping
# ==========================================
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, embed_dim)
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # In C, we do simple Q, K, V with no separate output projection
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        wei = q @ k.transpose(-2, -1) / math.sqrt(C)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T) == 0
        wei = wei.masked_fill(mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        return wei @ v

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = LayerNorm()
        self.attn = Attention()
        self.ln2 = LayerNorm()
        self.ff = FeedForward()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.final_ln = LayerNorm()
        self.out_w = nn.Linear(embed_dim, vocab_size)
    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_ln(x)
        logits = self.out_w(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ==========================================
# 3. Model Weight Exporter (C binary compat)
# ==========================================
def save_model_to_c(model, path):
    print(f"Exporting PyTorch weights exactly to C struct binary logic > {path}")
    model.eval()
    with open(path, "wb") as f:
        # Tensors need to exactly match C memory layout => flatten and convert to float32
        f.write(model.token_emb.weight.data.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.pos_emb.weight.data.cpu().numpy().astype(np.float32).tobytes())
        
        for block in model.blocks:
            # Linear weights in PyTorch are [out_features, in_features]
            # In C they are [in_features, out_features]. So we must transpose (.T)
            f.write(block.attn.wq.weight.data.T.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.attn.wk.weight.data.T.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.attn.wv.weight.data.T.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ln1.gamma.data.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ln1.beta.data.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ff.w1.weight.data.T.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ff.w1.bias.data.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ff.w2.weight.data.T.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ff.w2.bias.data.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ln2.gamma.data.cpu().numpy().astype(np.float32).tobytes())
            f.write(block.ln2.beta.data.cpu().numpy().astype(np.float32).tobytes())
            
        f.write(model.final_ln.gamma.data.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.final_ln.beta.data.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.out_w.weight.data.T.cpu().numpy().astype(np.float32).tobytes())
        f.write(model.out_w.bias.data.cpu().numpy().astype(np.float32).tobytes())

# ==========================================
# 4. Training Loop setup
# ==========================================
data_path = "data/processed/train.bin"
if not os.path.exists(data_path):
    # Depending on where the script is run from
    data_path = "../data/processed/train.bin"

with open(data_path, "rb") as f:
    raw_data = np.frombuffer(f.read(), dtype=np.uint16)
data = torch.tensor(raw_data.astype(np.int64), dtype=torch.long)

def get_batch():
    # Random batch slicing
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

model = MiniTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Beginning GPU run on {len(data)} token dataset...")
model.train()
t0 = time.time()

for step in range(train_steps):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 500 == 0:
        t1 = time.time()
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | Time: {t1-t0:.2f}s")
        t0 = time.time()

print(f"Final Train Loss: {loss.item():.4f}")

# Overwrite the model.bin that C looks for
save_model_to_c(model, "model.bin")
print("SUCCESS: C Compatible 'model.bin' written! You can now run infer.exe natively in C.")
