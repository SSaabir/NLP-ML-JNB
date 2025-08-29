import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import math
# -------------------
# Load a small English→French dataset
# -------------------
# We will use a tiny subset for demo purposes
ds = load_dataset("opus_books", "en-fr")

subset = ds['train'].select(range(100))
src_sentences = [s['translation']['en'] for s in subset]
tgt_sentences = [s['translation']['fr'] for s in subset]

#src_sentences = [s['translation']['en'] for s in ds['train'][:100]]
#tgt_sentences = [s['translation']['fr'] for s in ds['train'][:100]]

# -------------------
# Vocabulary
# -------------------
def build_vocab(sentences):
    vocab = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
    idx = 4
    for s in sentences:
        for w in s.lower().split():
            if w not in vocab:
                vocab[w] = idx
                idx +=1
    return vocab

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)
inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}

max_src_len = max(len(s.split()) for s in src_sentences)+2
max_tgt_len = max(len(s.split()) for s in tgt_sentences)+2
max_sqc_len = max(max_src_len, max_tgt_len)
def encode_sentence(sentence, vocab, max_len):
    tokens = [vocab.get(w,vocab["<UNK>"]) for w in sentence.lower().split()]
    tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
    if len(tokens)<max_len:
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

encoded_src = torch.tensor([encode_sentence(s, src_vocab, max_src_len) for s in src_sentences])
encoded_tgt = torch.tensor([encode_sentence(s, tgt_vocab, max_tgt_len) for s in tgt_sentences])

# -------------------
# Dataset
# -------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

train_loader = DataLoader(Seq2SeqDataset(encoded_src, encoded_tgt), batch_size=4, shuffle=True)

# -------------------
# Transformer components
# -------------------
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1),:]

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model%num_heads==0
        self.d_k = d_model//num_heads
        self.num_heads = num_heads
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.out = nn.Linear(d_model,d_model)
    def forward(self,q,k,v,mask=None):
        B = q.size(0)
        q = self.q(q).view(B,-1,self.num_heads,self.d_k).transpose(1,2)
        k = self.k(k).view(B,-1,self.num_heads,self.d_k).transpose(1,2)
        v = self.v(v).view(B,-1,self.num_heads,self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        attn = F.softmax(scores,dim=-1)
        context = torch.matmul(attn,v)
        context = context.transpose(1,2).contiguous().view(B,-1,self.num_heads*self.d_k)
        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ff = FeedForward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask=None):
        x2 = self.mha(x,x,x,mask)
        x = self.norm1(x+self.dropout(x2))
        x2 = self.ff(x)
        x = self.norm2(x+self.dropout(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.cross_attn = MultiHeadAttention(d_model,num_heads)
        self.ff = FeedForward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,enc_out,src_mask=None,tgt_mask=None):
        x2 = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x+self.dropout(x2))
        x2 = self.cross_attn(x,enc_out,enc_out,src_mask)
        x = self.norm2(x+self.dropout(x2))
        x2 = self.ff(x)
        x = self.norm3(x+self.dropout(x2))
        return x

class Transformer(nn.Module):
    def __init__(self,src_vocab,tgt_vocab,d_model=64,num_heads=4,num_enc_layers=2,num_dec_layers=2,d_ff=128,max_len=max_sqc_len):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab,d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab,d_model)
        self.pos_enc = PositionalEncoding(d_model,max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff) for _ in range(num_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff) for _ in range(num_dec_layers)])
        self.out = nn.Linear(d_model,tgt_vocab)
    
    def make_pad_mask(self, seq, pad_idx):
        """
        Returns mask of shape (batch, 1, 1, seq_len) suitable for attention.
        True where tokens are NOT padding, False where padding.
        """
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def forward(self, src, tgt, src_pad_idx=None, tgt_pad_idx=None):
        # create masks if pad indices are provided
        src_mask = self.make_pad_mask(src, src_pad_idx) if src_pad_idx is not None else None
        tgt_mask = self.make_pad_mask(tgt, tgt_pad_idx) if tgt_pad_idx is not None else None
        if tgt_mask is not None:
            seq_len = tgt.size(1)
            look_ahead_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
            tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0).unsqueeze(1)

        enc_out = self.pos_enc(self.src_emb(src))
        for layer in self.enc_layers:
            enc_out = layer(enc_out, src_mask)

        dec_out = self.pos_enc(self.tgt_emb(tgt))
        for layer in self.dec_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        return self.out(dec_out)


# -------------------
# Training
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(len(src_vocab),len(tgt_vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for src_batch, tgt_batch in train_loader:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        tgt_input, tgt_out = tgt_batch[:, :-1], tgt_batch[:, 1:]
        optimizer.zero_grad()
        output = model(src_batch, tgt_input)
        loss = criterion(output.reshape(-1,len(tgt_vocab)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

# -------------------
# Greedy decoding
# -------------------
def greedy_decode(model, src_sentence, max_len=max_tgt_len):
    model.eval()
    src_ids = torch.tensor([encode_sentence(src_sentence, src_vocab, max_src_len)]).to(device)
    tgt_ids = torch.tensor([[tgt_vocab["<SOS>"]]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            out = model(src_ids, tgt_ids, src_pad_idx=src_vocab["<PAD>"], tgt_pad_idx=tgt_vocab["<PAD>"])
            next_id = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_ids = torch.cat([tgt_ids, next_id], dim=1)
            if next_id.item() == tgt_vocab["<EOS>"]:
                break

    words = []
    for i in tgt_ids[0][1:]:  # skip <SOS>
        w = inv_tgt_vocab.get(i.item(), "<UNK>")
        if w == "<EOS>":
            break
        words.append(w)
    return " ".join(words)

print("Translation of 'hello world':", greedy_decode(model, "hello world"))
# Save the entire model (architecture + weights)
torch.save(model, "transformer_model.pth")

# Or save just the state dict (recommended)
torch.save(model.state_dict(), "transformer_state_dict.pth")
