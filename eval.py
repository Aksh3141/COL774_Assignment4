#!/usr/bin/env python3
import sys, ast, re, math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONSTANTS
# ==============================================================================
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
MAX_DECODING_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# SHARED HELPERS
# ==============================================================================
def make_src_mask(src_batch, src_lens):
    B, S = src_batch.size()
    return torch.arange(S, device=src_batch.device).unsqueeze(0) < src_lens.unsqueeze(1)

def safe_extract(tag, text):
    pat = rf"<\s*{re.escape(tag)}\s*START\s*>(.*?)<\s*{re.escape(tag)}\s*END\s*>"
    m = re.search(pat, text, re.S | re.I)
    return m.group(1).strip() if m else None

def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums)==2 else None

# ==============================================================================
# SAFE MAZE VISUALIZATION
# ==============================================================================
def plot_maze_safe(tokens, title=None):
    """
    Never crashes. If ADJLIST/ORIGIN/TARGET/PATH are missing, it skips.
    """
    text = " ".join(tokens)

    adj = safe_extract("ADJLIST", text)
    origin = safe_extract("ORIGIN", text)
    target = safe_extract("TARGET", text)
    path_sec = safe_extract("PATH", text)

    if adj is None or origin is None or target is None:
        print("⚠ Visualization skipped: Maze info missing.")
        return

    origin = parse_coords(origin)
    target = parse_coords(target)
    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+,\s*-?\d+\s*\)", path_sec)]

    rows, cols = 6,6
    vertical = np.ones((rows, cols+1), bool)
    horiz = np.ones((rows+1, cols), bool)

    edges = re.findall(r"\(\s*-?\d+,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+,\s*-?\d+\s*\)", adj)
    for e in edges:
        pts = re.findall(r"\(\s*-?\d+,\s*-?\d+\s*\)", e)
        a = parse_coords(pts[0]); b = parse_coords(pts[1])
        if a and b:
            r1,c1 = a; r2,c2 = b
            if r1==r2 and abs(c1-c2)==1:
                vertical[r1, min(c1,c2)+1] = False
            elif c1==c2 and abs(r1-r2)==1:
                horiz[min(r1,r2)+1, c1] = False

    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_aspect("equal")

    for r in range(rows):
        for c in range(cols+1):
            if vertical[r,c]:
                ax.plot([c,c],[rows-r-1,rows-r], color="black", lw=4)
    for r in range(rows+1):
        for c in range(cols):
            if horiz[r,c]:
                ax.plot([c,c+1],[rows-r,rows-r], color="black", lw=4)

    if path:
        xs=[c+0.5 for _,c in path]
        ys=[rows-r-0.5 for r,_ in path]
        ax.plot(xs,ys,'--',color='red')

    plt.title(title)
    plt.show()

# ==============================================================================
# RNN MODELS  (exact same architecture as your training script)
# ==============================================================================
class EncoderRNN(nn.Module):
    def __init__(self, vocab, emb, hid, layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.rnn = nn.RNN(emb, hid, num_layers=layers, batch_first=True)

    def forward(self, src, lens):
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return out, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.W1 = nn.Linear(hid, hid, bias=False)
        self.W2 = nn.Linear(hid, hid, bias=False)
        self.v  = nn.Linear(hid, 1, bias=False)

    def forward(self, hidden_top, enc_out, mask):
        W1e = self.W1(enc_out)
        W2h = self.W2(hidden_top).unsqueeze(1)
        score = self.v(torch.tanh(W1e + W2h)).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        attn = torch.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn

class DecoderRNN(nn.Module):
    def __init__(self, vocab, emb, hid, layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=pad_idx)
        self.attn = BahdanauAttention(hid)
        self.rnn = nn.RNN(emb + hid, hid, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward_step(self, inp, hidden, enc_out, mask):
        emb = self.embedding(inp).unsqueeze(1)
        ctx, _ = self.attn(hidden[-1], enc_out, mask)
        rnn_in = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)
        out, hidden = self.rnn(rnn_in, hidden)
        return self.out(out.squeeze(1)), hidden

def rnn_greedy_decode(encoder, decoder, src, lens, stoi, itos):
    B = src.size(0)
    enc_out, hidden = encoder(src, lens)
    mask = make_src_mask(src, lens)
    inp = torch.full((B,), stoi[SOS_TOKEN], dtype=torch.long, device=src.device)
    seq = [[] for _ in range(B)]

    for _ in range(MAX_DECODING_LEN):
        logits, hidden = decoder.forward_step(inp, hidden, enc_out, mask)
        nxt = logits.argmax(-1)
        for i in range(B):
            tok = itos.get(nxt[i].item(), PAD_TOKEN)
            if tok != EOS_TOKEN:
                seq[i].append(tok)
        inp = nxt
    return seq

# ==============================================================================
# TRANSFORMER MODELS — EXACT MATCH WITH TRAINING CODE
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        x = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(x*div)
        pe[:,1::2] = torch.cos(x*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def subsequent_mask(sz, device):
    return torch.triu(torch.ones((sz,sz), device=device), diagonal=1).bool()

class TransformerSeq2Seq(nn.Module):
    """
    EXACT module names as training script:
    embedding, pos_enc, encoder, decoder, out
    """
    def __init__(self, vocab, d_model, nhead, layers, ff, dropout, pad_idx, max_len):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        self.pos_enc  = PositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)

        self.out = nn.Linear(d_model, vocab)

    def encode(self, src, lens):
        emb = self.embedding(src) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)

        src_mask = ~(make_src_mask(src, lens))
        mem = self.encoder(emb, src_key_padding_mask=src_mask)
        return mem, src_mask

    def decode_step(self, cur, mem, src_mask):
        emb = self.embedding(cur) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)
        T = cur.size(1)
        tgt_mask = subsequent_mask(T, cur.device)
        dec = self.decoder(
            emb, mem,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        return self.out(dec[:, -1, :])

def transformer_greedy_decode(model, src, lens, stoi, itos):
    mem, src_mask = model.encode(src, lens)
    B = src.size(0)
    cur = torch.full((B,1), stoi[SOS_TOKEN], dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    seq = [[] for _ in range(B)]

    for _ in range(MAX_DECODING_LEN):
        logits = model.decode_step(cur, mem, src_mask)
        nxt = logits.argmax(-1)
        cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)

        for i in range(B):
            if done[i]: continue
            tok = itos.get(nxt[i].item(), PAD_TOKEN)
            if tok == EOS_TOKEN:
                done[i] = True
            else:
                seq[i].append(tok)

        if done.all(): break

    return seq

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    model_path = sys.argv[1]
    model_type = sys.argv[2].strip().lower()
    data_path  = sys.argv[3]
    output_path= sys.argv[4]

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=DEVICE)
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    pad_idx = stoi[PAD_TOKEN]

    # Build model
    if model_type == "rnn":
        cfg = ckpt["config"]
        vocab = len(stoi)
        encoder = EncoderRNN(vocab, cfg["embed_dim"], cfg["hidden_dim"],
                             cfg["num_layers"], pad_idx).to(DEVICE)
        decoder = DecoderRNN(vocab, cfg["embed_dim"], cfg["hidden_dim"],
                             cfg["num_layers"], pad_idx).to(DEVICE)
        encoder.load_state_dict(ckpt["encoder_state"])
        decoder.load_state_dict(ckpt["decoder_state"])
        use_transformer = False

    elif model_type == "transformer":
        cfg = ckpt["config"]
        vocab = len(stoi)

        model = TransformerSeq2Seq(
            vocab,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            layers=cfg["num_layers"],
            ff=512,
            dropout=0.1,
            pad_idx=pad_idx,
            max_len=512
        ).to(DEVICE)

        model.load_state_dict(ckpt["state_dict"])
        use_transformer = True

    else:
        raise ValueError("model_type must be 'rnn' or 'transformer'")

    # Load eval CSV
    df = pd.read_csv(data_path)
    predictions = []

    # Inference
    for idx, row in df.iterrows():
        inp = ast.literal_eval(row["input_sequence"])
        src = torch.tensor([[stoi.get(t, pad_idx) for t in inp]], device=DEVICE)
        src_len = torch.tensor([len(inp)], device=DEVICE)

        if use_transformer:
            pred = transformer_greedy_decode(model, src, src_len, stoi, itos)[0]
        else:
            pred = rnn_greedy_decode(encoder, decoder, src, src_len, stoi, itos)[0]

        predictions.append(str(pred))

    # Save output
    out = df.copy()
    out["output_path"] = predictions
    out.to_csv(output_path, index=False)
    print("Saved predictions to:", output_path)

    # Debug visualization
    VISUALIZE = False  # Set True for debugging

    if VISUALIZE:
        for i in range(min(3, len(out))):
            inp = ast.literal_eval(df.iloc[i]["input_sequence"])
            pred = ast.literal_eval(out.iloc[i]["output_path"])
            toks = inp + ["<PATH START>"] + pred + ["<PATH END>"]
            plot_maze_safe(toks, title=f"Row {i}")
