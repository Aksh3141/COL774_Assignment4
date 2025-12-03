import os, re, random, math, ast, time
from collections import Counter
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ---------- Paths (edit if needed)
TRAIN_CSV = "/kaggle/input/zera123/train_6x6_mazes.csv"
VAL_CSV   = "/kaggle/input/zera123/test_6x6_mazes.csv"
ASSIGNMENT_PDF = "/mnt/data/col774_ass4.pdf"

# ---------- Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
OPTIMIZER_LR = 1.0
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LEN = 512
WARMUP_STEPS = 8000    

LABEL_SMOOTHING = 0.1    
PATIENCE = 6

USE_AMP = True
ACCUM_STEPS = 1

NUM_WORKERS = 0
PIN_MEMORY = False

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SEED = 42

# ---------- Device & determinism
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, flush=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(1)
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# ---------- Helpers
def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def extract_between(tag, text):
    pattern = rf"<\s*{re.escape(tag)}\s*START\s*>(.*?)<\s*{re.escape(tag)}\s*END\s*>"
    m = re.search(pattern, text, re.S | re.I)
    return m.group(1).strip() if m else ""

def plot_maze(tokens, title=None, save_path=None, rows=6, cols=6):
    text = " ".join(tokens)
    adj_section = extract_between("ADJLIST", text)
    path_section = extract_between("PATH", text)
    edge_matches = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_matches:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        a = parse_coords(coords[0]); b = parse_coords(coords[1])
        if a and b:
            edges.append((a,b))
    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)]
    if not edges and not path:
        raise ValueError("No edges or path to visualize")
    vertical_walls = np.ones((rows, cols+1), dtype=bool)
    horizontal_walls = np.ones((rows+1, cols), dtype=bool)
    for (r1,c1),(r2,c2) in edges:
        if r1==r2 and abs(c1-c2)==1:
            c_between = min(c1,c2)+1
            if 0<=r1<rows and 0<=c_between<=cols:
                vertical_walls[r1,c_between]=False
        elif c1==c2 and abs(r1-r2)==1:
            r_between = min(r1,r2)+1
            if 0<=r_between<=rows and 0<=c1<cols:
                horizontal_walls[r_between,c1]=False
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_aspect('equal')
    for r in range(rows):
        for c in range(cols):
            x0,x1=c,c+1
            y_top=rows-r; y_bot=rows-r-1
            ax.plot([x0,x1],[y_top,y_top], color='lightgray', lw=0.5)
            ax.plot([x0,x1],[y_bot,y_bot], color='lightgray', lw=0.5)
            ax.plot([x0,x0],[y_bot,y_top], color='lightgray', lw=0.5)
            ax.plot([x1,x1],[y_bot,y_top], color='lightgray', lw=0.5)
    for r in range(rows):
        for c in range(cols+1):
            if vertical_walls[r,c]:
                x=c; y_top=rows-r; y_bot=rows-r-1
                ax.plot([x,x],[y_bot,y_top], color='black', lw=4, solid_capstyle='butt')
    for r in range(rows+1):
        for c in range(cols):
            if horizontal_walls[r,c]:
                y=rows-r
                ax.plot([c,c+1],[y,y], color='black', lw=4, solid_capstyle='butt')
    if path:
        for (r,c) in path:
            if 0<=r<rows and 0<=c<cols:
                ax.add_patch(plt.Rectangle((c,rows-r-1),1,1,facecolor=(1,0.9,0.9),zorder=0))
        path_x=[c+0.5 for (r,c) in path]; path_y=[rows-r-0.5 for (r,c) in path]
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
        ax.scatter(path_x[0], path_y[0], c='red', s=80, marker='o', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], c='red', s=80, marker='x', zorder=5)
    if title: ax.set_title(title)
    ax.set_xlim(0,cols); ax.set_ylim(0,rows)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()

# ---------- Vocab & dataset
def build_vocab_from_dfs(dfs: List[pd.DataFrame]):
    tokens = set()
    for df in dfs:
        for col in ['input_sequence','output_path']:
            if col not in df.columns:
                continue
            for s in df[col].astype(str).values:
                try:
                    toks = ast.literal_eval(s) if isinstance(s, str) else s
                    if isinstance(toks, (list,tuple)):
                        tokens.update(toks)
                except Exception:
                    tokens.update(str(s).split())
    token_list = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(t for t in tokens if t not in {PAD_TOKEN,SOS_TOKEN,EOS_TOKEN})
    stoi = {t:i for i,t in enumerate(token_list)}
    itos = {i:t for i,t in enumerate(token_list)}
    return stoi, itos

class MazeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stoi: dict):
        self.df = df.reset_index(drop=True)
        self.stoi = stoi
        self.pad_idx = stoi[PAD_TOKEN]; self.sos_idx = stoi[SOS_TOKEN]; self.eos_idx = stoi[EOS_TOKEN]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        inp_tokens = ast.literal_eval(row['input_sequence']) if isinstance(row['input_sequence'], str) else row['input_sequence']
        out_tokens = ast.literal_eval(row['output_path']) if isinstance(row['output_path'], str) else row['output_path']
        src_idx = [self.stoi.get(t, self.pad_idx) for t in inp_tokens]
        trg_in_idx = [self.sos_idx] + [self.stoi.get(t, self.pad_idx) for t in out_tokens]
        trg_tgt_idx = [self.stoi.get(t, self.pad_idx) for t in out_tokens] + [self.eos_idx]
        return {'src_idx': torch.tensor(src_idx, dtype=torch.long),
                'src_tokens': inp_tokens,
                'trg_in_idx': torch.tensor(trg_in_idx, dtype=torch.long),
                'trg_tgt_idx': torch.tensor(trg_tgt_idx, dtype=torch.long),
                'trg_tokens': out_tokens}

PAD_IDX = None

def collate_fn(batch):
    global PAD_IDX
    assert PAD_IDX is not None, "set PAD_IDX before dataloader"
    src_seqs = [b['src_idx'] for b in batch]
    trg_in_seqs = [b['trg_in_idx'] for b in batch]
    trg_tgt_seqs = [b['trg_tgt_idx'] for b in batch]
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    trg_lens = torch.tensor([len(s) for s in trg_tgt_seqs], dtype=torch.long)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    trg_in_padded = nn.utils.rnn.pad_sequence(trg_in_seqs, batch_first=True, padding_value=PAD_IDX)
    trg_tgt_padded = nn.utils.rnn.pad_sequence(trg_tgt_seqs, batch_first=True, padding_value=PAD_IDX)
    return {'src': src_padded, 'src_lens': src_lens, 'trg_in': trg_in_padded, 'trg_tgt': trg_tgt_padded,
            'src_tokens':[b['src_tokens'] for b in batch], 'trg_tokens':[b['trg_tokens'] for b in batch], 'trg_lens':trg_lens}

# ---------- masks & metrics
def make_src_mask(src_batch, src_lens):
    B,S = src_batch.size()
    return (torch.arange(S, device=src_batch.device).unsqueeze(0) < src_lens.unsqueeze(1))

def subsequent_mask(sz:int, device=None):
    # boolean mask (compatible types with key_padding_mask)
    return torch.triu(torch.ones((sz, sz), device=device), diagonal=1).bool()

def compute_token_seq_metrics(preds: List[List[str]], targets: List[List[str]]):
    total_positions=0; correct_positions=0; seq_correct=0
    tp=0; pred_count=0; gold_count=0
    for p,g in zip(preds, targets):
        L = max(len(p), len(g))
        for i in range(L):
            total_positions += 1
            pi = p[i] if i < len(p) else None
            gi = g[i] if i < len(g) else None
            if pi == gi and pi is not None:
                correct_positions += 1
        if p==g: seq_correct += 1
        pc = Counter([tok for tok in p if tok not in {PAD_TOKEN, EOS_TOKEN, SOS_TOKEN}])
        gc = Counter([tok for tok in g if tok not in {PAD_TOKEN, EOS_TOKEN, SOS_TOKEN}])
        for tok in pc:
            tp += min(pc[tok], gc.get(tok,0))
        pred_count += sum(pc.values()); gold_count += sum(gc.values())
    token_acc = correct_positions / total_positions if total_positions>0 else 0.0
    seq_acc = seq_correct / len(preds) if len(preds)>0 else 0.0
    precision = tp / pred_count if pred_count>0 else 0.0
    recall = tp / gold_count if gold_count>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return token_acc, seq_acc, f1

# ---------- Transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, pad_idx=0, max_len=512):
        super().__init__()
        self.pad_idx = pad_idx; self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)
        try:
            self.out.weight = self.embedding.weight
        except Exception:
            pass
    def encode(self, src, src_lens):
        emb = self.embedding(src) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)
        src_key_padding_mask = ~(make_src_mask(src, src_lens))
        enc_out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        return enc_out, src_key_padding_mask
    def decode_train(self, trg_in, enc_out, src_key_padding_mask):
        emb = self.embedding(trg_in) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)
        T = trg_in.size(1)
        tgt_mask = subsequent_mask(T, device=trg_in.device)
        tgt_key_padding_mask = (trg_in == self.pad_idx)
        dec_out = self.decoder(emb, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        logits = self.out(dec_out)
        return logits
    def greedy_decode(self, enc_out, src_key_padding_mask, stoi, itos, max_len=200, device='cpu'):
        B = enc_out.size(0)
        generated = [[] for _ in range(B)]
        cur_tokens = torch.full((B,1), stoi[SOS_TOKEN], dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(max_len):
            logits = self.decode_train(cur_tokens, enc_out, src_key_padding_mask)
            next_logits = logits[:, -1, :]
            next_ids = next_logits.argmax(-1)
            cur_tokens = torch.cat([cur_tokens, next_ids.unsqueeze(1)], dim=1)
            for i in range(B):
                if finished[i]: continue
                tid = next_ids[i].item()
                if tid == stoi[EOS_TOKEN]:
                    finished[i] = True
                else:
                    generated[i].append(itos.get(tid, PAD_TOKEN))
            if finished.all(): break
        return generated

# ---------- Label smoothing
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index):
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0
        self.smoothing = label_smoothing
        self.vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
    def forward(self, pred_logits, target):
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs, device=log_probs.device)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            mask = target.eq(self.ignore_index)
            target_clamped = target.clone()
            target_clamped[mask] = 0
            true_dist.scatter_(1, target_clamped.unsqueeze(1), self.confidence)
            true_dist[mask] = 0
        return torch.sum(-true_dist * log_probs)

# ---------- Noam scheduler
def get_noam_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step):
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------- DataParallel helpers
def underlying_model(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m

def m_encode(model, src, src_lens):
    return underlying_model(model).encode(src, src_lens)

def m_decode_train(model, trg_in, enc_out, src_key_padding_mask):
    return underlying_model(model).decode_train(trg_in, enc_out, src_key_padding_mask)

# ---------- eval & train
def evaluate_transformer(transformer, dataloader, stoi, itos, pad_idx, device):
    transformer.eval()
    all_preds=[]; all_targets=[]; total_loss=0.0; total_tokens=0
    if LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingLoss(LABEL_SMOOTHING, tgt_vocab_size=len(stoi), ignore_index=pad_idx)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval', leave=False):
            src = batch['src'].to(device, non_blocking=True); src_lens = batch['src_lens'].to(device, non_blocking=True)
            trg_in = batch['trg_in'].to(device, non_blocking=True); trg_tgt = batch['trg_tgt'].to(device, non_blocking=True)
            enc_out, src_key_padding_mask = m_encode(transformer, src, src_lens)
            logits = m_decode_train(transformer, trg_in, enc_out, src_key_padding_mask)
            min_len = min(logits.size(1), trg_tgt.size(1))
            if min_len<=0: continue
            logit_seg = logits[:, :min_len, :].reshape(-1, logits.size(-1))
            targets = trg_tgt[:, :min_len].reshape(-1)
            loss = criterion(logit_seg, targets)
            mask_tokens = (targets != pad_idx).sum().item()
            total_loss += loss.item(); total_tokens += mask_tokens
            pred_ids = logits.argmax(-1)
            for i in range(pred_ids.size(0)):
                pred_seq=[]; tgt_seq=[]
                for tid in pred_ids[i].tolist():
                    tok = itos.get(tid, PAD_TOKEN)
                    if tok==EOS_TOKEN: break
                    pred_seq.append(tok)
                for tid in trg_tgt[i].tolist():
                    tok = itos.get(tid, PAD_TOKEN)
                    if tok==EOS_TOKEN: break
                    tgt_seq.append(tok)
                all_preds.append(pred_seq); all_targets.append(tgt_seq)
    avg_loss = total_loss / total_tokens if total_tokens>0 else 0.0
    token_acc, seq_acc, f1 = compute_token_seq_metrics(all_preds, all_targets)
    return avg_loss, token_acc, seq_acc, f1, all_preds, all_targets

def train_transformer(transformer, train_loader, val_loader, test_loader, stoi, itos, pad_idx, device, epochs=EPOCHS):
    # OPT lr set to 1.0; scheduler will scale
    optimizer = AdamW(transformer.parameters(), lr=OPTIMIZER_LR, weight_decay=1e-5)
    scheduler = get_noam_scheduler(optimizer, D_MODEL, warmup_steps=WARMUP_STEPS)
    if LABEL_SMOOTHING > 0.0:
        train_criterion = LabelSmoothingLoss(LABEL_SMOOTHING, tgt_vocab_size=len(stoi), ignore_index=pad_idx)
    else:
        train_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type=='cuda'))

    history = {'train_losses':[], 'val_losses':[], 'test_losses':[],
               'train_token_accs':[], 'val_token_accs':[], 'test_token_accs':[],
               'train_seq_accs':[], 'val_seq_accs':[], 'test_seq_accs':[],
               'train_f1s':[], 'val_f1s':[], 'test_f1s':[]}

    best_val_seq = -1.0
    wait = 0
    global_step = 0

    for epoch in range(1, epochs+1):
        transformer.train()
        running_loss_sum = 0.0
        total_tokens_for_epoch = 0
        t0 = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False)):
            src = batch['src'].to(device, non_blocking=True); src_lens = batch['src_lens'].to(device, non_blocking=True)
            trg_in = batch['trg_in'].to(device, non_blocking=True); trg_tgt = batch['trg_tgt'].to(device, non_blocking=True)

            enc_out, src_key_padding_mask = m_encode(transformer, src, src_lens)
            with torch.amp.autocast(device_type="cuda", enabled=(USE_AMP and device.type=='cuda')):
                logits = m_decode_train(transformer, trg_in, enc_out, src_key_padding_mask)
                min_len = min(logits.size(1), trg_tgt.size(1))
                if min_len<=0: continue
                logit_seg = logits[:, :min_len, :].reshape(-1, logits.size(-1))
                targets = trg_tgt[:, :min_len].reshape(-1)
                loss_sum = train_criterion(logit_seg, targets)

            token_count = (targets != pad_idx).sum().item()
            if token_count == 0: continue
            loss = loss_sum / token_count
            loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler *after* optimizer.step
                scheduler.step()

            running_loss_sum += loss_sum.item()
            total_tokens_for_epoch += token_count
            global_step += 1

        epoch_train_loss = (running_loss_sum / total_tokens_for_epoch) if total_tokens_for_epoch>0 else 0.0
        history['train_losses'].append(epoch_train_loss)

        # evaluate
        t_loss, t_tok, t_seq, t_f1, _, _ = evaluate_transformer(transformer, train_loader, stoi, itos, pad_idx, device)
        v_loss, v_tok, v_seq, v_f1, _, _ = evaluate_transformer(transformer, val_loader, stoi, itos, pad_idx, device)
        te_loss, te_tok, te_seq, te_f1, _, _ = evaluate_transformer(transformer, test_loader, stoi, itos, pad_idx, device)

        history['val_losses'].append(v_loss); history['test_losses'].append(te_loss)
        history['train_token_accs'].append(t_tok); history['val_token_accs'].append(v_tok); history['test_token_accs'].append(te_tok)
        history['train_seq_accs'].append(t_seq); history['val_seq_accs'].append(v_seq); history['test_seq_accs'].append(te_seq)
        history['train_f1s'].append(t_f1); history['val_f1s'].append(v_f1); history['test_f1s'].append(te_f1)

        print(f"Epoch {epoch} ({time.time()-t0:.1f}s): TrainTokAcc={t_tok:.4f} TrainSeqAcc={t_seq:.4f} TrainF1={t_f1:.4f} | ValTokAcc={v_tok:.4f} ValSeqAcc={v_seq:.4f} ValF1={v_f1:.4f}", flush=True)

        current_state_dict = underlying_model(transformer).state_dict()
        if v_seq > best_val_seq:
            best_val_seq = v_seq
            wait = 0
            torch.save({'state_dict': current_state_dict, 'stoi': stoi, 'itos': itos, 'config':{'d_model':D_MODEL,'nhead':NHEAD,'num_layers':NUM_LAYERS}}, 'best_transformer_q3.pth')
            print(f"Saved best_transformer_q3.pth (val seq acc {best_val_seq:.4f})", flush=True)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping: no improvement in {PATIENCE} epochs (best val seq {best_val_seq:.4f})", flush=True)
                break

    return history

# ---------- Prepare data & dataloaders
print("Loading CSVs...", flush=True)
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(VAL_CSV)

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_total = len(train_df); n_val = max(1, int(0.10*n_total))
train_main_df = train_df.iloc[:-n_val].reset_index(drop=True)
train_val_df  = train_df.iloc[-n_val:].reset_index(drop=True)

stoi, itos = build_vocab_from_dfs([train_main_df, train_val_df, test_df])
PAD_IDX = stoi[PAD_TOKEN]
print("Vocab size:", len(stoi), "PAD_IDX:", PAD_IDX, flush=True)

train_ds = MazeDataset(train_main_df, stoi)
train_val_ds = MazeDataset(train_val_df, stoi)
test_ds = MazeDataset(test_df, stoi)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
val_loader   = DataLoader(train_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

# ---------- debug batch (paste output here if seq acc remains 0)
batch = next(iter(train_loader))
print("\n--- DEBUG BATCH ---", flush=True)
print("SRC TOKENS (sample):", batch['src_tokens'][0], flush=True)
print("TRG TOKENS (sample):", batch['trg_tokens'][0], flush=True)
print("SRC Padded shape:", batch['src'].shape, flush=True)
print("TRG_IN Padded shape:", batch['trg_in'].shape, flush=True)
print("TRG_TGT Padded shape:", batch['trg_tgt'].shape, flush=True)
print("SRC IDX (first 40):", batch['src'][0][:40].tolist(), flush=True)
print("TRG_IN IDX (first 40):", batch['trg_in'][0][:40].tolist(), flush=True)
print("TRG_TGT IDX (first 40):", batch['trg_tgt'][0][:40].tolist(), flush=True)
print("Lengths (tokens):", len(batch['src_tokens'][0]), len(batch['trg_tokens'][0]), flush=True)
print("--- END DEBUG ---\n", flush=True)

# ---------- instantiate model + train
transformer = TransformerSeq2Seq(len(stoi), d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, pad_idx=PAD_IDX, max_len=MAX_LEN)
transformer = transformer.to(DEVICE)
if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1:
    transformer = nn.DataParallel(transformer)

print("Model instantiated and moved to", DEVICE, flush=True)
history = train_transformer(transformer, train_loader, val_loader, test_loader, stoi, itos, PAD_IDX, DEVICE, epochs=EPOCHS)

# ---------- save + plots
final_sd = underlying_model(transformer).state_dict()
torch.save({'state_dict': final_sd, 'stoi': stoi, 'itos': itos, 'config':{'d_model':D_MODEL,'nhead':NHEAD,'num_layers':NUM_LAYERS}}, 'transformer_q3_final_v2.pth')
print('Saved transformer_q3_final_v2.pth', flush=True)

if isinstance(history, dict):
    h = history
    plt.figure(figsize=(8,5)); plt.plot(h['train_losses'], label='Train Loss'); plt.plot(h['val_losses'], label='Val Loss'); plt.plot(h['test_losses'], label='Test Loss'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("loss_curves_v2.png", dpi=200); plt.show()
    plt.figure(figsize=(8,5)); plt.plot(h['train_token_accs'], label='Train Token Acc'); plt.plot(h['val_token_accs'], label='Val Token Acc'); plt.plot(h['test_token_accs'], label='Test Token Acc'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("token_acc_v2.png", dpi=200); plt.show()

# ---------- visualize some test examples if desired (like before)
if os.path.exists('transformer_q3_final_v2.pth'):
    ckpt = torch.load('transformer_q3_final_v2.pth', map_location=DEVICE)
    stoi = ckpt['stoi']; itos = ckpt['itos']
    transformer.load_state_dict(ckpt['state_dict']); transformer.eval()
    for i in range(min(3, len(test_df))):
        row = test_df.iloc[i]
        inp_tokens = ast.literal_eval(row['input_sequence']) if isinstance(row['input_sequence'], str) else row['input_sequence']
        tgt_tokens = ast.literal_eval(row['output_path']) if isinstance(row['output_path'], str) else row['output_path']
        src_idx = torch.tensor([[stoi.get(t, PAD_IDX) for t in inp_tokens]], dtype=torch.long, device=DEVICE)
        src_len = torch.tensor([len(inp_tokens)], dtype=torch.long, device=DEVICE)
        enc_out, src_kpm = underlying_model(transformer).encode(src_idx, src_len)
        pred = underlying_model(transformer).greedy_decode(enc_out, src_kpm, stoi, itos, max_len=200, device=DEVICE)[0]
        final_tokens = inp_tokens.copy() + ['<PATH START>'] + pred + ['<PATH END>']
        print("Example", i, "GT len", len(tgt_tokens), "Pred len", len(pred), flush=True)
        try:
            plot_maze(final_tokens, title=f'Example {i}', save_path=f'vis_v2_{i}.png')
        except Exception as e:
            print("vis failed:", e, flush=True)