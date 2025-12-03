import os
import re
import random
from collections import Counter
from typing import List

22.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TRAIN_CSV = "/content/train_6x6_mazes.csv"
VAL_CSV   = "/content/test_6x6_mazes.csv"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EMBED_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS = 2
TEACHER_FORCING_RATIO = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "rnn_bahdanau_final.pth"

PLOT_PNG = "training_curves_3way.png"
PLOT_PDF = "training_curves_3way.pdf"

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

MAX_DECODING_LEN = 200
RANDOM_STATE = 42


def parse_coords(s):
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def extract_between(tag, text):
    patterns = [
        rf"<\s*{tag}\s*[_\-\s]?\s*START\s*>(.*?)<\s*{tag}\s*[_\-\s]?\s*END\s*>",
        rf"<\s*{tag}START\s*>(.*?)<\s*{tag}END\s*>",
        rf"<\s*{tag}\s*START\s*>(.*?)<\s*{tag}\s*END\s*>",
        rf"<\s*{tag.replace(' ', '_')}\s*START\s*>(.*?)<\s*{tag.replace(' ', '_')}\s*END\s*>",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m:
            return m.group(1).strip()
    raise ValueError(f"Could not find section for tag '{tag}'.")

def plot_maze(tokens, title=None, save_path=None):
    text = " ".join(tokens)
    adj_section = extract_between("ADJLIST", text)
    origin_section = extract_between("ORIGIN", text)
    target_section = extract_between("TARGET", text)
    path_section = extract_between("PATH", text)

    origin = parse_coords(origin_section)
    target = parse_coords(target_section)

    edge_matches = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_matches:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        a = parse_coords(coords[0])
        b = parse_coords(coords[1])
        edges.append((a, b))

    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)]
    if not path:
        nums = re.findall(r"-?\d+\s*,\s*-?\d+", path_section)
        path = [tuple(map(int, re.findall(r"-?\d+", s))) for s in nums]

    if not edges:
        raise ValueError("No edges found in adjacency list.")

    rows, cols = 6, 6
    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            c_between = min(c1, c2) + 1
            vertical_walls[r1, c_between] = False
        elif c1 == c2:
            r_between = min(r1, r2) + 1
            horizontal_walls[r_between, c1] = False
        else:
            pass

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    # light grid
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=2)
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=2)
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=2)
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=2)

    # walls
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=5, solid_capstyle='butt')
    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=5, solid_capstyle='butt')

    # shade path cells & draw path
    if path:
        for (r, c) in path:
            rect = plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=(1, 0.9, 0.9), edgecolor=None, zorder=0)
            ax.add_patch(rect)
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
        ax.scatter(path_x[0], path_y[0], c='red', s=80, marker='o', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], c='red', s=80, marker='x', zorder=5)
    else:
        ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
        tx, ty = target[1] + 0.5, rows - target[0] - 0.5
        ax.scatter(ox, oy, c='red', s=80, marker='o', zorder=5)
        ax.scatter(tx, ty, c='red', s=80, marker='x', zorder=5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    plt.yticks([])
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def build_vocab_from_dfs(dfs: List[pd.DataFrame]):
    tokens = set()
    for df in dfs:
        for col in ["input_sequence", "output_path"]:
            for s in df[col].astype(str).values:
                toks = eval(s)
                tokens.update(toks)
    token_list = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(t for t in tokens if t not in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN})
    stoi = {t: i for i, t in enumerate(token_list)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos

class MazeDataset(Dataset):
    def __init__(self, df, stoi):
        self.df = df.reset_index(drop=True)
        self.stoi = stoi
        self.pad_idx = stoi[PAD_TOKEN]
        self.sos_idx = stoi[SOS_TOKEN]
        self.eos_idx = stoi[EOS_TOKEN]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        inp_tokens = eval(row["input_sequence"]) if isinstance(row["input_sequence"], str) else row["input_sequence"]
        out_tokens = eval(row["output_path"]) if isinstance(row["output_path"], str) else row["output_path"]
        src_idx = [self.stoi.get(t, self.pad_idx) for t in inp_tokens]
        trg_in_idx = [self.sos_idx] + [self.stoi.get(t, self.pad_idx) for t in out_tokens]
        trg_tgt_idx = [self.stoi.get(t, self.pad_idx) for t in out_tokens] + [self.eos_idx]
        return {
            "src_idx": torch.tensor(src_idx, dtype=torch.long),
            "src_tokens": inp_tokens,
            "trg_in_idx": torch.tensor(trg_in_idx, dtype=torch.long),
            "trg_tgt_idx": torch.tensor(trg_tgt_idx, dtype=torch.long),
            "trg_tokens": out_tokens
        }

# collate will use PAD_IDX global set after vocab built
PAD_IDX = None
def collate_fn(batch):
    global PAD_IDX
    assert PAD_IDX is not None, "PAD_IDX must be set before creating DataLoader"
    src_seqs = [b["src_idx"] for b in batch]
    trg_in_seqs = [b["trg_in_idx"] for b in batch]
    trg_tgt_seqs = [b["trg_tgt_idx"] for b in batch]
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    trg_lens = torch.tensor([len(s) for s in trg_tgt_seqs], dtype=torch.long)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    trg_in_padded = nn.utils.rnn.pad_sequence(trg_in_seqs, batch_first=True, padding_value=PAD_IDX)
    trg_tgt_padded = nn.utils.rnn.pad_sequence(trg_tgt_seqs, batch_first=True, padding_value=PAD_IDX)
    return {
        "src": src_padded,
        "src_lens": src_lens,
        "trg_in": trg_in_padded,
        "trg_tgt": trg_tgt_padded,
        "src_tokens": [b["src_tokens"] for b in batch],
        "trg_tokens": [b["trg_tokens"] for b in batch],
        "trg_lens": trg_lens
    }


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
    def forward(self, src, src_lens):
        emb = self.embedding(src)
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)
        out_unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out_unpacked, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, hidden_top, encoder_outputs, mask=None):
        W1e = self.W1(encoder_outputs)
        W2h = self.W2(hidden_top).unsqueeze(1)
        score = self.v(torch.tanh(W1e + W2h)).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(~mask, -1e9)
        attn = torch.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attn = BahdanauAttention(hidden_dim)
        self.rnn = nn.RNN(embed_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, trg_inputs, initial_hidden, encoder_outputs, encoder_mask, teacher_forcing_ratio=0.5):
        B, T = trg_inputs.size()
        V = self.out.out_features
        outputs = torch.zeros(B, T, V, device=trg_inputs.device)
        hidden = initial_hidden
        input_tok = trg_inputs[:, 0]
        for t in range(1, T):
            emb = self.embedding(input_tok).unsqueeze(1)
            hidden_top = hidden[-1]
            context, attw = self.attn(hidden_top, encoder_outputs, mask=encoder_mask)
            rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
            out, hidden = self.rnn(rnn_in, hidden)
            out = out.squeeze(1)
            logits = self.out(out)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(-1)
            input_tok = trg_inputs[:, t] if teacher_force else top1
        return outputs
    def forward_step(self, input_tok, last_hidden, encoder_outputs, encoder_mask):
        emb = self.embedding(input_tok).unsqueeze(1)
        hidden_top = last_hidden[-1]
        context, attw = self.attn(hidden_top, encoder_outputs, mask=encoder_mask)
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, hidden = self.rnn(rnn_in, last_hidden)
        logits = self.out(out.squeeze(1))
        return logits, hidden, attw


def make_src_mask(src_batch, src_lens):
    B, S = src_batch.size()
    return torch.arange(S, device=src_batch.device).unsqueeze(0) < src_lens.unsqueeze(1)

def greedy_decode(encoder, decoder, src_tensor, src_len, stoi, itos, max_len=MAX_DECODING_LEN):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        enc_outs, enc_hidden = encoder(src_tensor, src_len)
        enc_mask = make_src_mask(src_tensor, src_len).to(src_tensor.device)
        batch_size = src_tensor.size(0)
        generated = [[] for _ in range(batch_size)]
        input_tok = torch.tensor([stoi[SOS_TOKEN]] * batch_size, dtype=torch.long, device=src_tensor.device)
        hidden = enc_hidden
        for t in range(max_len):
            logits, hidden, _ = decoder.forward_step(input_tok, hidden, enc_outs, enc_mask)
            top1 = logits.argmax(-1)
            input_tok = top1
            for i in range(batch_size):
                tid = top1[i].item()
                if tid == stoi[EOS_TOKEN]:
                    continue
                generated[i].append(itos.get(tid, PAD_TOKEN))
        return generated

def compute_token_seq_metrics(preds: List[List[str]], targets: List[List[str]]):
    total_positions = 0
    correct_positions = 0
    seq_correct = 0
    tp = 0
    pred_count = 0
    gold_count = 0
    for p, g in zip(preds, targets):
        L = max(len(p), len(g))
        for i in range(L):
            total_positions += 1
            pi = p[i] if i < len(p) else None
            gi = g[i] if i < len(g) else None
            if pi == gi and pi is not None:
                correct_positions += 1
        if p == g:
            seq_correct += 1
        pc = Counter([tok for tok in p if tok not in {PAD_TOKEN, EOS_TOKEN, SOS_TOKEN}])
        gc = Counter([tok for tok in g if tok not in {PAD_TOKEN, EOS_TOKEN, SOS_TOKEN}])
        for tok in pc:
            tp += min(pc[tok], gc.get(tok, 0))
        pred_count += sum(pc.values()); gold_count += sum(gc.values())
    token_acc = correct_positions / total_positions if total_positions > 0 else 0.0
    seq_acc = seq_correct / len(preds) if len(preds) > 0 else 0.0
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return token_acc, seq_acc, f1


def evaluate_split(encoder, decoder, dataloader, stoi, itos, pad_idx, desc="Evaluating"):
    encoder.eval(); decoder.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
    eval_bar = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in eval_bar:
            src = batch["src"].to(DEVICE)
            src_lens = batch["src_lens"].to(DEVICE)
            trg_in = batch["trg_in"].to(DEVICE)
            trg_tgt = batch["trg_tgt"].to(DEVICE)
            B = src.size(0)

            enc_outs, enc_hidden = encoder(src, src_lens)
            enc_mask = make_src_mask(src, src_lens).to(DEVICE)
            outputs = decoder(trg_in, enc_hidden, enc_outs, enc_mask, teacher_forcing_ratio=0.0)

            # --- FIX: compute loss only over min common timesteps
            dec_time_steps = outputs.size(1) - 1  # positions 1..T-1 are predictions
            tgt_time_steps = trg_tgt.size(1)
            min_len = min(dec_time_steps, tgt_time_steps)
            if min_len <= 0:
                continue
            logits = outputs[:, 1:1 + min_len, :].reshape(-1, outputs.size(-1))
            targets = trg_tgt[:, :min_len].reshape(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # greedy preds from logits over full outputs (use argmax)
            pred_ids = outputs.argmax(-1)  # (B, T)
            for i in range(B):
                # predicted tokens (positions 1..)
                pred_seq = []
                for tid in pred_ids[i, 1:].tolist():
                    tok = itos.get(tid, PAD_TOKEN)
                    if tok == EOS_TOKEN:
                        break
                    pred_seq.append(tok)
                all_preds.append(pred_seq)
                tgt_seq = []
                for tid in trg_tgt[i].tolist():
                    tok = itos.get(tid, PAD_TOKEN)
                    if tok == EOS_TOKEN:
                        break
                    tgt_seq.append(tok)
                all_targets.append(tgt_seq)

    avg_loss = total_loss / len(dataloader.dataset)
    token_acc, seq_acc, f1 = compute_token_seq_metrics(all_preds, all_targets)
    return avg_loss, token_acc, seq_acc, f1, all_preds, all_targets


def train_and_eval():
    print("Loading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(VAL_CSV)  # external test

    # split train_df into train_main (90%) and train_val (10%)
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = len(train_df)
    n_val = max(1, int(0.10 * n_total))
    train_main_df = train_df.iloc[:-n_val].reset_index(drop=True)
    train_val_df = train_df.iloc[-n_val:].reset_index(drop=True)
    print(f"Train main: {len(train_main_df)}, Train-val (10%): {len(train_val_df)}, Test: {len(test_df)}")

    print("Building vocabulary...")
    stoi, itos = build_vocab_from_dfs([train_main_df, train_val_df, test_df])
    global PAD_IDX
    PAD_IDX = stoi[PAD_TOKEN]

    # datasets and loaders
    train_main_ds = MazeDataset(train_main_df, stoi)
    train_val_ds = MazeDataset(train_val_df, stoi)
    test_ds = MazeDataset(test_df, stoi)

    train_loader = DataLoader(train_main_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_main_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    train_val_loader = DataLoader(train_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(stoi)
    print(f"Vocab size = {vocab_size}, PAD_IDX = {PAD_IDX}")

    encoder = EncoderRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, PAD_IDX).to(DEVICE)
    decoder = DecoderRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, PAD_IDX).to(DEVICE)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum')

    # trackers for 3 splits
    train_losses = []
    train_token_accs = []
    train_seq_accs = []
    train_f1s = []

    val_losses = []
    val_token_accs = []
    val_seq_accs = []
    val_f1s = []

    test_losses = []
    test_token_accs = []
    test_seq_accs = []
    test_f1s = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        encoder.train(); decoder.train()

        running_loss = 0.0
        total_examples = 0

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
        for batch in train_bar:
            src = batch["src"].to(DEVICE)
            src_lens = batch["src_lens"].to(DEVICE)
            trg_in = batch["trg_in"].to(DEVICE)
            trg_tgt = batch["trg_tgt"].to(DEVICE)
            B = src.size(0)

            optimizer.zero_grad()
            enc_outs, enc_hidden = encoder(src, src_lens)
            enc_mask = make_src_mask(src, src_lens).to(DEVICE)
            outputs = decoder(trg_in, enc_hidden, enc_outs, enc_mask, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            # compute loss on min common timesteps
            dec_time_steps = outputs.size(1) - 1
            tgt_time_steps = trg_tgt.size(1)
            min_len = min(dec_time_steps, tgt_time_steps)
            if min_len <= 0:
                continue
            logits = outputs[:, 1:1 + min_len, :].reshape(-1, outputs.size(-1))
            targets = trg_tgt[:, :min_len].reshape(-1)

            loss = criterion(logits, targets) / B
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            running_loss += loss.item() * B
            total_examples += B
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / total_examples if total_examples > 0 else 0.0
        train_losses.append(epoch_train_loss)

        # Evaluate on train_main, train_val (10%), test
        t_loss, t_tok_acc, t_seq_acc, t_f1, _, _ = evaluate_split(encoder, decoder, train_eval_loader, stoi, itos, PAD_IDX, desc="Eval Train")
        v_loss, v_tok_acc, v_seq_acc, v_f1, _, _ = evaluate_split(encoder, decoder, train_val_loader, stoi, itos, PAD_IDX, desc="Eval Train-Val")
        te_loss, te_tok_acc, te_seq_acc, te_f1, _, _ = evaluate_split(encoder, decoder, test_loader, stoi, itos, PAD_IDX, desc="Eval Test")

        train_token_accs.append(t_tok_acc); train_seq_accs.append(t_seq_acc); train_f1s.append(t_f1)
        val_token_accs.append(v_tok_acc); val_seq_accs.append(v_seq_acc); val_f1s.append(v_f1)
        test_token_accs.append(te_tok_acc); test_seq_accs.append(te_seq_acc); test_f1s.append(te_f1)

        val_losses.append(v_loss); test_losses.append(te_loss)

        print(f"Epoch {epoch} summary:")
        print(f"  Train   - Loss: {t_loss:.4f}  TokenAcc: {t_tok_acc:.4f}  SeqAcc: {t_seq_acc:.4f}  F1: {t_f1:.4f}")
        print(f"  TrainVal- Loss: {v_loss:.4f}  TokenAcc: {v_tok_acc:.4f}  SeqAcc: {v_seq_acc:.4f}  F1: {v_f1:.4f}")
        print(f"  Test    - Loss: {te_loss:.4f}  TokenAcc: {te_tok_acc:.4f}  SeqAcc: {te_seq_acc:.4f}  F1: {te_f1:.4f}")

    # Save final model
    torch.save({
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": {"embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS}
    }, MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")

    # Plot curves for three splits
    epochs = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(12, 10))

    # Loss: training loss (epoch-wise), train-val loss (val_losses), test loss (test_losses)
    plt.subplot(4, 1, 1)
    plt.plot(epochs, train_losses, label="train (train-main) loss")
    plt.plot(epochs, val_losses, label="train-val (10%) loss")
    plt.plot(epochs, test_losses, label="test loss")
    plt.ylabel("Loss"); plt.legend()

    # Token acc
    plt.subplot(4, 1, 2)
    plt.plot(epochs, train_token_accs, label="train token acc")
    plt.plot(epochs, val_token_accs, label="train-val token acc")
    plt.plot(epochs, test_token_accs, label="test token acc")
    plt.ylabel("Token Acc"); plt.legend()

    # Seq acc
    plt.subplot(4, 1, 3)
    plt.plot(epochs, train_seq_accs, label="train seq acc")
    plt.plot(epochs, val_seq_accs, label="train-val seq acc")
    plt.plot(epochs, test_seq_accs, label="test seq acc")
    plt.ylabel("Seq Acc"); plt.legend()

    # F1
    plt.subplot(4, 1, 4)
    plt.plot(epochs, train_f1s, label="train F1")
    plt.plot(epochs, val_f1s, label="train-val F1")
    plt.plot(epochs, test_f1s, label="test F1")
    plt.ylabel("F1"); plt.xlabel("Epoch"); plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=200)
    plt.savefig(PLOT_PDF)
    print(f"Saved training curves to {PLOT_PNG} and {PLOT_PDF}")
    plt.show()

    # Visualize 5 random test examples
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    stoi = ckpt["stoi"]; itos = ckpt["itos"]
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval(); decoder.eval()

    indices = random.sample(range(len(test_df)), min(5, len(test_df)))
    for i, idx in enumerate(indices):
        row = test_df.iloc[idx]
        inp_tokens = eval(row["input_sequence"]) if isinstance(row["input_sequence"], str) else row["input_sequence"]
        tgt_tokens = eval(row["output_path"]) if isinstance(row["output_path"], str) else row["output_path"]

        src_idx = torch.tensor([[stoi.get(t, PAD_IDX) for t in inp_tokens]], dtype=torch.long, device=DEVICE)
        src_len = torch.tensor([len(inp_tokens)], dtype=torch.long, device=DEVICE)
        pred = greedy_decode(encoder, decoder, src_idx, src_len, stoi, itos, max_len=MAX_DECODING_LEN)[0]

        final_tokens = inp_tokens.copy()
        final_tokens += ["<PATH START>"] + pred + ["<PATH END>"]
        vis_save = f"vis_test_example_{i+1}.png"
        print(f"Test idx {idx} — GT len {len(tgt_tokens)}, Pred len {len(pred)}")
        plot_maze(final_tokens, title=f"Test idx {idx}", save_path=vis_save)
        print(f"Saved {vis_save}")

if __name__ == "__main__":
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    train_and_eval()