"""
NBA GCN Training — V2 (Bigger Embeddings)
Identical to nba_train.py except:
  - NODE_IN_DIM = 20 (was 13)
  - Loads nba_embeddings_v2.pkl
  - Saves to models/v2/

Run after nba_process_v2.py.
"""

import pickle
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nba_stackel import (
    GCN_DeepSet_AntiSym_Invariant,
    PeriodDataset,
    projector,
    N_PLAYERS,
    N_HALF,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "v2"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (V2 changes marked) ──────────────────────────────────────
NODE_IN_DIM  = 20      # V2: was 13, now 20 (added shot zones + assists)
GCN_HIDDEN   = 128
GCN_LAYERS   = 6
DS_PHI       = 128
DS_RHO       = 128
VECTOR_SIZE  = 128
DROPOUT      = 0.05
BATCH_SIZE   = 256
EPOCHS       = 30
LR           = 1e-4
N_FOLDS      = 6
SAVE_EVERY   = 10


def margin_to_prob(margin, scale=10.0):
    return torch.sigmoid(torch.tensor(margin, dtype=torch.float32) / scale)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.zeros_(m.bias)


def collate_fn(batch):
    X, A, hm, am, y = zip(*batch)
    return (
        torch.stack(X),
        torch.stack(A),
        torch.stack(hm),
        torch.stack(am),
        torch.stack(y),
    )


def build_node_features(period_player_dicts, embeddings):
    feat_cols = [c for c in embeddings.columns if c not in ["player_id", "year"]]
    emb_lookup = {}
    for _, row in embeddings.iterrows():
        emb_lookup[(int(row["player_id"]), int(row["year"]))] = row[feat_cols].values.astype(np.float32)

    node_features = []
    for pid_dict in tqdm(period_player_dicts, desc="Building V2 node features"):
        mat = np.zeros((N_PLAYERS, len(feat_cols)), dtype=np.float32)
        for idx, pid in pid_dict.items():
            for yr in [2023, 2022]:
                key = (int(pid), yr)
                if key in emb_lookup:
                    mat[idx] = emb_lookup[key]
                    break
        node_features.append(mat)

    return node_features


def main():
    print("Loading V2 data...")
    with open(DATA_DIR / "nba_embeddings_v2.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open(DATA_DIR / "nba_period_matrices.pkl", "rb") as f:
        period_matrices = pickle.load(f)
    with open(DATA_DIR / "nba_period_player_dicts.pkl", "rb") as f:
        period_player_dicts = pickle.load(f)
    with open(DATA_DIR / "nba_targets.pkl", "rb") as f:
        targets_raw = pickle.load(f)

    print(f"  {len(period_matrices)} games, {len(embeddings)} player-seasons")

    targets = [margin_to_prob(t).item() for t in targets_raw]
    targets = [t if not np.isnan(t) else 0.5 for t in targets]

    print("Projecting adjacency matrices...")
    adj = []
    for mat in tqdm(period_matrices):
        t = torch.tensor(mat.to_numpy(), dtype=torch.float32)
        adj.append(projector.project(t))

    vec = build_node_features(period_player_dicts, embeddings)

    valid = [i for i, v in enumerate(vec) if v.shape == (N_PLAYERS, NODE_IN_DIM)]
    print(f"  {len(valid)} / {len(vec)} valid samples (V2 NODE_IN_DIM={NODE_IN_DIM})")
    adj     = [adj[i] for i in valid]
    vec     = [vec[i] for i in valid]
    targets = [targets[i] for i in valid]

    vec_arr = np.array(vec)
    tgt_arr = np.array(targets)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(kf.split(tgt_arr))
    criterion = nn.MSELoss()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}  |  train={len(train_idx)}  test={len(test_idx)}")
        print('='*60)

        train_ds = PeriodDataset(
            vec_arr[train_idx],
            [adj[i] for i in train_idx],
            tgt_arr[train_idx],
        )
        test_ds = PeriodDataset(
            vec_arr[test_idx],
            [adj[i] for i in test_idx],
            tgt_arr[test_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        torch.manual_seed(42 + fold_idx)
        model = GCN_DeepSet_AntiSym_Invariant(
            node_in_dim=NODE_IN_DIM,   # V2: 20
            gcn_hidden=GCN_HIDDEN,
            gcn_layers=GCN_LAYERS,
            deepset_phi_dim=DS_PHI,
            deepset_rho_dim=DS_RHO,
            n_nodes=N_PLAYERS,
            vector_size=VECTOR_SIZE,
            n_half=N_HALF,
            dropout=DROPOUT,
        ).to(device)
        model.apply(init_weights)

        optimizer = torch.optim.Adamax(model.parameters(), lr=LR,
                                       weight_decay=1e-4, betas=(0.8, 0.98), eps=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for X, A, home_mask, away_mask, y_true in train_loader:
                X         = X.to(device)
                A         = A.to(device)
                home_mask = home_mask.to(device)
                y_true    = y_true.to(device)

                y_pred = model(A, X, home_mask).unsqueeze(-1)
                loss   = criterion(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() * X.size(0)

            scheduler.step()
            train_loss = total_loss / len(train_loader.dataset)

            model.eval()
            eval_loss = 0.0
            correct   = 0
            with torch.no_grad():
                for X, A, home_mask, away_mask, y_true in test_loader:
                    X         = X.to(device)
                    A         = A.to(device)
                    home_mask = home_mask.to(device)
                    y_true    = y_true.to(device)
                    y_pred    = model(A, X, home_mask).unsqueeze(-1)
                    eval_loss += criterion(y_pred, y_true).item() * X.size(0)
                    correct   += ((y_pred > 0.5) == (y_true > 0.5)).sum().item()

            eval_loss /= len(test_loader.dataset)
            acc = correct / len(test_loader.dataset)
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.5f} | "
                  f"val_loss={eval_loss:.5f} | val_acc={acc:.4f}")

            if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
                ckpt = MODELS_DIR / f"nba_model_v2_{fold_idx}_{epoch}.pth"
                torch.save(model.state_dict(), ckpt)
                print(f"  Saved {ckpt.name}")

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nV2 training complete.")


if __name__ == "__main__":
    main()
