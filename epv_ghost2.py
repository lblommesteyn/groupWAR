"""Gravity and identity ghosts (PC GPU). Same moments as epv_ghost.py (kind != 2, every other 5 Hz frame).

gravity  the identity-blind MAE (mae_long) hides defender j for the whole 2 s window and redraws where a typical
         defender would be, given the other nine players and the ball. Stores the ghost's position at "now";
         epv_gravity.py compares the real defenders' distances to each attacker with the ghosts' distances
         (how much more the defense leans toward a player than his spot and context call for).
id ghost mae_id hides player i and redraws his path twice: with his own identity embedding and with the mean
         embedding of his movement group (k-means of the identity embeddings). Both are reconstructions, so the
         blur is the same on both sides and the comparison stays within his kind of player:
           dVid[s, i] = EPV(window with his own-identity ghost) - EPV(window with his group-mean ghost)
         Players without a learned embedding (not in the MAE player list) get NaN.
Note: mae_id saw games before 2026-04-01 in pretraining (positions only, no outcomes).
Output: out4/ghost2.npz  s, gpos [n,5,2] f16 (ghost defender positions at now, feet, normalised court frame),
        dVid [n,10] f16, gdid [n,10] f16 (feet between the two id ghosts at now), group [545] int, pids [545]
  python epv_ghost2.py [stride]
"""
import os, sys, time
import numpy as np
import torch
from epv3_train import V3, Data, DEV, EPV_DIR, SCALE
from epv_ghost import ghost
import jepa

OUT = os.path.join(EPV_DIR, 'out4')
J = r'C:\Users\LukeB\dev\court-model\results\jepa'
K_GROUPS = 8


@torch.no_grad()
def id_ghost(mae, x, ids, i, emb_i):
    """hide agent i, redraw with identity embedding emb_i [B, D] for his query tokens (others keep their own)"""
    B = len(x)
    keep = torch.ones(10, 11, dtype=torch.bool, device=DEV); keep[:, i] = False; keep = keep.flatten()
    ctx = mae.encode(x, ids, keep)
    ti = torch.arange(10, device=DEV); ai = torch.full((10,), i, device=DEV)
    q = mae.emb.query(B, ti, ai, None) + emb_i[:, None] + mae.mask_tok
    pr = mae.head(mae.pred(torch.cat([ctx, q], 1))[:, ctx.shape[1]:])
    xg = x.clone(); xg[:, :, i] = pr.to(x.dtype)
    return xg


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    D = Data(os.path.join(EPV_DIR, 'v3.npz'))
    fold = np.load(os.path.join(EPV_DIR, 'out3', 'epv3_cf.npz'))['fold']
    fi = D.fi.cpu().numpy()
    sel = np.flatnonzero((D.kind != 2) & (fi % stride == 0))
    print('moments', len(sel), flush=True)
    mae = jepa.Net(1, 'mae_long').to(DEV)
    mae.load_state_dict(torch.load(os.path.join(J, 'mae_long.pt'), map_location=DEV)); mae.eval()
    pids = np.load(os.path.join(J, 'windows.npz'))['pids']
    maei = jepa.Net(len(pids), 'mae_id').to(DEV)
    maei.load_state_dict(torch.load(os.path.join(J, 'mae_id.pt'), map_location=DEV)); maei.eval()
    W = maei.emb.pid.weight.detach()  # [545 + 1, D], last row = ball slot
    E = W[:-1].float().cpu().numpy()
    live = np.linalg.norm(E, axis=1) > 1e-3
    from sklearn.cluster import KMeans
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    grp = np.full(len(E), -1)
    grp[live] = KMeans(K_GROUPS, n_init=10, random_state=0).fit_predict(En[live])
    gmean = torch.stack([W[:-1][torch.from_numpy(grp == k).to(DEV)].mean(0) for k in range(K_GROUPS)])  # [K, D]
    print('group sizes', np.bincount(grp[live]), flush=True)
    pidx = {int(p): j for j, p in enumerate(pids)}
    nid = len(pids)  # unknown players -> zero embedding (index of ball slot is nid; use a zero row instead)
    Wz = torch.cat([W[:-1], torch.zeros(1, W.shape[1], device=DEV), W[-1:]])  # unknown -> row nid (zeros), ball -> nid+1
    maei.emb.pid = torch.nn.Embedding.from_pretrained(Wz, freeze=True)
    grp_t = torch.from_numpy(np.r_[grp, -1]).to(DEV)
    gmean_z = torch.cat([gmean, torch.zeros(1, W.shape[1], device=DEV)])  # group -1 -> zeros
    ids_all = np.vectorize(lambda p: pidx.get(int(p), nid))(D.meta[:, 3:13]).astype(np.int64)

    n = len(sel)
    gpos = np.zeros((n, 5, 2), np.float16); dVid = np.full((n, 10), np.nan, np.float16); gdid = np.full((n, 10), np.nan, np.float16)
    pos = np.empty(len(D.kind), np.int64); pos[sel] = np.arange(n)
    t0 = time.time(); done = 0
    for k in range(5):
        m = V3(False).to(DEV); m.load_state_dict(torch.load(os.path.join(EPV_DIR, 'out3', f'epv3_cf{k}.pt'), map_location=DEV)); m.eval()
        idx = sel[fold[sel] == k]
        for s0 in range(0, len(idx), 2048):
            b = idx[s0:s0 + 2048]; rows = pos[b]
            x, g = D.batch(b)
            ids = torch.from_numpy(ids_all[b]).to(DEV)
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                for j in range(5, 10):
                    xg = ghost(mae, x, j)
                    gpos[rows, j - 5] = (xg[:, 9, j, :2].float() * SCALE[:2].to(DEV)).cpu().numpy()
                for i in range(10):
                    known = ids[:, i] < nid
                    own = maei.emb.pid(ids[:, i])
                    mean = gmean_z[grp_t[ids[:, i].clamp(max=nid)].clamp(min=-1)]
                    xa = id_ghost(maei, x, ids, i, own); xb = id_ghost(maei, x, ids, i, mean)
                    d = (m(xa, g).float() - m(xb, g).float()).cpu().numpy()
                    dd = torch.linalg.norm((xa[:, 9, i, :2] - xb[:, 9, i, :2]).float() * SCALE[:2].to(DEV), dim=-1).cpu().numpy()
                    kn = known.cpu().numpy()
                    dVid[rows[kn], i] = d[kn]; gdid[rows[kn], i] = dd[kn]
            done += len(b)
            if (s0 // 2048) % 50 == 0:
                print(f'fold {k} {done}/{n} ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(os.path.join(OUT, 'ghost2.npz'), s=sel, gpos=gpos, dVid=dVid, gdid=gdid, group=grp, pids=pids)
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
