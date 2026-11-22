"""Off-ball credit by ghosting (PC GPU).

For a moment s (EPV v3 sample, kind != 2, every other 5 Hz frame) and each of the 10 players i:
  ghost   the MAE (mae_long, no identity) hides player i for the whole 2 s window and redraws his path from the
          other nine players and the ball: where a typical player would have been in that context
  dV[s,i] = EPV(actual window) - EPV(window with player i replaced by his ghost)
EPV for sample s comes from the cross-fit model of its fold (epv3_cf{k}.pt, never saw the game), raw scale.
Attackers: positive dV = his actual movement made the possession better than a typical player's would have.
Defenders: negative dV is good defense (credit = -dV, applied downstream).
Caveat: the other nine players react to the real player, so the ghost inherits some of his context; dV measures
value from moving differently than the context predicts, not total value.
Output: out4/ghost.npz  s [n] sample index into v3.npz, dV [n,10] f16, gd [n,10] f16 (feet between player and
ghost at "now"), epv [n] f16
  python epv_ghost.py [stride]
"""
import os, sys, time
import numpy as np
import torch
from epv3_train import V3, Data, DEV, EPV_DIR, SCALE
import jepa

OUT = os.path.join(EPV_DIR, 'out4')


@torch.no_grad()
def ghost(mae, x, i):
    """x [B,10,11,4] normalised; returns x with agent i's 10 frames replaced by the MAE reconstruction."""
    B = len(x)
    keep = torch.ones(10, 11, dtype=torch.bool, device=DEV); keep[:, i] = False; keep = keep.flatten()
    ctx = mae.encode(x, None, keep)
    ti = torch.arange(10, device=DEV); ai = torch.full((10,), i, device=DEV)
    q = mae.emb.query(B, ti, ai, None) + mae.mask_tok
    pr = mae.head(mae.pred(torch.cat([ctx, q], 1))[:, ctx.shape[1]:])  # [B, 10, 4]
    xg = x.clone(); xg[:, :, i] = pr.to(x.dtype)
    return xg


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    os.makedirs(OUT, exist_ok=True)
    D = Data(os.path.join(EPV_DIR, 'v3.npz'))
    cf = np.load(os.path.join(EPV_DIR, 'out3', 'epv3_cf.npz'))
    fold = cf['fold']
    fi = D.fi.cpu().numpy()
    sel = np.flatnonzero((D.kind != 2) & (fi % stride == 0))
    print('moments', len(sel), flush=True)
    mae = jepa.Net(1, 'mae_long').to(DEV)
    mae.load_state_dict(torch.load(os.path.join(r'C:\Users\LukeB\dev\court-model\results\jepa', 'mae_long.pt'), map_location=DEV)); mae.eval()
    dV = np.zeros((len(sel), 10), np.float16); gd = np.zeros((len(sel), 10), np.float16); ev = np.zeros(len(sel), np.float16)
    pos = {s: j for j, s in enumerate(sel)}
    t0 = time.time(); done = 0
    for k in range(5):
        m = V3(False).to(DEV); m.load_state_dict(torch.load(os.path.join(EPV_DIR, 'out3', f'epv3_cf{k}.pt'), map_location=DEV)); m.eval()
        idx = sel[fold[sel] == k]
        for s0 in range(0, len(idx), 2048):
            b = idx[s0:s0 + 2048]; rows = np.array([pos[s] for s in b])
            x, g = D.batch(b)
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                base = m(x, g).float()
                for i in range(10):
                    xg = ghost(mae, x, i)
                    v = m(xg, g).float()
                    dV[rows, i] = (base - v).cpu().numpy()
                    gd[rows, i] = (torch.linalg.norm((xg[:, 9, i, :2] - x[:, 9, i, :2]).float() * SCALE[:2], dim=-1)).cpu().numpy()
            ev[rows] = base.cpu().numpy()
            done += len(b)
            if (s0 // 2048) % 50 == 0:
                print(f'fold {k} {done}/{len(sel)} ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(os.path.join(OUT, 'ghost.npz'), s=sel, dV=dV, gd=gd, epv=ev)
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
