"""Exploitable-space ("holes in the defense") field, batched in torch.

Port of the ACB paper (opendata-basketball analysis/paper/spatial_holes.md) to the NBA archive:
  arrival time  kinematic: keep current velocity for the reaction time, then accelerate toward q at up to
                A_MAX from the speed component along that line, capped at V_MAX
  uncertainty   T_i ~ N(mu_i, s0^2 + (pe/v_ref)^2 + (0.1 mu_i)^2); NBA optical tracking has no predError:
                pe = 0.5 ft when observed, growing with distance in time to the nearest real sample when interpolated
  control       C_O(q) = P(min_O T < min_D T), Clark moment matching for each team minimum
  value         parametric open-shot map V(q) (NBA line geometry), hoop at (+HOOP_X, 0)
  H = C_O V,  A = sum max(0, H - TAU) over the 1 ft half-court grid
Inputs are already oriented so the offense attacks +x.
"""
import math
import torch

HOOP_X = 41.75
V_MAX, A_MAX, T_REACT = 20.0, 20.0, 0.2
S0, PE, V_REF, K_DIST = 0.15, 0.5, 10.0, 0.1
TAU, OCC_R = 0.5, 5.0
SQ2 = math.sqrt(2.0)


def grid(device, step=1.0):
    xs = torch.arange(0.5, 47.0, step, device=device)
    ys = torch.arange(-24.5, 25.0, step, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    return gx.reshape(-1), gy.reshape(-1)


def value_map(gx, gy):
    d = torch.hypot(gx - HOOP_X, gy)
    corner = gx > 47.0 - 14.0
    three = torch.where(corner, gy.abs() >= 22.0, d >= 23.75)
    two_v = torch.where(d <= 15, 1.35 - 0.5 * d / 15, 0.85 - 0.05 * (d - 15) / 8.75)
    three_v = torch.clamp(1.05 - 0.7 * (d - 23.75) / 11.25, max=1.10)
    three_v = torch.where(d > 35, torch.clamp(0.35 - 0.35 * (d - 35) / 10, min=0.0), three_v)
    return torch.where(three, three_v, two_v).clamp(min=0.0)


def zones(gx, gy):
    d = torch.hypot(gx - HOOP_X, gy)
    corner3 = (gx > 33.0) & (gy.abs() >= 22.0)
    arc3 = (gx <= 33.0) & (d >= 23.75)
    return {'rim': d < 8, 'paint': (gx > 47 - 19) & (gy.abs() < 8), 'corner3': corner3, 'arc3': arc3}


def arrival(pos, vel, gx, gy):
    """pos, vel [B, N, 2] -> mean arrival time [B, N, G]."""
    sp = vel.norm(dim=-1, keepdim=True)
    vel = vel * torch.clamp(V_MAX / sp.clamp(min=1e-6), max=1.0)
    pr = pos + vel * T_REACT
    dx = gx[None, None] - pr[..., :1]; dy = gy[None, None] - pr[..., 1:]
    d = torch.hypot(dx, dy).clamp(min=1e-3)
    u0 = ((vel[..., :1] * dx + vel[..., 1:] * dy) / d).clamp(-V_MAX, V_MAX)
    t1 = (V_MAX - u0) / A_MAX
    s1 = u0 * t1 + 0.5 * A_MAX * t1 ** 2
    t_acc = (-u0 + torch.sqrt(u0 ** 2 + 2 * A_MAX * d)) / A_MAX
    t = torch.where(s1 >= d, t_acc, t1 + (d - s1) / V_MAX)
    return T_REACT + t


def variance(mu, pe=None):
    pe = PE if pe is None else pe[..., None]  # per-player positional error [B, N] (ft)
    return S0 ** 2 + (pe / V_REF) ** 2 + (K_DIST * mu) ** 2


def _phi(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + torch.erf(x / SQ2))


def clark_min(mu, var, mask=None):
    """Min of independent Gaussians along dim 1. mu, var [B, K, G]; mask [K] bool selects members."""
    idx = range(mu.shape[1]) if mask is None else [k for k in range(mu.shape[1]) if mask[k]]
    idx = list(idx)
    m, v = mu[:, idx[0]], var[:, idx[0]]
    for k in idx[1:]:
        m2, v2 = mu[:, k], var[:, k]
        a = torch.sqrt(v + v2)
        al = (m - m2) / a
        P1, P2, ph = _Phi(-al), _Phi(al), _phi(al)
        mean = m * P1 + m2 * P2 - a * ph
        sec = (m * m + v) * P1 + (m2 * m2 + v2) * P2 - (m + m2) * a * ph
        m, v = mean, (sec - mean * mean).clamp(min=1e-6)
    return m, v


def control(mu, var, off_mask=None, def_mask=None):
    mo, vo = clark_min(mu[:, :5], var[:, :5], off_mask)
    md, vd = clark_min(mu[:, 5:], var[:, 5:], def_mask)
    return _Phi((md - mo) / torch.sqrt(vo + vd))


class Field:
    def __init__(self, device):
        self.dev = device
        self.gx, self.gy = grid(device)
        self.V = value_map(self.gx, self.gy)
        self.Z = zones(self.gx, self.gy)

    @torch.no_grad()
    def summarize(self, pos, vel, ball, pe=None, removal=True):
        """pos, vel [B, 10, 2] (offense 0-4, defense 5-9); ball [B, 2]. Returns dict of [B] / [B, 5] tensors."""
        gx, gy, V = self.gx, self.gy, self.V
        mu = arrival(pos, vel, gx, gy); var = variance(mu, pe)
        C = control(mu, var)
        H = C * V[None]
        ex = (H - TAU).clamp(min=0)
        dpl = torch.hypot(gx[None, None] - pos[:, :5, :1], gy[None, None] - pos[:, :5, 1:])  # [B,5,G]
        near = dpl < OCC_R
        occ_any = near.any(1)
        dball = torch.hypot(gx[None] - ball[:, :1], gy[None] - ball[:, 1:])
        out = {
            'A': ex.sum(1), 'H_tot': H.sum(1), 'H_max': H.max(1).values,
            'A_empty': (ex * ~occ_any).sum(1), 'A_occ': (ex * occ_any).sum(1),
            'A_ball': ((H * torch.exp(-(dball - 8).clamp(min=0) / 25) - TAU).clamp(min=0)).sum(1),
            'C_mean': (C * (V[None] > 0)).mean(1),
        }
        for z, m in self.Z.items():
            out[f'A_{z}'] = (ex * m[None]).sum(1)
        # Voronoi baseline: nearest player wins
        dall = torch.hypot(gx[None, None] - pos[..., :1], gy[None, None] - pos[..., 1:])
        cv = (dall[:, :5].min(1).values < dall[:, 5:].min(1).values).float()
        out['A_vor'] = ((cv * V[None] - TAU).clamp(min=0)).sum(1)
        # per attacker: excess space within 5 ft (openness) and own spot value
        out['open_i'] = (ex[:, None] * near).sum(-1)
        cell = ((pos[:, :5, 0].clamp(0, 46.99)).floor().long() * len(torch.arange(-24.5, 25.0, 1.0)) +
                (pos[:, :5, 1] + 25).clamp(0, 49.99).floor().long())
        inside = (pos[:, :5, 0] >= 0) & (pos[:, :5, 0] < 47) & (pos[:, :5, 1].abs() < 25)
        out['H_at_i'] = torch.where(inside, torch.gather(H, 1, cell.clamp(0, H.shape[1] - 1)), torch.zeros_like(inside, dtype=H.dtype))
        if removal:
            full = [True] * 5
            dd, do = [], []
            for j in range(5):
                m = full.copy(); m[j] = False
                dd.append(((control(mu, var, def_mask=m) * V[None] - TAU).clamp(min=0)).sum(1) - out['A'])
                do.append(out['A'] - ((control(mu, var, off_mask=m) * V[None] - TAU).clamp(min=0)).sum(1))
            out['def_cover_j'] = torch.stack(dd, 1)  # space defender j currently denies
            out['off_reach_i'] = torch.stack(do, 1)  # space only attacker i reaches first
        return out

    @torch.no_grad()
    def mc_control(self, pos, vel, n=400, seed=0):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        mu = arrival(pos, vel, self.gx, self.gy); sd = variance(mu).sqrt()
        acc = torch.zeros(mu.shape[0], mu.shape[2], device=self.dev)
        for _ in range(n):
            t = mu + sd * torch.randn(mu.shape, generator=g, device=self.dev)
            acc += (t[:, :5].min(1).values < t[:, 5:].min(1).values).float()
        return acc / n
