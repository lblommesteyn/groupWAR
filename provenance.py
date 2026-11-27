"""Provenance block saved with every corrected run: code revision, script hash, data file fingerprints, time.
On the PC (no git checkout) the revision is read from REVISION, written by the Mac before copying code over."""
import hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path

from masked_paths import ROOT


def _rev():
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    f = ROOT / 'REVISION'
    return f.read_text().strip() if f.exists() else 'unknown'


def _fp(path):
    p = ROOT / path
    if not p.exists():
        return dict(path=path, exists=False)
    st = p.stat(); h = hashlib.sha256()
    with open(p, 'rb') as f:
        h.update(f.read(1 << 20))          # first MiB + size + mtime: cheap and changes when the file does
    return dict(path=path, bytes=st.st_size, mtime=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec='seconds'),
                sha256_first_mib=h.hexdigest()[:16])


def provenance(script, data_paths):
    s = Path(script)
    return dict(revision=_rev(), script=s.name, script_sha256=hashlib.sha256(s.read_bytes()).hexdigest()[:16],
                run_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), data=[_fp(d) for d in data_paths])
