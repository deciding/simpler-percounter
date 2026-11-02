"""Compare two qwen_dump.py output directories, the first taken as reference.

usage: python qwen_compare.py <reference_dir> <candidate_dir>

Uses the qwen3 test's own tolerance (rtol=5e-2, atol=1e-1) on the output and
the written KV rows, and reports exact equality, the largest absolute
difference and how many elements exceed the tolerance. The whole-pool sums are
compared as a relative difference: a stray write elsewhere moves them.
"""
import sys

import torch

RTOL = 5e-2
ATOL = 1e-1
ref_dir, cand_dir = sys.argv[1], sys.argv[2]
ok = True

if not torch.equal(torch.load(f"{ref_dir}/slot_mapping.pt"), torch.load(f"{cand_dir}/slot_mapping.pt")):
    print("slot_mapping differs: the two runs used different fixtures")
    ok = False

for name in ("out", "k_written", "v_written"):
    ref = torch.load(f"{ref_dir}/{name}.pt").float().reshape(-1)
    cand = torch.load(f"{cand_dir}/{name}.pt").float().reshape(-1)
    if ref.shape != cand.shape:
        print(f"{name}: shape mismatch {tuple(ref.shape)} vs {tuple(cand.shape)}")
        ok = False
        continue
    diff = (ref - cand).abs()
    bad = (diff > ATOL + RTOL * ref.abs()).sum().item()
    nan = (torch.isnan(ref) != torch.isnan(cand)).sum().item()
    print(
        f"{name}: elems={ref.numel()} exact={torch.equal(ref, cand)} max_abs_diff={diff.max().item():.6g} "
        f"mean_abs_diff={diff.mean().item():.6g} beyond_tol={bad} nan_mismatch={nan} "
        f"ref_absmax={ref.abs().max().item():.6g}"
    )
    if bad or nan:
        ok = False

for name in ("k_sums", "v_sums"):
    ref = torch.load(f"{ref_dir}/{name}.pt")
    cand = torch.load(f"{cand_dir}/{name}.pt")
    rel = ((ref - cand).abs() / ref.abs().clamp_min(1.0)).max().item()
    print(f"{name}: layers={ref.numel()} exact={torch.equal(ref, cand)} max_rel_diff={rel:.3g}")
    if rel > 1e-3:
        ok = False

print("MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
