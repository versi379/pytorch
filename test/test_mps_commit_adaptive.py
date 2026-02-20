"""
Validates the COMMIT_ADAPTIVE op-count trigger (kAdaptiveOpThreshold=32).

With the fix, MPSStream commits every 32 ops so the GPU starts executing early
while the CPU keeps encoding. Commit count scales linearly with N.

Without the fix, nothing commits until synchronize(). Commit count stays at
0-1 regardless of N.

Pass/fail is based on commit count. Wall time is informational.

Usage:
  python test_mps_commit_adaptive.py --mode with_fix
  python test_mps_commit_adaptive.py --mode without_fix

Outputs:
  test/test_mps_commit_adaptive/results_with_fix.txt
  test/test_mps_commit_adaptive/results_without_fix.txt
"""

import argparse
import os
import subprocess
import time

import torch
import torch.mps

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_mps_commit_adaptive")

OPS = {
    "relu":       (lambda x: torch.relu(x), 512),
    "matmul_256": (lambda x: x @ x,          256),
    "matmul_512": (lambda x: x @ x,          512),
}

OP_THRESHOLD = 32
N_TRIALS = 10
N_VALUES = [16, 32, 64, 128, 256]


def run_n_ops(n: int, op_fn, size: int) -> torch.Tensor:
    x = torch.randn(size, size, device="mps")
    for _ in range(n):
        x = op_fn(x)
    return x


def measure(n_ops: int, op_fn, size: int) -> tuple[float, int]:
    """Returns (avg_wall_time_s, total_commits_across_all_trials)."""
    pid = os.getpid()
    log_proc = subprocess.Popen(
        [
            "log", "stream",
            "--predicate", f'processID == {pid} AND eventMessage CONTAINS "[MPS commit]"',
            "--style", "compact",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    time.sleep(0.3)

    wall_times = []
    for _ in range(N_TRIALS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        x = run_n_ops(n_ops, op_fn, size)
        torch.mps.synchronize()
        wall_times.append(time.perf_counter() - t0)
        _ = x

    log_proc.terminate()
    raw, _ = log_proc.communicate()
    n_commits = sum(1 for line in raw.splitlines() if "[MPS commit]" in line)

    wall_times.sort()
    trimmed = wall_times[1:-1] if len(wall_times) > 4 else wall_times
    return sum(trimmed) / len(trimmed), n_commits


def check(mode: str, n: int, n_commits: int) -> tuple[str, bool]:
    """Returns (status_string, passed)."""
    expected_per_trial = n // OP_THRESHOLD
    if expected_per_trial == 0:
        return "skip", True
    if mode == "with_fix":
        ok = abs(n_commits - expected_per_trial * N_TRIALS) <= N_TRIALS
    else:
        ok = n_commits <= 1
    return ("pass" if ok else "FAIL"), ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["with_fix", "without_fix"], required=True)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("MPS not available")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"mode: {args.mode}  |  commit every {OP_THRESHOLD} ops  |  {N_TRIALS} trials")
    print(f"PyTorch: {torch.__file__}")
    print()

    print("warming up...", end=" ", flush=True)
    for _, (op_fn, size) in OPS.items():
        run_n_ops(64, op_fn, size)
    torch.mps.synchronize()
    print("done.")
    print()

    HDR   = f"  {'N':>4}   {'wall ms':>8}   {'commits':>9}   {'expected':>9}   {'':>4}"
    SEP   = "  " + "-" * 44

    all_failures = []
    all_rows = []  # (op_name, size, n, wall_ms, n_commits, expected, status)

    for op_name, (op_fn, size) in OPS.items():
        print(f"{op_name}  ({size}x{size})")
        print(HDR)
        print(SEP)

        for n in N_VALUES:
            wall_s, n_commits = measure(n, op_fn, size)
            wall_ms = wall_s * 1000
            expected = (n // OP_THRESHOLD) * N_TRIALS
            status, ok = check(args.mode, n, n_commits)

            print(f"  {n:>4}   {wall_ms:>8.2f}   {n_commits:>9}   {expected:>9}   {status}")
            all_rows.append((op_name, size, n, wall_ms, n_commits, expected, status))

            if not ok:
                all_failures.append(f"{op_name} n={n}: got {n_commits} commits, expected ~{expected}")

        print()

    if all_failures:
        print("FAILED:")
        for msg in all_failures:
            print(f"  {msg}")

    # --- build results file ---
    if args.mode == "with_fix":
        lat_desc = (
            "Wall time per N ops (trimmed mean over 10 trials).\n"
            "With the trigger active the GPU starts early, so wall time grows slowly with N."
        )
        cmt_desc = (
            "Commit counts observed over 10 trials via [MPS commit] NSLog.\n"
            "The trigger fires every 32 ops, so expected = floor(N/32) * 10.\n"
            "pass = observed within ±10 of expected."
        )
    else:
        lat_desc = (
            "Wall time per N ops (trimmed mean over 10 trials).\n"
            "Without the trigger the GPU only starts at synchronize(), so wall time = encode + execute."
        )
        cmt_desc = (
            "Commit counts observed over 10 trials via [MPS commit] NSLog.\n"
            "The trigger is disabled, so no periodic commits should fire.\n"
            "pass = 0 or 1 commits observed (1 is warm-up noise)."
        )

    file_lines = [
        f"mode: {args.mode}  |  PyTorch: {torch.__file__}",
        "",
        # --- commits section ---
        "COMMITS",
        cmt_desc,
        "",
        f"  {'op':>12}   {'N':>4}   {'commits':>9}   {'expected':>9}   {'':>4}",
        "  " + "-" * 44,
    ]
    for op_name, size, n, _, n_commits, expected, status in all_rows:
        file_lines.append(f"  {op_name:>12}   {n:>4}   {n_commits:>9}   {expected:>9}   {status}")

    file_lines += [
        "",
        # --- latency section ---
        "LATENCY",
        lat_desc,
        "",
        f"  {'op':>12}   {'N':>4}   {'wall ms':>8}",
        "  " + "-" * 30,
    ]
    for op_name, size, n, wall_ms, _, _, _ in all_rows:
        file_lines.append(f"  {op_name:>12}   {n:>4}   {wall_ms:>8.2f}")

    if all_failures:
        file_lines += ["", "FAILED:"] + [f"  {m}" for m in all_failures]

    file_lines.append("")

    out_path = os.path.join(OUT_DIR, f"results_{args.mode}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(file_lines) + "\n")

    print()
    print(f"wrote {out_path}")

    if all_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
