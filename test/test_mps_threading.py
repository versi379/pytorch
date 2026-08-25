# Owner(s): ["module: mps"]
"""
Threading regression tests for the MPS backend.

These tests intentionally race ATen MPS ops against torch.mps.synchronize()
and against each other. Before MPS thread-safety lockdown landed, they would
crash within seconds with assertions like:
  - "commit an already committed command buffer"
  - "command encoder already encoding"
After the lockdown they should run cleanly for the full duration.

Run only on MPS:
  pytest test/test_mps_threading.py -v -s
"""

import os
import threading
import time
import unittest

import torch
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS backend not available")
class TestMPSThreading(TestCase):
    # Each test runs for SOAK_SECONDS. Override with env var for longer runs.
    SOAK_SECONDS = float(os.environ.get("MPS_SOAK_SECONDS", "10"))

    def _run_threads(self, targets, duration):
        """Run each callable in `targets` on its own thread for `duration`
        seconds. Re-raise the first exception any thread saw.

        If a worker fails early, `stop.wait` returns immediately so the test
        finishes fast instead of soaking the full duration.

        If a worker is still alive after a 5 s join, fail loudly: a hung
        thread mid-MPS-op would otherwise leak into the next test and
        contaminate its state (fresh tensors, fresh optimizer, etc.).
        """
        stop = threading.Event()
        errors = []
        threads = []

        def wrap(fn):
            def runner():
                try:
                    while not stop.is_set():
                        fn()
                except BaseException as e:
                    errors.append(e)
                    stop.set()

            return runner

        for fn in targets:
            t = threading.Thread(target=wrap(fn), daemon=True)
            t.start()
            threads.append(t)

        stop.wait(duration)
        stop.set()
        for t in threads:
            t.join(timeout=5.0)

        hung = [t for t in threads if t.is_alive()]
        if hung:
            self.fail(
                f"{len(hung)} worker thread(s) did not exit within 5s; "
                f"possible deadlock"
            )

        if errors:
            raise errors[0]

    def test_concurrent_synchronize_with_matmul(self):
        """Thread A spams matmul.item(); thread B spams torch.mps.synchronize().

        Pre-fix: crashes within seconds with "command buffer already committed".
        Post-fix: clean for SOAK_SECONDS.
        """
        x = torch.randn(512, 512, device="mps")
        w = torch.randn(512, 512, device="mps")

        def worker_matmul():
            y = (x @ w).sum().item()
            if not isinstance(y, float):
                raise AssertionError(f"expected float, got {type(y)}")

        def worker_sync():
            torch.mps.synchronize()

        self._run_threads(
            [worker_matmul, worker_sync, worker_matmul, worker_sync],
            self.SOAK_SECONDS,
        )

    def test_concurrent_item_and_synchronize(self):
        """Reproduces nerfstudio crash signature: loss.item() racing with
        torch.mps.synchronize() from another thread.
        """
        x = torch.randn(64, 64, device="mps", requires_grad=True)
        w = torch.randn(64, 64, device="mps")

        def worker_item():
            loss = ((x @ w) ** 2).sum()
            _ = loss.item()

        def worker_sync():
            torch.mps.synchronize()

        self._run_threads(
            [worker_item, worker_sync, worker_item, worker_sync],
            self.SOAK_SECONDS,
        )

    def test_train_render_pattern(self):
        """Mirror nerfstudio: thread A runs forward+backward+step on a small
        net; thread B repeatedly snapshots tensors with .cpu() like a viewer
        render thread.
        """
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
        ).to("mps")
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        x = torch.randn(32, 128, device="mps")
        target = torch.randn(32, 128, device="mps")

        def worker_train():
            opt.zero_grad()
            loss = ((net(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
            _ = loss.item()

        def worker_render():
            with torch.no_grad():
                _ = net(x).cpu()

        self._run_threads([worker_train, worker_render], self.SOAK_SECONDS)

    def test_single_thread_synchronize_interleaved_with_ops(self):
        """Smoke: interleaving MPS ops with explicit synchronize() must not crash or hang.

        Not a true re-entrancy test — Python cannot construct a re-entrant call into
        the serial queue; that requires a C++ extension (Phase 2 work).
        """
        x = torch.randn(64, 64, device="mps")
        w = torch.randn(64, 64, device="mps")
        if x.device.type != "mps":
            raise AssertionError("test requires MPS-backed tensors")
        deadline = time.time() + min(2.0, self.SOAK_SECONDS)
        while time.time() < deadline:
            for _ in range(50):
                y = x @ w
            torch.mps.synchronize()
            _ = y.sum().item()
            torch.mps.synchronize()

    def test_eye_concurrent_with_matmul(self):
        """Reproduces a race where torch.eye() (and other ops sharing the
        Eye-style pattern) capture the cached MPSStream compute encoder
        OUTSIDE the dispatch_sync block, then encode to it inside. Between
        the off-queue capture and the dispatch_sync, another thread can
        run torch.mps.synchronize() or any commit path, which ends and
        releases the cached encoder. The first thread's captured pointer
        then dangles.

        Pre-fix observed manifestations (depending on what GCD recycled
        the address as):
          - NSInvalidArgumentException: `-[OS_dispatch_mach setComputePipelineState:]:
            unrecognized selector sent to instance ...`
          - Metal assertion: `tryCoalescingPreviousComputeCommandEncoderWithConfig:
            nextEncoderClass:` ... `A command encoder is already encoding to
            this command buffer`

        Post-fix the captured pointer is constructed inside the dispatch_sync,
        so no other thread can interleave between capture and use; the test
        must run cleanly for SOAK_SECONDS.
        """

        def worker_eye():
            for _ in range(8):
                _ = torch.eye(100, 100, device="mps")
                _ = torch.eye(50, 50, device="mps")

        def worker_sync():
            torch.mps.synchronize()

        def worker_matmul():
            a = torch.randn(64, 64, device="mps")
            b = torch.randn(64, 64, device="mps")
            for _ in range(20):
                c = a @ b
            _ = c.sum().item()

        self._run_threads(
            [
                worker_eye,
                worker_sync,
                worker_eye,
                worker_sync,
                worker_matmul,
                worker_matmul,
            ],
            self.SOAK_SECONDS,
        )

    def test_reentrancy_probe_under_load(self):
        """probe_mps_reentrancy() must be safe when called concurrently from
        multiple threads. Each thread dispatch_syncs to the serial queue; GCD
        serialises them. Verifies no crash or deadlock under concurrent load.
        """
        _ext_path = os.path.join(
            os.path.dirname(__file__), "cpp_extensions", "mps_extension.mm"
        )
        module = torch.utils.cpp_extension.load(
            name="torch_test_mps_extension",
            sources=[_ext_path],
            verbose=False,
            keep_intermediates=False,
        )
        _ = torch.zeros(1, device="mps")

        def worker():
            module.probe_mps_reentrancy()

        self._run_threads([worker, worker, worker], self.SOAK_SECONDS)


if __name__ == "__main__":
    run_tests()
