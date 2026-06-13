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


@unittest.skipUnless(
    torch.backends.mps.is_available(), "MPS backend not available"
)
class TestMPSThreading(unittest.TestCase):
    # Each test runs for SOAK_SECONDS. Override with env var for longer runs.
    SOAK_SECONDS = float(os.environ.get("MPS_SOAK_SECONDS", "10"))

    def _run_threads(self, targets, duration):
        """Run each callable in `targets` on its own thread for `duration`
        seconds. Re-raise the first exception any thread saw.
        """
        stop = threading.Event()
        errors = []
        threads = []

        def wrap(fn):
            def runner():
                try:
                    while not stop.is_set():
                        fn()
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
                    stop.set()

            return runner

        for fn in targets:
            t = threading.Thread(target=wrap(fn), daemon=True)
            t.start()
            threads.append(t)

        time.sleep(duration)
        stop.set()
        for t in threads:
            t.join(timeout=5.0)

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
            assert isinstance(y, float)

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

        self._run_threads(
            [worker_train, worker_render], self.SOAK_SECONDS
        )


if __name__ == "__main__":
    unittest.main()
