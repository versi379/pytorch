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
                except BaseException as e:  # noqa: BLE001
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

    def test_synchronize_from_within_extension_dispatch(self):
        """An extension that wraps its work in dispatch_sync(stream->queue())
        and then calls torch.mps.synchronize() must not deadlock.

        We approximate this from Python by issuing many ATen MPS ops
        (which go through dispatch_sync_with_rethrow internally) interleaved
        with synchronize calls on the same thread. If the re-entrancy
        detection is broken, the second synchronize would deadlock.
        """
        x = torch.randn(64, 64, device="mps")
        w = torch.randn(64, 64, device="mps")
        import time
        deadline = time.time() + min(2.0, self.SOAK_SECONDS)
        while time.time() < deadline:
            for _ in range(50):
                y = x @ w
            torch.mps.synchronize()
            _ = y.sum().item()
            torch.mps.synchronize()


if __name__ == "__main__":
    unittest.main()
