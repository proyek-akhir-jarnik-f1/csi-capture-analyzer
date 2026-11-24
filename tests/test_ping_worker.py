import asyncio
import re
import sys

import pytest

from shared.ping_worker import SubprocessPingWorker


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ICMP backend test")
def test_windows_ping_worker_localhost_high_rate() -> None:
    async def run_check() -> None:
        worker = SubprocessPingWorker()
        await worker.start("127.0.0.1", interval_sec=0.01)
        await asyncio.sleep(0.6)
        await worker.stop()

        stats = worker.stats
        assert stats.running is False
        assert stats.transmitted >= 35
        assert stats.received >= 35
        assert stats.last_rtt_ms is not None
        assert stats.last_error is None
        assert any(re.search(r"icmp_seq=\d+", line) for line in stats.recent_lines)

    asyncio.run(run_check())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ICMP backend test")
def test_windows_ping_worker_invalid_target_sets_error() -> None:
    async def run_check() -> None:
        worker = SubprocessPingWorker()
        await worker.start("256.256.256.256", interval_sec=0.01)

        assert worker.stats.running is False
        assert worker.stats.last_error is not None

    asyncio.run(run_check())
