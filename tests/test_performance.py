import pytest

from y2karaoke.utils.performance import PerformanceMonitor, timing_decorator


def test_timing_decorator_returns_result():
    @timing_decorator
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


def test_timing_decorator_propagates_exception():
    @timing_decorator
    def boom():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        boom()


def test_performance_monitor_success():
    with PerformanceMonitor("op") as monitor:
        assert monitor.operation_name == "op"


def test_performance_monitor_failure():
    with pytest.raises(RuntimeError):
        with PerformanceMonitor("op"):
            raise RuntimeError("fail")
