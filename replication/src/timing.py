"""RAW v2: explicit primary clock. Never monkeypatch Python or rewrite old data."""
import time

def raw_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

def raw_seconds():
    return raw_ns()/1e9
