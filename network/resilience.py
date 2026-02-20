"""
Network Resilience — retry logic, timeouts, and connection management
for cross-machine deployments where network failures are expected.
"""

import functools
import time
from typing import Callable, TypeVar

import grpc

T = TypeVar("T")

# Default timeouts (seconds)
CONNECT_TIMEOUT = 10.0
FORWARD_TIMEOUT = 60.0     # per-token forward pass (includes all hops)
SHARD_TRANSFER_TIMEOUT = 600.0  # shard download can be slow on WAN

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0   # seconds
RETRY_BACKOFF_MAX = 10.0   # cap on backoff

# Transient gRPC errors worth retrying
RETRYABLE_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}


def with_retry(max_retries: int = MAX_RETRIES,
               backoff_base: float = RETRY_BACKOFF_BASE,
               backoff_max: float = RETRY_BACKOFF_MAX,
               retryable_codes: set = None):
    """
    Decorator that retries a gRPC call on transient failures.

    Uses exponential backoff with jitter.
    """
    codes = retryable_codes or RETRYABLE_CODES

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except grpc.RpcError as e:
                    last_error = e
                    if e.code() not in codes:
                        # Non-retryable error
                        raise
                    if attempt < max_retries:
                        delay = min(backoff_base * (2 ** attempt), backoff_max)
                        print(f"  [Retry] Attempt {attempt + 1}/{max_retries} failed "
                              f"({e.code().name}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


def create_resilient_channel(address: str, grpc_options: list = None) -> grpc.Channel:
    """
    Create a gRPC channel with keepalive and retry settings suitable
    for cross-machine / WAN deployment.
    """
    options = list(grpc_options or [])

    # Add keepalive settings for WAN resilience
    options.extend([
        # Send keepalive pings every 30 seconds
        ("grpc.keepalive_time_ms", 30000),
        # Wait 10 seconds for keepalive response before considering dead
        ("grpc.keepalive_timeout_ms", 10000),
        # Allow keepalive even when there are no active RPCs
        ("grpc.keepalive_permit_without_calls", 1),
        # After a connection failure, wait before reconnecting
        ("grpc.initial_reconnect_backoff_ms", 1000),
        ("grpc.max_reconnect_backoff_ms", 10000),
        # Enable retries at the gRPC level
        ("grpc.enable_retries", 1),
    ])

    return grpc.insecure_channel(address, options=options)


def wait_for_server(address: str, timeout: float = CONNECT_TIMEOUT,
                    grpc_options: list = None) -> bool:
    """
    Wait for a gRPC server to become reachable.

    Returns True if the server is reachable within the timeout, False otherwise.
    """
    channel = create_resilient_channel(address, grpc_options)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        return True
    except grpc.FutureTimeoutError:
        return False
    finally:
        channel.close()
