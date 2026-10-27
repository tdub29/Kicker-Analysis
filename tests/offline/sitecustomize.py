"""Refuse outbound network connections in a self-check subprocess.

The README promises every `--self-check` runs offline and needs no data. Without
a guard that promise is only checked on machines with no network, and CI has
network, so a self-check that quietly started fetching nflverse would still pass.
tests/test_self_checks.py puts this directory on PYTHONPATH, Python imports
`sitecustomize` at startup, and from then on any TCP connect (IPv4 or IPv6)
raises. Unix sockets and socketpairs are left alone so local process pools
still work.

This file is not a test and is never imported by the package itself.
"""
import socket

MARKER = "network access blocked: --self-check must run offline"

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_INET = {socket.AF_INET, getattr(socket, "AF_INET6", socket.AF_INET)}


def _guard(sock: socket.socket, address) -> None:
    if sock.family in _INET:
        raise OSError(f"{MARKER} (tried {address!r})")


def _blocked_connect(self, address):
    _guard(self, address)
    return _connect(self, address)


def _blocked_connect_ex(self, address):
    _guard(self, address)
    return _connect_ex(self, address)


socket.socket.connect = _blocked_connect
socket.socket.connect_ex = _blocked_connect_ex
