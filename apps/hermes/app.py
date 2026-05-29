import os
import signal
import sys
import threading
from typing import TextIO


class _GatewayStdin:
    def __init__(self, stream: TextIO):
        self._stream = stream

    def isatty(self):
        return True

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _watch_parent_eof(parent_stdin: TextIO, done: threading.Event):
    try:
        parent_stdin.buffer.read()
    except (AttributeError, OSError, ValueError):
        pass
    if not done.is_set():
        signal.raise_signal(signal.SIGINT)


def main():
    parent_stdin = sys.stdin
    done = threading.Event()
    watcher = threading.Thread(
        target=_watch_parent_eof,
        args=(parent_stdin, done),
        daemon=True,
        name="desktop-eof-watch",
    )
    watcher.start()

    encoding = parent_stdin.encoding or "utf-8"
    errors = parent_stdin.errors or "strict"
    gateway_stdin = open(os.devnull, encoding=encoding, errors=errors)
    sys.stdin = _GatewayStdin(gateway_stdin)
    sys.argv = ["hermes", "gateway", *sys.argv[1:]]

    try:
        from hermes_cli.main import main as hermes_main

        hermes_main()
    except KeyboardInterrupt:
        return 0
    finally:
        done.set()
        gateway_stdin.close()
        sys.stdin = parent_stdin
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
