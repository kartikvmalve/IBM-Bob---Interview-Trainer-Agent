"""
Entry point for IBM Interview Trainer Agent.
Run with: python run.py
         python run.py --port 8001   (use a different port)
"""
import argparse
import socket

import uvicorn

from app.config import get_settings


def _is_port_free(host: str, port: int) -> bool:
    """Return True if the given TCP port is available to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _find_free_port(host: str, start: int, max_tries: int = 10) -> int:
    """Find the next available port starting from *start*."""
    for port in range(start, start + max_tries):
        if _is_port_free(host, port):
            return port
    raise RuntimeError(
        f"Could not find a free port in range {start}–{start + max_tries - 1}. "
        "Stop any other running servers and try again."
    )


def main() -> None:
    cfg = get_settings()

    parser = argparse.ArgumentParser(description="IBM Interview Trainer Agent")
    parser.add_argument("--host", default=cfg.app_host, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=cfg.app_port, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    host: str = args.host
    port: int = args.port

    if not _is_port_free(host, port):
        alt_port = _find_free_port(host, port + 1)
        print(
            f"\n[WARNING] Port {port} is already in use.\n"
            f"   Starting on port {alt_port} instead.\n"
            f"   http://localhost:{alt_port}\n"
        )
        port = alt_port

    print(f"\n*** IBM Interview Trainer Agent ***")
    print(f"    UI  ->  http://localhost:{port}")
    print(f"    API ->  http://localhost:{port}/docs\n")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
