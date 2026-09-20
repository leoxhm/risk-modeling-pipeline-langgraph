"""零依赖启动独立训练监控网页，不连接 OpenCode 或建模流程。"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    """在本地启动静态文件服务器。"""
    parser = argparse.ArgumentParser(description="Serve the standalone modeling monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    class Handler(SimpleHTTPRequestHandler):
        """把请求根目录固定在 monitor_web，避免暴露项目其他文件。"""

        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(root), **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"训练监控网页: http://{args.host}:{args.port}/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n监控网页已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
