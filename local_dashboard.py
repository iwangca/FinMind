#!/usr/bin/env python3
import json
import sys
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import subprocess


BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "local_dashboard.html"


def _run_command(args):
    proc = subprocess.run(
        args,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _end_date(mode, custom):
    today = date.today()
    if mode == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    if mode == "custom" and custom:
        return custom
    return today.strftime("%Y-%m-%d")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self):
        if not HTML_PATH.exists():
            self.send_error(404, "local_dashboard.html not found")
            return
        html = HTML_PATH.read_text(encoding="utf-8")
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_html()

        if parsed.path == "/api/fetch":
            params = parse_qs(parsed.query)
            stock_file = params.get("stock_file", ["stock_ids_futures.txt"])[0]
            mode = params.get("mode", ["today"])[0]
            custom_date = params.get("date", [""])[0]
            end_date = _end_date(mode, custom_date)
            chunk_months = params.get("chunk_months", ["6"])[0]
            datasets = params.get(
                "datasets",
                [
                    "price,price_adj,per_pbr,institutional,margin,shareholding,holding_shares_per,securities_lending"
                ],
            )[0]
            args = [
                sys.executable,
                "finmind_fetch_daily.py",
                "--stock-file",
                stock_file,
                "--end-date",
                end_date,
                "--chunk-months",
                chunk_months,
                "--datasets",
                datasets,
                "--skip-on-error",
            ]
            result = _run_command(args)
            return self._send_json({"ok": True, "result": result})

        if parsed.path == "/api/doji":
            params = parse_qs(parsed.query)
            target_date = params.get("date", [""])[0]
            max_ticks = params.get("max_ticks", ["4"])[0]
            data_dir = params.get("data_dir", ["data/finmind/price"])[0]
            args = [
                sys.executable,
                "find_doji_candidates.py",
                "--data-dir",
                data_dir,
                "--max-ticks",
                max_ticks,
            ]
            if target_date:
                args += ["--date", target_date]
            result = _run_command(args)
            return self._send_json({"ok": True, "result": result})

        self.send_error(404, "Not found")


def main():
    host = "127.0.0.1"
    port = 8000
    server = HTTPServer((host, port), Handler)
    print(f"Local dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
