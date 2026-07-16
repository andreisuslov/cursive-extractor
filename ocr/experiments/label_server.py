"""Remote labelling server: serves the personal site, the letter labeller, and its JSON API.

Deployed on avid-m1 behind the cloudflared tunnel as andreisuslov.com:

  /                     -> static personal site (SITE_DIR)
  /cursive              -> the labeller (TOOL_FILE, letter_label.html with remote mode)
  /cursive/api/pages    -> page list + progress          (worker or admin token)
  /cursive/api/next     -> first uncompleted page id     (?after=<id> to continue past one)
  /cursive/api/page/<id>   GET latest doc / POST a save  (saves are versioned, never destructive)
  /cursive/api/download/<id> -> latest doc               (admin only; for ingest)

Storage under DATA_DIR:
  pages/<id>.json    seed docs ({page, words}) uploaded by the admin (scp or POST w/ admin token)
  current/<id>.json  latest save (what workers load and the admin downloads)
  saves/<id>/<utc>_<worker>.json  every save, append-only (bad edits can always be rolled back)
  tokens.json        {"admin": "<token>", "workers": {"<token>": "worker-name"}}

Auth: every /cursive/api request needs ?t=<token> (worker or admin). Tokens are long random
hex strings carried in the URL a worker receives; page ids are sanitized against traversal.

Zero third-party dependencies (stdlib http.server) so the host only needs python3.

Run:      python -m ocr.experiments.label_server --site SITE --tool TOOL.html --data DATA
Selftest: python -m ocr.experiments.label_server --selftest
"""

from __future__ import annotations

import argparse
import datetime
import hmac
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_BODY = 64 * 1024 * 1024  # embedded page images are a few MB; leave headroom
PAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".gif": "image/gif",
    ".woff2": "font/woff2",
}


class Store:
    """Versioned page-document store on disk. All writes go through a lock."""

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir
        self.pages = data_dir / "pages"
        self.current = data_dir / "current"
        self.saves = data_dir / "saves"
        for d in (self.pages, self.current, self.saves):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def page_ids(self) -> list[str]:
        return sorted(p.stem for p in self.pages.glob("*.json"))

    def load(self, page_id: str) -> dict | None:
        for d in (self.current, self.pages):
            f = d / f"{page_id}.json"
            if f.exists():
                return json.loads(f.read_text())
        return None

    def summary(self, page_id: str) -> dict:
        doc = self.load(page_id) or {}
        words = doc.get("words") or []
        done = sum(1 for w in words if w.get("done") or w.get("skip"))
        cur = self.current / f"{page_id}.json"
        return {
            "id": page_id,
            "words": len(words),
            "done": done,
            "completed": bool(doc.get("completed")),
            "updated": (
                datetime.datetime.fromtimestamp(cur.stat().st_mtime, tz=datetime.UTC).isoformat(
                    timespec="seconds"
                )
                if cur.exists()
                else None
            ),
        }

    def save(self, page_id: str, doc: dict, worker: str) -> str:
        """Merge a worker save onto the stored doc and persist a version + the new current."""
        if not isinstance(doc.get("words"), list) or not doc["words"]:
            raise ValueError("save must contain a non-empty words list")
        with self.lock:
            prev = self.load(page_id)
            if prev is None:
                raise KeyError(page_id)
            if not doc.get("page"):  # page image unchanged -> client omits it to save bandwidth
                doc["page"] = prev.get("page")
            doc["savedBy"] = worker
            ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            vdir = self.saves / page_id
            vdir.mkdir(parents=True, exist_ok=True)
            blob = json.dumps(doc)
            safe_worker = re.sub(r"[^A-Za-z0-9_-]", "_", worker)[:40]
            (vdir / f"{ts}_{safe_worker}.json").write_text(blob)
            tmp = self.current / f".{page_id}.tmp"
            tmp.write_text(blob)
            tmp.replace(self.current / f"{page_id}.json")
            return ts

    def seed(self, page_id: str, doc: dict) -> None:
        if not isinstance(doc.get("words"), list) or not doc.get("page"):
            raise ValueError("seed must be a self-contained {page, words} doc")
        with self.lock:
            (self.pages / f"{page_id}.json").write_text(json.dumps(doc))


class App:
    def __init__(self, site_dir: Path, tool_file: Path, store: Store, tokens: dict) -> None:
        self.site_dir = site_dir
        self.tool_file = tool_file
        self.store = store
        self.admin = tokens.get("admin", "")
        self.workers: dict[str, str] = tokens.get("workers", {})

    def identify(self, token: str) -> str | None:
        """Return 'admin', a worker name, or None."""
        if self.admin and hmac.compare_digest(token, self.admin):
            return "admin"
        for t, name in self.workers.items():
            if hmac.compare_digest(token, t):
                return name
        return None


class Handler(BaseHTTPRequestHandler):
    app: App  # set by serve()
    protocol_version = "HTTP/1.1"

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control", "no-store" if ctype == "application/json" else "public, max-age=300"
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, code: int, msg: str) -> None:
        self._json(code, {"error": msg})

    def _auth(self) -> str | None:
        q = parse_qs(urlparse(self.path).query)
        who = self.app.identify(q.get("t", [""])[0])
        if who is None:
            self._err(403, "missing or invalid token")
        return who

    def log_message(self, fmt: str, *args) -> None:  # quiet default logger; one line per request
        line = re.sub(r"([?&]t=)[^&\s\"]+", r"\1***", fmt % args)  # never log auth tokens
        print(
            f"{datetime.datetime.now(tz=datetime.UTC).isoformat(timespec='seconds')} "
            f"{self.address_string()} {line}",
            flush=True,
        )

    # -- static site -----------------------------------------------------
    def _static(self, url_path: str) -> None:
        rel = url_path.lstrip("/") or "index.html"
        f = (self.app.site_dir / rel).resolve()
        if not str(f).startswith(str(self.app.site_dir.resolve())):
            return self._err(404, "not found")
        if f.is_dir():
            f = f / "index.html"
        if not f.is_file():
            return self._err(404, "not found")
        self._send(200, f.read_bytes(), MIME.get(f.suffix.lower(), "application/octet-stream"))

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = url.path
        if path in ("/cursive", "/cursive/"):
            return self._send(200, self.app.tool_file.read_bytes(), MIME[".html"])
        if path.startswith("/cursive/api/"):
            return self._api_get(path)
        return self._static(path)

    def _api_get(self, path: str) -> None:
        who = self._auth()
        if who is None:
            return
        store = self.app.store
        if path == "/cursive/api/pages":
            return self._json(
                200, {"who": who, "pages": [store.summary(i) for i in store.page_ids()]}
            )
        if path == "/cursive/api/next":
            q = parse_qs(urlparse(self.path).query)
            after = q.get("after", [""])[0]
            ids = store.page_ids()
            if after in ids:  # continue scanning past the page just finished
                ids = ids[ids.index(after) + 1 :] + ids[: ids.index(after)]
            for pid in ids:
                if not store.summary(pid)["completed"]:
                    return self._json(200, {"id": pid})
            return self._json(200, {"id": None})
        m = re.match(r"^/cursive/api/(page|download)/([^/]+)$", path)
        if m:
            kind, pid = m.group(1), m.group(2)
            if not PAGE_ID_RE.match(pid):
                return self._err(400, "bad page id")
            if kind == "download" and who != "admin":
                return self._err(403, "admin only")
            doc = store.load(pid)
            if doc is None:
                return self._err(404, "no such page")
            return self._json(200, doc)
        return self._err(404, "unknown endpoint")

    def do_POST(self) -> None:
        who = self._auth()
        if who is None:
            return
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._err(413, "bad content length")
        try:
            doc = json.loads(self.rfile.read(n))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._err(400, "invalid JSON")
        m = re.match(r"^/cursive/api/(page|seed)/([^/]+)$", path)
        if not m or not PAGE_ID_RE.match(m.group(2)):
            return self._err(404, "unknown endpoint")
        kind, pid = m.group(1), m.group(2)
        try:
            if kind == "seed":
                if who != "admin":
                    return self._err(403, "admin only")
                self.app.store.seed(pid, doc)
                return self._json(200, {"ok": True, "id": pid})
            ts = self.app.store.save(pid, doc, who)
            return self._json(200, {"ok": True, "saved": ts})
        except KeyError:
            return self._err(404, "no such page")
        except ValueError as e:
            return self._err(400, str(e))


def serve(site: Path, tool: Path, data: Path, tokens_file: Path, port: int) -> ThreadingHTTPServer:
    tokens = json.loads(tokens_file.read_text())
    Handler.app = App(site, tool, Store(data), tokens)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return httpd


def _selftest() -> None:
    """Spin the server on an ephemeral port and exercise every endpoint."""
    import tempfile
    import urllib.request

    def req(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "site").mkdir()
        (root / "site" / "index.html").write_text("<h1>site</h1>")
        (root / "tool.html").write_text("<h1>tool</h1>")
        (root / "tokens.json").write_text(
            json.dumps({"admin": "AAA", "workers": {"WWW": "worker-1"}})
        )
        httpd = serve(root / "site", root / "tool.html", root / "data", root / "tokens.json", 0)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        seed = {
            "page": "data:image/png;base64,x",
            "words": [{"text": "hi", "box": [0, 0, 9, 9], "cuts": []}],
        }
        assert req("POST", "/cursive/api/seed/p1?t=WWW", seed)[0] == 403, "worker must not seed"
        assert req("POST", "/cursive/api/seed/p1?t=AAA", seed)[0] == 200
        assert req("POST", "/cursive/api/seed/..evil?t=AAA", seed)[0] == 404, (
            "traversal id rejected"
        )
        assert req("GET", "/cursive/api/pages")[0] == 403, "no token -> 403"
        code, pages = req("GET", "/cursive/api/pages?t=WWW")
        assert (
            code == 200 and pages["pages"][0]["id"] == "p1" and not pages["pages"][0]["completed"]
        )
        assert req("GET", "/cursive/api/next?t=WWW")[1]["id"] == "p1"
        code, doc = req("GET", "/cursive/api/page/p1?t=WWW")
        assert code == 200 and doc["words"][0]["text"] == "hi"
        save = {
            "words": [{"text": "hi", "box": [0, 0, 9, 9], "cuts": [], "done": True}],
            "completed": True,
        }
        code, r = req("POST", "/cursive/api/page/p1?t=WWW", save)
        assert code == 200 and r["ok"]
        code, doc = req("GET", "/cursive/api/page/p1?t=WWW")
        assert doc["page"] == seed["page"], "omitted page must be merged from the stored doc"
        assert doc["savedBy"] == "worker-1" and doc["completed"]
        assert req("GET", "/cursive/api/next?t=WWW")[1]["id"] is None, "all pages completed"
        assert req("GET", "/cursive/api/download/p1?t=WWW")[0] == 403
        assert req("GET", "/cursive/api/download/p1?t=AAA")[0] == 200
        assert req("POST", "/cursive/api/page/nope?t=WWW", save)[0] == 404
        vdir = root / "data" / "saves" / "p1"
        assert len(list(vdir.glob("*.json"))) == 1, "save must be versioned"

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert b"site" in resp.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/cursive") as resp:
            assert b"tool" in resp.read()
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/../etc/passwd")
            raise AssertionError("traversal must not succeed")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        httpd.shutdown()
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, help="static site root")
    ap.add_argument("--tool", type=Path, help="labeller HTML file served at /cursive")
    ap.add_argument("--data", type=Path, help="data dir (pages/current/saves)")
    ap.add_argument("--tokens", type=Path, help="tokens.json")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not all([args.site, args.tool, args.data, args.tokens]):
        ap.error("--site, --tool, --data, --tokens are required (or --selftest)")
    httpd = serve(args.site, args.tool, args.data, args.tokens, args.port)
    print(
        f"labelling server on 127.0.0.1:{args.port} (site={args.site}, data={args.data})",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
