"""Remote labelling server: serves the personal site, the letter labeller, and its JSON API.

Deployed on avid-m1 behind the cloudflared tunnel as andreisuslov.com:

  /                     -> static personal site (SITE_DIR)
  /cursive              -> the labeller (TOOL_FILE) for a logged-in user, else the login page
  /cursive/login        -> login page; /cursive/admin -> admin console (admin session only)
  /cursive/api/...      -> JSON API (session cookie, or legacy ?t=<admin token> acts as admin)

Storage under DATA_DIR:
  pages/<id>.json    seed docs ({page, words}) uploaded by the admin (scp or POST w/ admin token)
  current/<id>.json  latest save (what workers load and the admin downloads)
  saves/<id>/<utc>_<worker>.json  every save, append-only (bad edits can always be rolled back)
  tokens.json        {"admin": "<token>"} ("workers" entries are DEPRECATED and ignored)
  users.json         {name: {hash, salt, role, active, created_iso}} pbkdf2-sha256, 600k iters
  sessions.json      {sid: {user, expires_iso}} pruned on access, 30-day lifetime
  assignments.json   {page_id: username}
  activity.json      {username: {ts_iso, page, cursor}} updated on every save

Auth: username+password login -> HttpOnly "cursive_session" cookie scoped to /cursive. Workers
only see/save the pages assigned to them; admins see everything. Login is rate limited per
client IP (5 failures / 10 min). The legacy ?t=<admin token> still authenticates as admin for
API scripting. Page ids are sanitized against traversal; logs never contain tokens, passwords,
or session ids. The login/admin HTML files live next to the --tool file.

Zero third-party dependencies (stdlib http.server) so the host only needs python3.

Run:      python -m ocr.experiments.label_server --site SITE --tool TOOL.html --data DATA
Useradd:  python -m ocr.experiments.label_server --data DATA --useradd NAME --password PW
Selftest: python -m ocr.experiments.label_server --selftest
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_BODY = 64 * 1024 * 1024  # embedded page images are a few MB; leave headroom
PAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
USER_RE = re.compile(r"^[a-z0-9_-]{2,32}$")
PBKDF2_ITERS = 600_000
SESSION_TTL = 30 * 24 * 3600  # seconds
RATE_WINDOW, RATE_MAX = 600.0, 5  # login failures per client IP
COOKIE = "cursive_session"
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


def _iso(dt: datetime.datetime | None = None) -> str:
    return (dt or datetime.datetime.now(tz=datetime.UTC)).isoformat(timespec="seconds")


def _redact(line: str) -> str:
    """Strip auth tokens and session ids out of anything headed for the log."""
    line = re.sub(r"([?&]t=)[^&\s\"]+", r"\1***", line)
    return re.sub(rf"({COOKIE}=)[^;\s\"]+", r"\1***", line)


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


class Auth:
    """Users, sessions, assignments, activity — small JSON files under DATA, created on demand.

    `clock` is only used by the login rate limiter and is injectable so the selftest can
    advance time without sleeping (defaults to time.monotonic).
    """

    def __init__(self, data_dir: Path, clock=time.monotonic) -> None:
        self.root = data_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.clock = clock
        self.fails: dict[str, list[float]] = {}  # client ip -> monotonic times of login failures

    def _read(self, name: str) -> dict:
        f = self.root / f"{name}.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def _write(self, name: str, obj: dict) -> None:
        tmp = self.root / f".{name}.tmp"
        tmp.write_text(json.dumps(obj))
        tmp.replace(self.root / f"{name}.json")

    @staticmethod
    def _hash(password: str, salt_hex: str) -> str:
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERS)
        return dk.hex()

    @staticmethod
    def _prune(sessions: dict) -> dict:
        now = _iso()
        return {sid: s for sid, s in sessions.items() if s.get("expires_iso", "") > now}

    def upsert_user(
        self, name: str, password: str | None, role: str | None, active: bool | None = None
    ) -> dict:
        if not USER_RE.match(name or ""):
            raise ValueError("bad username (want ^[a-z0-9_-]{2,32}$)")
        if role not in (None, "admin", "worker"):
            raise ValueError("role must be admin or worker")
        with self.lock:
            users = self._read("users")
            u = users.get(name)
            if u is None:
                if not password:
                    raise ValueError("password required for a new user")
                u = {"hash": "", "salt": "", "role": role or "worker", "active": True,
                     "created_iso": _iso()}  # fmt: skip
            if password:
                u["salt"] = secrets.token_bytes(16).hex()
                u["hash"] = self._hash(password, u["salt"])
            if role:
                u["role"] = role
            if active is not None:
                u["active"] = bool(active)
            users[name] = u
            self._write("users", users)
            if not u["active"]:  # disabling a user invalidates their sessions
                live = {k: s for k, s in self._read("sessions").items() if s["user"] != name}
                self._write("sessions", live)
            return u

    def users(self) -> dict:
        with self.lock:
            return self._read("users")

    def login(self, name: str, password: str) -> tuple[str, str] | None:
        """Verify credentials; return (session id, role) or None."""
        with self.lock:
            u = self._read("users").get(name)
            if not u or not u.get("active"):
                self._hash(password, "00" * 16)  # constant work: no username-existence timing leak
                return None
            if not hmac.compare_digest(self._hash(password, u["salt"]), u["hash"]):
                return None
            sid = secrets.token_hex(32)
            exp = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(seconds=SESSION_TTL)
            sessions = self._prune(self._read("sessions"))
            sessions[sid] = {"user": name, "expires_iso": _iso(exp)}
            self._write("sessions", sessions)
            return sid, u["role"]

    def session(self, sid: str) -> tuple[str, str] | None:
        """Return (user, role) for a live session of an active user, else None."""
        if not sid:
            return None
        with self.lock:
            sessions = self._prune(self._read("sessions"))
            self._write("sessions", sessions)
            s = sessions.get(sid)
            u = self._read("users").get(s["user"]) if s else None
            if not s or not u or not u.get("active"):
                return None
            return s["user"], u["role"]

    def logout(self, sid: str) -> None:
        with self.lock:
            sessions = self._read("sessions")
            if sessions.pop(sid, None) is not None:
                self._write("sessions", sessions)

    def assignments(self) -> dict:
        with self.lock:
            return self._read("assignments")

    def assign(self, user: str | None, page_ids: list[str]) -> None:
        with self.lock:
            a = self._read("assignments")
            for pid in page_ids:
                if user is None:
                    a.pop(pid, None)
                else:
                    a[pid] = user
            self._write("assignments", a)

    def activity(self) -> dict:
        with self.lock:
            return self._read("activity")

    def touch(self, user: str, page: str, cursor) -> None:
        with self.lock:
            act = self._read("activity")
            act[user] = {"ts_iso": _iso(), "page": page, "cursor": cursor}
            self._write("activity", act)

    # -- login rate limit (sliding window of failures per client ip) ------
    def throttled(self, ip: str) -> bool:
        with self.lock:
            now = self.clock()
            self.fails[ip] = [t for t in self.fails.get(ip, []) if now - t < RATE_WINDOW]
            return len(self.fails[ip]) >= RATE_MAX

    def record_fail(self, ip: str) -> None:
        with self.lock:
            self.fails.setdefault(ip, []).append(self.clock())


class App:
    def __init__(
        self, site_dir: Path, tool_file: Path, store: Store, tokens: dict, auth: Auth
    ) -> None:
        self.site_dir = site_dir
        self.tool_file = tool_file
        self.store = store
        self.auth = auth
        self.admin = tokens.get("admin", "")  # tokens.json "workers" entries are deprecated

    def html(self, name: str) -> Path:
        return self.tool_file.parent / name


class Handler(BaseHTTPRequestHandler):
    app: App  # set by serve()
    protocol_version = "HTTP/1.1"

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        hdrs = dict(extra or {})
        hdrs.setdefault(
            "Cache-Control", "no-store" if ctype == "application/json" else "public, max-age=300"
        )
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict, extra: dict | None = None) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json", extra)

    def _err(self, code: int, msg: str) -> None:
        self._json(code, {"error": msg})

    def _html(self, name: str) -> None:
        f = self.app.html(name)
        if not f.is_file():
            return self._err(404, "not found")
        self._send(200, f.read_bytes(), MIME[".html"], {"Cache-Control": "no-store"})

    def _cookie_sid(self) -> str:
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE:
                return v
        return ""

    def _whoami(self) -> tuple[str, str, str] | None:
        """(name, role, via) where via is 'token' or 'cookie', else None."""
        tok = parse_qs(urlparse(self.path).query).get("t", [""])[0]
        if tok and self.app.admin and hmac.compare_digest(tok, self.app.admin):
            return "admin", "admin", "token"
        got = self.app.auth.session(self._cookie_sid())
        return (got[0], got[1], "cookie") if got else None

    def _client_ip(self) -> str:
        return self.headers.get("Cf-Connecting-Ip") or self.client_address[0]

    def _origin_ok(self) -> bool:
        """If an Origin header is present its host must match the Host header's host."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return bool(host) and (urlparse(origin).hostname or "").lower() == host

    def _cookie_hdr(self, sid: str, max_age: int) -> dict:
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "").lower() == "https" else ""
        return {
            "Set-Cookie": (
                f"{COOKIE}={sid}; HttpOnly; SameSite=Lax; Path=/cursive; Max-Age={max_age}{secure}"
            )
        }

    def log_message(self, fmt: str, *args) -> None:  # quiet default logger; one line per request
        print(
            f"{_iso()} {self.address_string()} {_redact(fmt % args)}",  # never log secrets
            flush=True,
        )

    # -- static site -----------------------------------------------------
    def _static(self, url_path: str) -> None:
        rel = url_path.lstrip("/") or "index.html"
        f = (self.app.site_dir / rel).resolve()
        if not f.is_relative_to(self.app.site_dir.resolve()):
            return self._err(404, "not found")
        if f.is_dir():
            f = f / "index.html"
        if not f.is_file():
            return self._err(404, "not found")
        self._send(200, f.read_bytes(), MIME.get(f.suffix.lower(), "application/octet-stream"))

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/cursive", "/cursive/"):
            if self._whoami() is None:
                return self._html("label_login.html")
            return self._send(200, self.app.tool_file.read_bytes(), MIME[".html"],
                              {"Cache-Control": "no-store"})  # fmt: skip
        if path == "/cursive/login":
            return self._html("label_login.html")
        if path == "/cursive/admin":
            who = self._whoami()
            ok = who is not None and who[1] == "admin"
            return self._html("label_admin.html" if ok else "label_login.html")
        if path.startswith("/cursive/api/"):
            return self._api_get(path)
        return self._static(path)

    def _scope_ids(self, name: str, role: str, assigned: dict) -> list[str]:
        ids = self.app.store.page_ids()
        return ids if role == "admin" else [i for i in ids if assigned.get(i) == name]

    def _api_get(self, path: str) -> None:
        who = self._whoami()
        if who is None:
            return self._err(401, "not logged in")
        name, role, _via = who
        if path == "/cursive/api/me":
            return self._json(200, {"name": name, "role": role})
        store, auth = self.app.store, self.app.auth
        assigned = auth.assignments()
        if path == "/cursive/api/pages":
            ids = self._scope_ids(name, role, assigned)
            pages = [{**store.summary(i), "assigned_to": assigned.get(i)} for i in ids]
            return self._json(200, {"who": name, "role": role, "pages": pages})
        if path == "/cursive/api/next":
            after = parse_qs(urlparse(self.path).query).get("after", [""])[0]
            ids = self._scope_ids(name, role, assigned)
            if after in ids:  # continue scanning past the page just finished
                ids = ids[ids.index(after) + 1 :] + ids[: ids.index(after)]
            for pid in ids:
                if not store.summary(pid)["completed"]:
                    return self._json(200, {"id": pid})
            return self._json(200, {"id": None})
        if path == "/cursive/api/admin/overview":
            if role != "admin":
                return self._err(403, "admin only")
            return self._json(200, self._overview())
        m = re.match(r"^/cursive/api/(page|download)/([^/]+)$", path)
        if m:
            kind, pid = m.group(1), m.group(2)
            if not PAGE_ID_RE.match(pid):
                return self._err(400, "bad page id")
            if kind == "download" and role != "admin":
                return self._err(403, "admin only")
            if kind == "page" and role != "admin" and assigned.get(pid) != name:
                return self._err(403, "page not assigned to you")
            doc = store.load(pid)
            if doc is None:
                return self._err(404, "no such page")
            return self._json(200, doc)
        return self._err(404, "unknown endpoint")

    def _overview(self) -> dict:
        store, auth = self.app.store, self.app.auth
        assigned, activity = auth.assignments(), auth.activity()
        summaries = {i: store.summary(i) for i in store.page_ids()}
        users = []
        for uname, u in sorted(auth.users().items()):
            mine = sorted(i for i, w in assigned.items() if w == uname and i in summaries)
            prog = {
                "words_done": sum(summaries[i]["done"] for i in mine),
                "words_total": sum(summaries[i]["words"] for i in mine),
                "pages_done": sum(1 for i in mine if summaries[i]["completed"]),
                "pages_total": len(mine),
            }
            act = activity.get(uname)
            users.append({
                "name": uname, "role": u["role"], "active": u["active"],
                "created": u["created_iso"], "assigned": mine, "progress": prog,
                "last_active": act["ts_iso"] if act else None,
                "current": {"page": act["page"], "cursor": act.get("cursor")} if act else None,
            })  # fmt: skip
        pages = [{**summaries[i], "assigned_to": assigned.get(i)} for i in sorted(summaries)]
        return {"users": users, "pages": pages}

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._err(400, "bad content length")
        if n < 0 or n > MAX_BODY:
            return self._err(413, "bad content length")
        try:
            doc = json.loads(self.rfile.read(n)) if n else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._err(400, "invalid JSON")
        if not isinstance(doc, dict):
            return self._err(400, "body must be a JSON object")
        if path == "/cursive/api/login":
            return self._login(doc)
        who = self._whoami()
        if who is None:
            return self._err(401, "not logged in")
        name, role, via = who
        if via == "cookie" and not self._origin_ok():
            return self._err(403, "origin mismatch")
        if path == "/cursive/api/logout":
            self.app.auth.logout(self._cookie_sid())
            return self._json(200, {"ok": True}, self._cookie_hdr("", 0))
        if path == "/cursive/api/admin/users":
            if role != "admin":
                return self._err(403, "admin only")
            try:
                u = self.app.auth.upsert_user(
                    doc.get("name", ""), doc.get("password"), doc.get("role"), doc.get("active")
                )
            except ValueError as e:
                return self._err(400, str(e))
            return self._json(
                200, {"ok": True, "name": doc["name"], "role": u["role"], "active": u["active"]}
            )
        if path == "/cursive/api/admin/assign":
            if role != "admin":
                return self._err(403, "admin only")
            user, pids = doc.get("user"), doc.get("page_ids")
            if not isinstance(pids, list) or not all(
                isinstance(p, str) and PAGE_ID_RE.match(p) for p in pids
            ):
                return self._err(400, "page_ids must be a list of page ids")
            if user is not None and user not in self.app.auth.users():
                return self._err(400, "no such user")
            self.app.auth.assign(user, pids)
            return self._json(200, {"ok": True})
        m = re.match(r"^/cursive/api/(page|seed)/([^/]+)$", path)
        if not m or not PAGE_ID_RE.match(m.group(2)):
            return self._err(404, "unknown endpoint")
        kind, pid = m.group(1), m.group(2)
        try:
            if kind == "seed":
                if role != "admin":
                    return self._err(403, "admin only")
                self.app.store.seed(pid, doc)
                return self._json(200, {"ok": True, "id": pid})
            if role != "admin" and self.app.auth.assignments().get(pid) != name:
                return self._err(403, "page not assigned to you")
            ts = self.app.store.save(pid, doc, name)
            self.app.auth.touch(name, pid, doc.get("cursor"))
            return self._json(200, {"ok": True, "saved": ts})
        except KeyError:
            return self._err(404, "no such page")
        except ValueError as e:
            return self._err(400, str(e))

    def _login(self, doc: dict) -> None:
        ip, auth = self._client_ip(), self.app.auth
        if auth.throttled(ip):
            return self._err(429, "too many attempts, try again later")
        name = str(doc.get("username") or "")
        got = auth.login(name, str(doc.get("password") or ""))
        if got is None:
            auth.record_fail(ip)
            return self._err(403, "bad credentials")
        sid, role = got
        hdr = self._cookie_hdr(sid, SESSION_TTL)
        return self._json(200, {"ok": True, "name": name, "role": role}, hdr)


def serve(
    site: Path, tool: Path, data: Path, tokens_file: Path, port: int, clock=time.monotonic
) -> ThreadingHTTPServer:
    tokens = json.loads(tokens_file.read_text())
    Handler.app = App(site, tool, Store(data), tokens, Auth(data, clock=clock))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return httpd


def _selftest() -> None:
    """Spin the server on an ephemeral port and exercise every endpoint."""
    import contextlib
    import io
    import tempfile
    import urllib.request

    def req(
        method: str, path: str, body: dict | None = None, headers: dict | None = None
    ) -> tuple[int, dict, dict]:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
        for k, v in (headers or {}).items():
            r.add_header(k, v)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read()), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), dict(e.headers)

    def page(path: str, headers: dict | None = None) -> bytes:
        r = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
        for k, v in (headers or {}).items():
            r.add_header(k, v)
        with urllib.request.urlopen(r) as resp:
            return resp.read()

    def ck(sid: str) -> dict:
        return {"Cookie": f"{COOKIE}={sid}"}

    def login(user: str, pw: str, headers: dict | None = None) -> tuple[int, str | None]:
        code, _, hdrs = req("POST", "/cursive/api/login", {"username": user, "password": pw},
                            headers)  # fmt: skip
        if code != 200:
            return code, None
        return code, re.search(rf"{COOKIE}=([0-9a-f]{{64}})", hdrs["Set-Cookie"]).group(1)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "site").mkdir()
        (root / "site" / "index.html").write_text("<h1>site</h1>")
        (root / "tool.html").write_text("<h1>tool</h1>")
        (root / "label_login.html").write_text("<h1>loginpage</h1>")
        (root / "label_admin.html").write_text("<h1>adminpage</h1>")
        (root / "tokens.json").write_text(
            json.dumps({"admin": "AAA", "workers": {"WWW": "worker-1"}})  # workers = deprecated
        )
        tnow = [0.0]
        httpd = serve(root / "site", root / "tool.html", root / "data", root / "tokens.json", 0,
                      clock=lambda: tnow[0])  # fmt: skip
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        logbuf = io.StringIO()
        with contextlib.redirect_stdout(logbuf):
            # seeds + legacy auth model
            seed = {
                "page": "data:image/png;base64,x",
                "words": [{"text": "hi", "box": [0, 0, 9, 9], "cuts": []}],
            }
            for pid in ("p1", "p2", "p3"):
                assert req("POST", f"/cursive/api/seed/{pid}?t=AAA", seed)[0] == 200
            assert req("POST", "/cursive/api/seed/..evil?t=AAA", seed)[0] == 404
            assert req("POST", "/cursive/api/seed/p9?t=WWW", seed)[0] == 401, "worker tokens dead"
            assert req("GET", "/cursive/api/pages")[0] == 401, "no auth -> 401"
            assert req("GET", "/cursive/api/me")[0] == 401

            # users: create via legacy admin token; validation
            new_users = "/cursive/api/admin/users?t=AAA"
            assert req("POST", new_users, {"name": "alice", "password": "pw-alice"})[0] == 200
            r = req("POST", new_users, {"name": "boss", "password": "pw-boss", "role": "admin"})
            assert r[0] == 200 and r[1]["role"] == "admin"
            assert req("POST", new_users, {"name": "Bad Name!", "password": "x"})[0] == 400
            assert req("POST", new_users, {"name": "nopass"})[0] == 400

            # login failure/success, me, cookie attributes
            assert login("alice", "wrong")[0] == 403
            assert login("ghost", "pw")[0] == 403
            code, body, hdrs = req(
                "POST", "/cursive/api/login", {"username": "alice", "password": "pw-alice"}
            )
            assert code == 200 and body == {"ok": True, "name": "alice", "role": "worker"}
            c = hdrs["Set-Cookie"]
            assert "HttpOnly" in c and "SameSite=Lax" in c and "Path=/cursive" in c
            assert "Secure" not in c, "plain http -> no Secure flag"
            sid_a = re.search(rf"{COOKIE}=([0-9a-f]{{64}})", c).group(1)
            code, body, _ = req("GET", "/cursive/api/me", headers=ck(sid_a))
            assert code == 200 and body == {"name": "alice", "role": "worker"}
            code, _, hdrs = req("POST", "/cursive/api/login",
                                {"username": "alice", "password": "pw-alice"},
                                {"X-Forwarded-Proto": "https"})  # fmt: skip
            assert code == 200 and "Secure" in hdrs["Set-Cookie"]

            # worker scoping: nothing assigned -> empty world
            code, body, _ = req("GET", "/cursive/api/pages", headers=ck(sid_a))
            assert code == 200 and body["who"] == "alice" and body["pages"] == []
            assert req("GET", "/cursive/api/next", headers=ck(sid_a))[1]["id"] is None
            assert req("GET", "/cursive/api/page/p1", headers=ck(sid_a))[0] == 403
            save = {
                "words": [{"text": "hi", "box": [0, 0, 9, 9], "cuts": [], "done": True}],
                "completed": True,
                "cursor": {"word": 0},
            }
            assert req("POST", "/cursive/api/page/p1", save, ck(sid_a))[0] == 403
            assert req("GET", "/cursive/api/admin/overview", headers=ck(sid_a))[0] == 403
            assert req("POST", "/cursive/api/admin/users", {"name": "x2", "password": "x"},
                       ck(sid_a))[0] == 403  # fmt: skip
            assert req("GET", "/cursive/api/download/p1", headers=ck(sid_a))[0] == 403

            # assign p1+p2 to alice -> scoped pages/next/page
            assert req("POST", "/cursive/api/admin/assign?t=AAA",
                       {"user": "alice", "page_ids": ["p1", "p2"]})[0] == 200  # fmt: skip
            code, body, _ = req("GET", "/cursive/api/pages", headers=ck(sid_a))
            assert [p["id"] for p in body["pages"]] == ["p1", "p2"] and body["role"] == "worker"
            assert all(p["assigned_to"] == "alice" for p in body["pages"])
            assert req("GET", "/cursive/api/page/p3", headers=ck(sid_a))[0] == 403
            assert req("GET", "/cursive/api/page/p1", headers=ck(sid_a))[1]["words"]
            assert req("GET", "/cursive/api/next", headers=ck(sid_a))[1]["id"] == "p1"

            # save p1 (cookie POST), activity stamped, merged page image
            code, body, _ = req("POST", "/cursive/api/page/p1", save, ck(sid_a))
            assert code == 200 and body["ok"]
            code, doc, _ = req("GET", "/cursive/api/page/p1", headers=ck(sid_a))
            assert doc["page"] == seed["page"], "omitted page must be merged from the stored doc"
            assert doc["savedBy"] == "alice" and doc["completed"]
            act = json.loads((root / "data" / "activity.json").read_text())
            assert act["alice"]["page"] == "p1" and act["alice"]["cursor"] == {"word": 0}
            assert act["alice"]["ts_iso"]
            assert req("GET", "/cursive/api/next", headers=ck(sid_a))[1]["id"] == "p2"

            # origin check on cookie POSTs
            save2 = {"words": [{"text": "hi", "box": [0, 0, 9, 9], "cuts": []}], "cursor": 3}
            evil = dict(ck(sid_a), Origin="https://evil.example")
            assert req("POST", "/cursive/api/page/p2", save2, evil)[0] == 403, "origin mismatch"
            good = dict(ck(sid_a), Origin=f"http://127.0.0.1:{port}")
            assert req("POST", "/cursive/api/page/p2", save2, good)[0] == 200
            assert len(list((root / "data" / "saves" / "p1").glob("*.json"))) == 1, "versioned"

            # admin session: full scope + overview shape/progress
            code, sid_b = login("boss", "pw-boss")
            assert code == 200
            code, body, _ = req("GET", "/cursive/api/pages", headers=ck(sid_b))
            assert [p["id"] for p in body["pages"]] == ["p1", "p2", "p3"]
            assert body["who"] == "boss" and body["role"] == "admin"
            assert req("GET", "/cursive/api/page/p3", headers=ck(sid_b))[0] == 200
            assert req("GET", "/cursive/api/next?after=p1", headers=ck(sid_b))[1]["id"] == "p2"
            code, ov, _ = req("GET", "/cursive/api/admin/overview", headers=ck(sid_b))
            assert code == 200 and {u["name"] for u in ov["users"]} == {"alice", "boss"}
            al = next(u for u in ov["users"] if u["name"] == "alice")
            assert al["role"] == "worker" and al["active"] and al["created"]
            assert al["assigned"] == ["p1", "p2"]
            assert al["progress"] == {"words_done": 1, "words_total": 2, "pages_done": 1,
                                      "pages_total": 2}  # fmt: skip
            assert al["last_active"] and al["current"] == {"page": "p2", "cursor": 3}
            boss = next(u for u in ov["users"] if u["name"] == "boss")
            assert boss["assigned"] == [] and boss["current"] is None
            p2s = next(p for p in ov["pages"] if p["id"] == "p2")
            assert p2s["assigned_to"] == "alice" and p2s["done"] == 0 and not p2s["completed"]

            # HTML routes
            assert b"loginpage" in page("/cursive")
            assert b"tool" in page("/cursive", ck(sid_a))
            assert b"loginpage" in page("/cursive/login", ck(sid_b))
            assert b"adminpage" in page("/cursive/admin", ck(sid_b))
            assert b"loginpage" in page("/cursive/admin", ck(sid_a)), "worker gets login page"

            # assign/unassign/reassign
            assert req("POST", "/cursive/api/admin/assign", {"user": None, "page_ids": ["p2"]},
                       ck(sid_b))[0] == 200  # fmt: skip
            code, body, _ = req("GET", "/cursive/api/pages", headers=ck(sid_a))
            assert [p["id"] for p in body["pages"]] == ["p1"]
            assert req("POST", "/cursive/api/admin/assign?t=AAA",
                       {"user": "boss", "page_ids": ["p2"]})[0] == 200  # fmt: skip
            assert req("GET", "/cursive/api/page/p2", headers=ck(sid_a))[0] == 403, "reassigned"
            assert req("POST", "/cursive/api/admin/assign?t=AAA",
                       {"user": "ghost", "page_ids": ["p2"]})[0] == 400  # fmt: skip

            # user update (password change), then disable kills sessions + login
            assert req("POST", new_users, {"name": "alice", "password": "pw-two"})[0] == 200
            assert login("alice", "pw-alice")[0] == 403, "old password dead"
            code, sid_a2 = login("alice", "pw-two")
            assert code == 200
            assert req("POST", new_users, {"name": "alice", "active": False})[0] == 200
            assert req("GET", "/cursive/api/me", headers=ck(sid_a2))[0] == 401, "session killed"
            assert req("GET", "/cursive/api/me", headers=ck(sid_a))[0] == 401
            assert login("alice", "pw-two")[0] == 403, "disabled user cannot log in"

            # logout expires the cookie and the session
            code, body, hdrs = req("POST", "/cursive/api/logout", headers=ck(sid_b))
            assert code == 200 and body == {"ok": True} and "Max-Age=0" in hdrs["Set-Cookie"]
            assert req("GET", "/cursive/api/me", headers=ck(sid_b))[0] == 401

            # legacy admin token still works everywhere
            code, body, _ = req("GET", "/cursive/api/pages?t=AAA")
            assert code == 200 and body["who"] == "admin" and len(body["pages"]) == 3
            assert req("GET", "/cursive/api/me?t=AAA")[1] == {"name": "admin", "role": "admin"}
            assert req("GET", "/cursive/api/download/p1?t=AAA")[0] == 200
            assert req("GET", "/cursive/api/download/p1")[0] == 401
            assert req("GET", "/cursive/api/admin/overview?t=AAA")[0] == 200
            assert req("POST", "/cursive/api/page/nope?t=AAA", save)[0] == 404

            # login rate limit: 5 failures -> 429 for that ip; injectable clock, no sleeping
            rl = {"Cf-Connecting-Ip": "10.9.9.9"}
            for _ in range(5):
                assert login("boss", "wrong", rl)[0] == 403
            assert login("boss", "pw-boss", rl)[0] == 429, "throttled even with good creds"
            code, sid_b2 = login("boss", "pw-boss")
            assert code == 200, "other client ips unaffected"
            tnow[0] += 601.0
            assert login("boss", "pw-boss", rl)[0] == 200, "window expired"
            req("POST", "/cursive/api/logout", headers=ck(sid_b2))

            # static site + traversal
            assert b"site" in page("/")
            try:
                page("/../etc/passwd")
                raise AssertionError("traversal must not succeed")
            except urllib.error.HTTPError as e:
                assert e.code == 404
            httpd.shutdown()

        # logs must never contain passwords, session ids, or tokens
        logs = logbuf.getvalue()
        for secret in ("pw-alice", "pw-boss", "pw-two", sid_a, sid_b, "t=AAA"):
            assert secret and secret not in logs, "secret leaked into logs"
        assert "cursive_session=***" in _redact(f"Cookie: {COOKIE}={sid_a}")
        assert _redact("GET /cursive/api/pages?t=SECRET HTTP/1.1").count("SECRET") == 0
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=Path, help="static site root")
    ap.add_argument("--tool", type=Path, help="labeller HTML file served at /cursive")
    ap.add_argument("--data", type=Path, help="data dir (pages/current/saves + auth json)")
    ap.add_argument("--tokens", type=Path, help="tokens.json")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--useradd", metavar="NAME", help="create/update a user, then exit")
    ap.add_argument("--password", help="password for --useradd")
    ap.add_argument("--userrole", choices=["admin", "worker"], help="role for --useradd")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.useradd:
        if not (args.data and args.password):
            ap.error("--useradd requires --data and --password")
        u = Auth(args.data).upsert_user(args.useradd, args.password, args.userrole)
        print(f"user {args.useradd} ({u['role']}, active={u['active']}) saved")
        return
    if not all([args.site, args.tool, args.data, args.tokens]):
        ap.error("--site, --tool, --data, --tokens are required (or --selftest / --useradd)")
    httpd = serve(args.site, args.tool, args.data, args.tokens, args.port)
    print(
        f"labelling server on 127.0.0.1:{args.port} (site={args.site}, data={args.data})",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
