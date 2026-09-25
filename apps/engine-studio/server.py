#!/usr/bin/env python3
"""Engine Studio -- a local chat app + control panel for `engine serve` (Qwen3.8-Flash-Next / E9
on this M3 Ultra). Stdlib only, no pip install required.

Manages the `engine serve` subprocess (start/stop, prompt cache, external binding), proxies
/v1/chat/completions to it while forwarding the engine_stats P058 attaches, keeps conversation
history as JSON files, and serves a small single-page chat UI. Run: `python3 server.py`.

Not part of the engine itself (Sources/ and Vendor/): this is an application built ON the engine,
not a change to it.
"""
import base64
import collections
import copy
import hashlib
import http.client
import http.server
import ipaddress
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import socket
import socketserver
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import extract

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WEB = HERE / "web"
DATA = HERE / "data"
CONV_DIR = DATA / "conversations"
CONFIG_PATH = DATA / "config.json"
# P103: the SAME disk prefix cache tools/serve.sh uses (ENGINE_DISK_CACHE=1), so whichever client starts the engine finds it warm.
STATE_CACHE_DIR_DEFAULT = Path("~/.cache/engine-prefix").expanduser()
STATE_CACHE_MAX_BYTES = 8 * 1024**3          # evict oldest rungs past this
ENGINE_BIN_DEFAULT = REPO / ".build" / "release" / "engine"
METALLIB_DEFAULT = REPO / ".build" / "release" / "mlx.metallib"
# P103: the model lives in the repository's weights/e9 (the same default as tools/serve.sh).
MODEL_DIR_DEFAULT = str(REPO / "weights" / "e9")

CONV_DIR.mkdir(parents=True, exist_ok=True)
STATE_CACHE_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "model_dir": MODEL_DIR_DEFAULT,
    # P103: 8099 is the port tools/serve.sh and its agent clients (OMP) use. There is ONE engine on this machine
    # (each wires ~412 GB), so the Studio and OMP are two clients of the same server, never two servers.
    "engine_port": 8099,
    # P078: 2048 cut the answer off before the soft stop's ramp (which starts at 2000 thinking
    # tokens) could ever close the block -- the reasoning was truncated by the CAP instead, which is
    # the failure mode the ramp exists to avoid. The ramp plus its deadline now guarantee the block
    # closes, so the cap can sit above a full reasoning chain instead of inside one.
    "max_tokens": 262144,       # P114: the whole context; the engine (--max-tokens-clamp 1) cuts it to what the prompt leaves
    "reasoning_effort": "xhigh",
    "state_cache_enabled": True,
    "state_cache_step": 512,     # serve.sh's value; 64 wrote a rung every 64 tokens for no measured gain
    "mtp_depth": 3,             # P067: speculative decode with the checkpoint's MTP head (0 = serial)
    # P068: the checkpoint's own generation_config (do_sample, temperature 1.0, top_p 0.95, top_k 20);
    # greedy (temperature 0) is what the app sent before, and greedy thinking loops at xhigh.
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    # P077/P078 SOFT STOP, and it REPLACES the budget as the default. The budget truncates the
    # reasoning mid-derivation; this adds a ramped bias to `</think>` so the model closes at its own
    # next sentence boundary. Measured: the six P068 chains all close and answer, 37% fewer tokens,
    # and the ~50 positions per chain where `</think>` is already a contender are all boundaries.
    # `deadline` is the agentic guarantee: past `full` the ramp climbs to a height no token can beat,
    # so a tool loop cannot stall on thinking, and it still lands on a boundary long before then.
    "think_bias_max": 12,
    "think_bias_start": 2000,
    "think_bias_full": 8000,
    "think_bias_deadline": 14000,
    # P068 thinking guard: a HARD close after this many thinking tokens, or when a 32-token window of
    # the reasoning recurs this many times (0 = off). Off by default now -- it cuts mid-derivation and
    # on this checkpoint that costs correctness (see engineThinkBudget in main.swift).
    "thinking_budget": 0,
    "loop_guard": 3,
    "external": False,          # whether THIS control app (the browser's own server) binds 0.0.0.0
    "studio_port": 7860,
    "api_token": None,          # bearer token gating the public /v1/* passthrough; set on first use
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
            # one-time migration: a config written before P078 carries the HARD budget and no soft
            # stop. Move it to the ramp rather than leaving a truncating default in place.
            if "think_bias_max" not in cfg and int(cfg.get("thinking_budget", 0) or 0) > 0:
                cfg["thinking_budget"] = 0
                if int(cfg.get("max_tokens", 0) or 0) <= 4096:
                    cfg["max_tokens"] = DEFAULT_CONFIG["max_tokens"]
            # P103 migration: a model directory that no longer exists falls back to weights/e9; the engine is shared on 8099
            if not Path(str(cfg.get("model_dir", ""))).expanduser().exists():
                cfg["model_dir"] = MODEL_DIR_DEFAULT
            if cfg.get("engine_port") == 8080:
                cfg["engine_port"] = DEFAULT_CONFIG["engine_port"]
            if int(cfg.get("state_cache_step", 0) or 0) == 64:
                cfg["state_cache_step"] = DEFAULT_CONFIG["state_cache_step"]
            if cfg.get("state_cache_dir") and not Path(cfg["state_cache_dir"]).exists():
                cfg.pop("state_cache_dir")
            return {**DEFAULT_CONFIG, **cfg}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def ensure_api_token(cfg):
    """The /v1/* passthrough (for exposing the raw OpenAI-compatible API through a tunnel) is
    gated on this token -- generated once, persisted, never sent anywhere but shown to the
    operator. `engine serve` itself has no authentication; this is the only guard between an
    externally reachable tunnel and unlimited free inference on this machine."""
    if not cfg.get("api_token"):
        cfg["api_token"] = secrets.token_urlsafe(24)
        save_config(cfg)
    return cfg["api_token"]


def lan_ip():
    """Best-effort LAN-reachable IPv4 (no packets sent -- UDP connect() to a routable address just
    picks the local interface the kernel would use)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _port_answers(port, timeout=2.0):
    """True when an OpenAI-compatible server answers /v1/models on this port."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout)
        conn.request("GET", "/v1/models")
        r = conn.getresponse(); r.read(); conn.close()
        return r.status == 200
    except OSError:
        return False


def _engine_pids():
    """PIDs of every `engine` process on this machine -- the one-process rule is machine-wide."""
    try:
        out = subprocess.run(["pgrep", "-x", "engine"], capture_output=True, text=True).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError):
        return []


class EngineProcess:
    """The Studio's view of THE engine. One instance per Studio process.

    P103: there is exactly one engine on this machine -- each `engine serve` wires ~412 GB -- and OMP already
    runs it via tools/serve.sh on 8099. So the Studio is a CLIENT first: if an engine answers on the
    configured port it ATTACHES to it (state "running", external=True) and never starts a second one. It only
    launches an engine itself when none is alive, and it only ever stops an engine it launched; an engine that
    OMP or serve.sh started is not the Studio's to kill.

    `state` is one of "stopped" / "starting" / "running" / "error". Loading a 195 GB checkpoint
    can take anywhere from seconds (warm page cache) to several minutes (cold disk read), so
    `start()` returns as soon as the subprocess is LAUNCHED -- readiness is polled on a
    background thread, and the browser polls `/api/status` to show progress instead of holding
    one HTTP request open for the whole load.
    """

    def __init__(self):
        self.proc = None
        self.port = None
        self.model_dir = None
        self.state_cache_dir = None
        self.started_at = None          # set once READY, not once launched
        self.state = "stopped"
        self.starting_since = None
        self.error_message = None
        self.lock = threading.Lock()
        self.log_path = DATA / "engine_serve.log"
        self.log_file = None
        self.external = False

    def status(self):
        with self.lock:
            own = self.proc is not None and self.proc.poll() is None
            if self.proc is None and self.state in ("stopped", "error", "running"):
                # not ours: is somebody else's engine serving the port? (OMP's, via serve.sh)
                if _port_answers(CFG["engine_port"]):
                    self.state, self.port, self.external = "running", CFG["engine_port"], True
                    self.model_dir = self.model_dir or CFG.get("model_dir")
                    self.error_message = None
                    if not self.started_at:
                        self.started_at = time.time()
                elif self.external:
                    self.state, self.external, self.started_at = "stopped", False, None
            running = self.state == "running" and (own or self.external)
            if self.state == "running" and not running:
                # the subprocess died without us noticing (crash, killed externally)
                self.state = "error"
                self.error_message = f"engine serve exited unexpectedly (see {self.log_path})"
            out = {
                "running": running, "state": self.state, "port": self.port, "model_dir": self.model_dir,
                "state_cache_dir": self.state_cache_dir,
                "uptime_s": (time.time() - self.started_at) if running and self.started_at else 0,
                "pid": self.proc.pid if own else (_engine_pids() or [None])[0],
                "external": bool(self.external),   # attached to an engine the Studio did not start
                "error": self.error_message,
            }
            if self.state == "starting" and self.starting_since:
                out["starting_elapsed_s"] = time.time() - self.starting_since
            return out

    def start(self, cfg):
        with self.lock:
            if self.state in ("starting", "running"):
                if self.external:
                    raise RuntimeError(f"already attached to the shared engine on port {self.port} (started "
                                       f"outside the Studio, e.g. by tools/serve.sh) -- nothing to start")
                raise RuntimeError("engine serve is already running -- stop it first")
            if _port_answers(cfg["engine_port"]):
                # an engine is already serving this port: attach, do not start a second one
                self.state, self.port, self.external = "running", cfg["engine_port"], True
                self.started_at, self.error_message = time.time(), None
                return
            alive = _engine_pids()
            if alive:
                raise RuntimeError(
                    f"an engine process is already alive (pid {alive[0]}) but not answering on port "
                    f"{cfg['engine_port']} -- one engine at a time (each wires ~412 GB). Point the Studio at "
                    f"its port, or stop it first.")
            engine_bin = os.environ.get("ENGINE_STUDIO_BIN", str(ENGINE_BIN_DEFAULT))
            if not Path(engine_bin).exists():
                raise RuntimeError(f"engine binary not found at {engine_bin} -- build it first "
                                    "(swift build -c release --product engine)")
            model_dir = cfg["model_dir"]
            if not Path(model_dir).expanduser().exists():
                raise RuntimeError(f"model directory not found: {model_dir}")
            args = [engine_bin, "serve", "--model", model_dir, "--port", str(cfg["engine_port"]),
                    "--tokens", "0", "--max-tokens-clamp", "1",   # P114: no request default below the context
                    # P103: what tools/serve.sh passes and the Studio did not -- the hot prefix store (P093:
                    # a turn resumes at the previous turn's state in GPU memory), room for a subagent fan-out,
                    # and the reasoning effort the chat UI already sends per request.
                    "--hot-cache-gb", str(cfg.get("hot_cache_gb", 48)),
                    "--max-concurrent", str(cfg.get("max_concurrent", 8)),
                    "--reasoning-effort", str(cfg.get("reasoning_effort", "xhigh"))]
            sc_dir = None
            if cfg.get("state_cache_enabled"):
                sc_dir = cfg.get("state_cache_dir") or str(STATE_CACHE_DIR_DEFAULT)
                Path(sc_dir).mkdir(parents=True, exist_ok=True)
                args += ["--state-cache", sc_dir, "--state-cache-step", str(cfg.get("state_cache_step", 512))]
            mtp = int(cfg.get("mtp_depth", 0) or 0)
            if mtp > 0:
                args += ["--mtp", str(mtp)]
            tb = int(cfg.get("thinking_budget", 0) or 0)
            lg = int(cfg.get("loop_guard", 0) or 0)
            bmax = float(cfg.get("think_bias_max", 0) or 0)
            if bmax > 0:
                args += ["--think-bias-max", str(bmax),
                         "--think-bias-start", str(int(cfg.get("think_bias_start", 2000) or 2000)),
                         "--think-bias-full", str(int(cfg.get("think_bias_full", 8000) or 8000))]
                dl = int(cfg.get("think_bias_deadline", 0) or 0)
                if dl > 0:
                    args += ["--think-bias-deadline", str(dl)]
            if tb > 0:
                args += ["--think-budget", str(tb)]
            if lg > 0:
                args += ["--loop-guard", str(lg)]
            env = dict(os.environ)
            if mtp > 0:
                env["ENGINE_MTP"] = "1"     # the head is built at load only when this is set
            if Path(METALLIB_DEFAULT).exists():
                env.setdefault("MLXFAST_MLX_METALLIB", str(METALLIB_DEFAULT))
            self.log_file = open(self.log_path, "a")
            self.log_file.write(f"\n=== engine serve start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                                 f"{' '.join(shlex.quote(a) for a in args)} ===\n")
            self.log_file.flush()
            self.proc = subprocess.Popen(args, stdout=self.log_file, stderr=subprocess.STDOUT, env=env)
            self.port = cfg["engine_port"]
            self.model_dir = model_dir
            self.state_cache_dir = sc_dir
            self.state = "starting"
            self.starting_since = time.time()
            self.error_message = None
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _wait_ready(self):
        proc = self.proc
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                with self.lock:
                    self.state = "error"
                    self.error_message = f"engine serve exited during startup (exit {proc.returncode}, see {self.log_path})"
                return
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                conn.request("GET", "/v1/models")
                r = conn.getresponse()
                r.read()
                conn.close()
                if r.status == 200:
                    with self.lock:
                        if self.proc is proc:      # not stopped/restarted meanwhile
                            self.state = "running"
                            self.started_at = time.time()
                    return
            except OSError:
                pass
            time.sleep(1)
        with self.lock:
            self.state = "error"
            self.error_message = "engine serve did not become ready within 900s"

    def stop(self):
        with self.lock:
            if self.proc is None and self.external:
                # OMP's engine (or one started by serve.sh): detaching is all the Studio may do
                raise RuntimeError("this engine was not started by the Studio (it was started outside it, e.g. "
                                   "by tools/serve.sh, and other clients may be using it) -- stop it where it was started")
            proc, self.proc = self.proc, None
            self.state = "stopped"
            self.error_message = None
            self.starting_since = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        if self.log_file:
            self.log_file.close()
            self.log_file = None


ENGINE = EngineProcess()
CFG = load_config()
ensure_api_token(CFG)


class TunnelManager:
    """A Cloudflare 'quick tunnel' (`cloudflared tunnel --url ...`): no account, no config, HTTPS,
    a random *.trycloudflare.com host that dies with the process. Exposes THIS control server
    (port 7860 by default), which is what carries the token-gated /v1/* passthrough and, unless
    the operator has locked that down separately, the app's own /api/* surface too."""

    URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

    def __init__(self):
        self.proc = None
        self.state = "stopped"          # stopped / starting / running / error
        self.url = None
        self.error_message = None
        self.started_at = None
        self.lock = threading.Lock()
        self.log_path = DATA / "tunnel.log"

    def status(self):
        with self.lock:
            running = self.state == "running" and self.proc is not None and self.proc.poll() is None
            if self.state == "running" and not running:
                self.state = "error"
                self.error_message = f"cloudflared exited unexpectedly (see {self.log_path})"
            return {
                "state": self.state, "url": self.url, "error": self.error_message,
                "uptime_s": (time.time() - self.started_at) if running and self.started_at else 0,
                "pid": self.proc.pid if (self.proc is not None and self.proc.poll() is None) else None,
            }

    def start(self, target_port):
        with self.lock:
            if self.state in ("starting", "running"):
                raise RuntimeError("tunnel is already running -- stop it first")
            binpath = shutil.which("cloudflared")
            if not binpath:
                raise RuntimeError("cloudflared is not installed (brew install cloudflared)")
            log_file = open(self.log_path, "w")   # truncate: a stale URL from a previous run must
                                                       # never be mistaken for this one's
            log_file.write(f"=== tunnel start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_file.flush()
            self.proc = subprocess.Popen([binpath, "tunnel", "--url", f"http://127.0.0.1:{target_port}"],
                                          stdout=log_file, stderr=subprocess.STDOUT)
            self.state = "starting"
            self.url = None
            self.error_message = None
            self.started_at = None
        threading.Thread(target=self._wait_url, daemon=True).start()

    def _wait_url(self):
        proc = self.proc
        deadline = time.time() + 60
        seen = 0
        while time.time() < deadline:
            if proc.poll() is not None:
                with self.lock:
                    self.state = "error"
                    self.error_message = f"cloudflared exited during startup (exit {proc.returncode}, see {self.log_path})"
                return
            try:
                text = self.log_path.read_text()
            except OSError:
                text = ""
            m = self.URL_RE.search(text[seen:]) or self.URL_RE.search(text)
            seen = len(text)
            if m:
                with self.lock:
                    if self.proc is proc:
                        self.state = "running"
                        self.url = m.group(0)
                        self.started_at = time.time()
                return
            time.sleep(0.5)
        with self.lock:
            self.state = "error"
            self.error_message = f"no tunnel URL appeared within 60s (see {self.log_path})"

    def stop(self):
        with self.lock:
            proc, self.proc = self.proc, None
            self.state = "stopped"
            self.url = None
            self.error_message = None
            self.started_at = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


TUNNEL = TunnelManager()


class TrafficLog:
    """A rolling in-memory log of every request this server has handled, for the UI's traffic
    panel. Not persisted -- restarting Studio clears it, same as the cumulative stats."""

    def __init__(self, maxlen=300):
        self.entries = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.total = 0

    def record(self, method, path, status, ms, source, via_tunnel):
        with self.lock:
            self.total += 1
            self.entries.appendleft({
                "ts": time.strftime("%H:%M:%S"), "method": method, "path": path, "status": status,
                "ms": round(ms, 1), "source": source, "via_tunnel": via_tunnel,
            })

    def snapshot(self):
        with self.lock:
            return {"total": self.total, "entries": list(self.entries)}


TRAFFIC = TrafficLog()

# ---------------------------------------------------------------------------------- conversations

def conv_path(cid):
    if not re.fullmatch(r"[0-9a-f-]{8,36}", cid):
        raise ValueError("bad conversation id")
    return CONV_DIR / f"{cid}.json"


def list_conversations():
    out = []
    for f in CONV_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append({"id": d["id"], "title": d.get("title") or "New chat", "created": d.get("created"),
                     "updated": d.get("updated"), "message_count": len(d.get("messages", []))})
    out.sort(key=lambda c: c["updated"] or c["created"] or "", reverse=True)
    return out


def load_conversation(cid):
    p = conv_path(cid)
    if not p.exists():
        raise FileNotFoundError(cid)
    return json.loads(p.read_text())


def save_conversation(conv):
    conv_path(conv["id"]).write_text(json.dumps(conv, indent=2))


def new_conversation():
    cid = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conv = {"id": cid, "title": None, "created": now, "updated": now, "messages": []}
    save_conversation(conv)
    return conv


# --------------------------------------------------------------------------------- running stats

class CumulativeStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_total_tokens = 0
        self.decode_tps_sum = 0.0
        self.decode_tps_n = 0
        self.peak_gpu_memory_bytes = 0
        self.started_at = time.time()

    def reset(self):
        with self.lock:
            self.__init__()

    def record(self, usage, engine_stats):
        with self.lock:
            self.requests += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
            self.cache_hit_tokens += engine_stats.get("cache_hit_tokens", 0) or 0
            self.cache_total_tokens += engine_stats.get("cache_total_tokens", 0) or 0
            tps = engine_stats.get("decode_tokens_per_second")
            if tps:
                self.decode_tps_sum += tps
                self.decode_tps_n += 1
            mem = engine_stats.get("peak_gpu_memory_bytes")
            if mem:
                self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes, mem)

    def snapshot(self):
        with self.lock:
            hit_pct = (100.0 * self.cache_hit_tokens / self.cache_total_tokens) if self.cache_total_tokens else 0.0
            avg_tps = (self.decode_tps_sum / self.decode_tps_n) if self.decode_tps_n else 0.0
            return {
                "requests": self.requests, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "total_tokens": self.prompt_tokens + self.completion_tokens,
                "cache_hit_tokens": self.cache_hit_tokens, "cache_total_tokens": self.cache_total_tokens,
                "cache_hit_pct": round(hit_pct, 1), "avg_decode_tokens_per_second": round(avg_tps, 2),
                "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes, "uptime_s": time.time() - self.started_at,
            }


STATS = CumulativeStats()


BASELINE_PATH = DATA / "stats_baseline.json"


def _load_baseline():
    try:
        return json.loads(BASELINE_PATH.read_text())
    except (OSError, ValueError):
        return None


def engine_sessions(raw=None):
    """The engine's sessions view, counted from the last Reset (if any). The Reset is a baseline kept HERE -- the
    engine's own counters, cache and process are never touched. A baseline older than the engine's start is dropped:
    a restarted engine already counts from zero. `raw` (one `_engine_sessions_raw()` read) is left untouched;
    `raw_totals` carries the engine's own totals since it started, whatever the baseline."""
    d = copy.deepcopy(raw) if raw is not None else _engine_sessions_raw()
    if d.get("available"):
        d["raw_totals"] = dict(d.get("totals") or {})
    base = _load_baseline() if d.get("available") else None
    if base and d.get("now", 0) - d.get("uptime_s", 0) > base["at"]:
        BASELINE_PATH.unlink(missing_ok=True)
        base = None
    if base:
        t, b = d.get("totals") or {}, base.get("totals") or {}
        d["totals"] = {k: max(0, (v or 0) - (b.get(k) or 0)) for k, v in t.items()}
        d["recent"] = [r for r in d.get("recent") or [] if (r.get("id") or 0) > base["last_id"]]
        d["since"] = base["at"]
    return d


def reset_engine_sessions():
    """Zero the counters from now: record the engine's current totals and newest request id as the baseline."""
    d = _engine_sessions_raw()
    if not d.get("available"):
        raise RuntimeError(d.get("reason", "engine unavailable"))
    ids = [r.get("id") or 0 for r in (d.get("recent") or []) + (d.get("active") or [])]
    base = {"at": d.get("now", time.time()), "totals": d.get("totals") or {}, "last_id": max(ids, default=0)}
    BASELINE_PATH.write_text(json.dumps(base))
    STATS.reset()
    return base


def _engine_sessions_raw():
    """Every request the engine is serving, whoever sent it (OMP, this app, curl): the engine's own
    `GET /v1/engine/sessions`. Counts and timings only -- the engine never puts prompt text there."""
    st = ENGINE.status()
    if not st["running"]:
        return {"available": False, "reason": "engine is not running"}
    try:
        conn = http.client.HTTPConnection("127.0.0.1", int(st["port"]), timeout=3)
        conn.request("GET", "/v1/engine/sessions")
        r = conn.getresponse()
        body = r.read()
        conn.close()
    except OSError as e:
        return {"available": False, "reason": f"engine did not answer: {e}"}
    if r.status == 404:
        return {"available": False, "reason": "this engine build has no sessions endpoint -- restart it on a current build"}
    if r.status != 200:
        return {"available": False, "reason": f"engine answered HTTP {r.status}"}
    d = json.loads(body)
    d["available"] = True
    return d


# ------------------------------------------------------------------------ engine overview (/engine)
#
# `GET /api/engine/overview`: everything the /engine dashboard shows, in one read -- the sessions view (with the Reset
# baseline, plus the engine's raw payload), the live process (argv, parsed flags, ENGINE_*/MLX* environment, sha256 of
# the binary and its metallib), the checkpoint's own facts, the machine, and a derived limits table. READ-ONLY: it
# never starts, stops or POSTs to the engine. /api/* has no authentication and binds 0.0.0.0 when `external` is on, so
# nothing secret goes in: never the Studio config (it holds api_token), no environment variable but ENGINE_*/MLX* (and
# of those a secret-looking one only as "(set, redacted)" -- ENGINE_API_KEY is the engine's bearer token), no prompt
# text. Every field may be missing on an older engine build; the page shows "n/a" for it, nothing here raises.

# a name is secret-looking when one of its -/_ separated words is one of these (whole words: `--max-tokens-clamp` and
# `--tokens` are limits, `ENGINE_API_KEY` / `--api-key` / `*_TOKEN` are credentials)
_SECRET_WORDS = {"KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL", "CREDENTIALS",
                 "COOKIE", "BEARER"}
_ENV_KEEP = re.compile(r"^(ENGINE_|MLX)")
_ENV_SPLIT = re.compile(r"\s+(?=[A-Za-z_][A-Za-z0-9_]*=)")
_REDACTED = "(set, redacted)"
# defence in depth for the generic raw tree: the sessions payload has no prompt text today; if a later build adds a
# text field under one of these names it is dropped here rather than put on an unauthenticated page
_TEXT_KEYS = {"prompt", "prompt_text", "messages", "content", "text", "reasoning_content", "completion", "system_prompt"}
# `engine serve`'s own defaults for a flag that is absent (Sources/EngineCLI/Serve.swift), so the limits table can say
# what an absent flag means instead of "n/a"
_SERVE_DEFAULTS = {"--tokens": 2048, "--max-concurrent": 8, "--mtp": 0, "--think-bias-max": 0, "--think-bias-start": 2000,
                   "--think-bias-full": 8000, "--think-bias-deadline": 0, "--reasoning-effort": "medium",
                   "--hot-cache-gb": 48, "--hot-keep-rungs": 2, "--batch-min": 4, "--batch-window-ms": 25,
                   "--prefill-chunk": 0, "--state-cache-step": 512, "--max-tokens-clamp": 0, "--think-budget": 0,
                   "--loop-guard": 0, "--host": "127.0.0.1", "--port": 8080, "--preserve-thinking": "template"}

_HASH_CACHE = {}                 # (path, size, mtime_ns) -> sha256 hex
_HASH_LOCK = threading.Lock()
_PROC_CACHE = {}                 # (pid, args) -> facts that cannot change while that process lives
_MODEL_CACHE = {}                # (dir, config mtime, dir mtime) -> checkpoint facts
_SYSTEM_CACHE = {"at": 0.0, "facts": None}


def _secret_name(name):
    return any(w in _SECRET_WORDS for w in re.split(r"[-_.]+", str(name).strip("-").upper()))


def _run(args, timeout=3.0, env=None):
    try:
        return subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout, env=env).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _file_sha256(path):
    """sha256 of a file, cached by (path, size, mtime): the binary is ~50 MB, the metallib ~160 MB, and the page polls
    every second. One hash at a time -- a second poll waits for the first instead of hashing the same file again."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _HASH_LOCK:
        if key not in _HASH_CACHE:
            h = hashlib.sha256()
            try:
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
            except OSError:
                return None
            for stale in [k for k in _HASH_CACHE if k[0] == key[0]]:
                del _HASH_CACHE[stale]      # a rebuilt file: keep only the current digest
            _HASH_CACHE[key] = h.hexdigest()
        return _HASH_CACHE[key]


def _file_facts(path, started_at=None):
    try:
        st = os.stat(path)
    except OSError:
        return {"path": str(path), "exists": False}
    out = {"path": str(path), "exists": True, "bytes": st.st_size, "mtime": st.st_mtime, "sha256": _file_sha256(path)}
    if started_at:
        # the file on disk was rebuilt after this process started: its hash is NOT what the process is running
        out["newer_than_process"] = st.st_mtime > started_at
    return out


def _parse_etime(s):
    """ps etime `[[dd-]hh:]mm:ss` -> seconds."""
    try:
        days, _, rest = s.strip().rpartition("-")
        parts = [int(p) for p in rest.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        return None


def _parse_flags(argv):
    """Every `--flag value` pair of an argv, generically. `--flag=value` is split; a `--flag` followed by another flag
    or by nothing is a switch (true). Words before the first flag are positional (the subcommand)."""
    flags, positional, i = {}, [], 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and len(tok) > 2:
            if "=" in tok:
                k, v = tok.split("=", 1)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                k, v = tok, argv[i + 1]
                i += 1
            else:
                k, v = tok, True
            flags[k] = _REDACTED if _secret_name(k) else v
        elif not flags:
            positional.append(tok)
        i += 1
    return flags, positional


def _parse_env(ps_env_out, args_str):
    """ENGINE_* / MLX* variables from `ps -E` (the argv, then the environment, space-separated). Everything else --
    PATH, HOME, any token a shell exported -- is skipped here and never leaves this function."""
    rest = ps_env_out[len(args_str):] if ps_env_out.startswith(args_str) else ps_env_out
    out = {}
    for part in _ENV_SPLIT.split(rest.strip()):
        name, eq, value = part.partition("=")
        if eq and _ENV_KEEP.match(name):
            out[name] = _REDACTED if _secret_name(name) else value.strip()
    return out


def _proc_facts(pid, now):
    """argv, flags, env, executable and hashes of one engine process. The argv/env/exe of a PID never change, so
    they are cached per (pid, argv); only the binary hashes are re-checked (by mtime) on each read."""
    line = _run(["ps", "-ww", "-o", "etime=", "-o", "args=", "-p", str(pid)]).strip()
    if not line:
        return None
    etime, _, args_str = line.partition(" ")
    args_str = args_str.strip()
    key = (pid, args_str)
    facts = _PROC_CACHE.get(key)
    if facts is None:
        up = _parse_etime(etime)
        argv = args_str.split()     # ps joins argv with spaces; a path with a space in it splits here
        flags, positional = _parse_flags(argv)
        env_out = _run(["ps", "-E", "-ww", "-o", "command=", "-p", str(pid)]).strip()
        env = _parse_env(env_out, args_str) if env_out else {}
        env_note = None
        if not env:
            env_note = ("ps shows no ENGINE_*/MLX* environment for this process (another user's process, or none set)"
                        if env_out else "ps -E returned nothing for this process")
        exe, cwd = None, None
        for ln in _run(["lsof", "-a", "-p", str(pid), "-d", "txt", "-Fn"]).splitlines():
            if ln.startswith("n/"):
                exe = ln[1:]        # the first txt entry is the executable the process was started from
                break
        for ln in _run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"]).splitlines():
            if ln.startswith("n/"):
                cwd = ln[1:]
        exe_source = "lsof (txt)"
        if not exe and argv:
            exe, exe_source = argv[0], "argv[0]"
            if not os.path.isabs(exe) and cwd:
                exe = os.path.normpath(os.path.join(cwd, exe))
        shown_argv = []                 # the argv as shown: a secret-looking flag's value redacted, as in `flags`
        for i, a in enumerate(argv):
            prev = argv[i - 1] if i else ""
            if prev.startswith("--") and "=" not in prev and _secret_name(prev) and not a.startswith("--"):
                a = _REDACTED
            elif a.startswith("--") and "=" in a and _secret_name(a.split("=", 1)[0]):
                a = a.split("=", 1)[0] + "=" + _REDACTED
            shown_argv.append(a)
        facts = {"pid": pid, "argv": shown_argv, "command": " ".join(shown_argv),
                 "subcommand": positional[0] if positional else None, "flags": flags,
                 "started_at": (now - up) if up is not None else None, "exe": exe, "exe_source": exe_source,
                 "cwd": cwd, "env": env, "env_note": env_note}
        if len(_PROC_CACHE) > 16:      # PIDs come and go (bench runs); never let this grow
            _PROC_CACHE.clear()
        _PROC_CACHE[key] = facts
    out = dict(facts)
    out["uptime_s"] = (now - facts["started_at"]) if facts["started_at"] else None
    return out


def _engine_process(now, port):
    """THE engine: of every `engine` process, the one serving `port` (else the first `serve`, else the first)."""
    pids = _engine_pids()
    procs = [p for p in (_proc_facts(pid, now) for pid in pids) if p]
    if not procs:
        return {"running": False, "pids": pids, "note": "no engine process found (pgrep -x engine)"}
    def rank(p):
        f = p["flags"]
        return (p["subcommand"] != "serve", str(f.get("--port", _SERVE_DEFAULTS["--port"])) != str(port))
    procs.sort(key=rank)
    main_p = dict(procs[0])
    main_p["running"] = True
    main_p["pids"] = pids
    main_p["others"] = [{"pid": p["pid"], "command": p["command"]} for p in procs[1:]]
    exe = main_p.get("exe")
    if exe:
        main_p["binary"] = _file_facts(exe, main_p.get("started_at"))
        main_p["metallib"] = _file_facts(os.path.join(os.path.dirname(exe), "mlx.metallib"), main_p.get("started_at"))
        main_p["metallib"]["source"] = "beside the binary"
        env_lib = (main_p.get("env") or {}).get("MLXFAST_MLX_METALLIB")
        if env_lib and os.path.realpath(env_lib) != os.path.realpath(main_p["metallib"]["path"]):
            main_p["metallib_env"] = _file_facts(env_lib, main_p.get("started_at"))
            main_p["metallib_env"]["source"] = "MLXFAST_MLX_METALLIB"
    return main_p


def _model_facts(model_dir, cwd=None):
    """The checkpoint's own facts from its directory: config.json (text_config first), generation_config.json, the
    reasoning levels chat_template.jinja accepts, and the weight size on disk. Cached until config.json changes."""
    p = Path(str(model_dir)).expanduser()
    if not p.is_absolute() and cwd:
        p = Path(cwd) / p
    try:
        p = p.resolve()
        key = (str(p), (p / "config.json").stat().st_mtime_ns, p.stat().st_mtime_ns)
    except OSError:
        return {"dir": str(model_dir), "error": "model directory or its config.json is not readable"}
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    out = {"dir": str(p), "id": p.name}
    try:
        cfg = json.loads((p / "config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
        out["error"] = "config.json unreadable"
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def pick(k):
        return tc.get(k, cfg.get(k))
    out["architectures"] = cfg.get("architectures")
    out["model_type"] = cfg.get("model_type")
    out["text_model_type"] = tc.get("model_type")
    for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
              "num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
              "max_position_embeddings", "vocab_size", "dtype", "torch_dtype"):
        v = pick(k)
        if v is not None:
            out[k] = v
    lt = pick("layer_types")
    if isinstance(lt, list):
        out["layer_types"] = dict(collections.Counter(str(x) for x in lt))
    mtp = tc.get("mtp")
    if isinstance(mtp, dict) and mtp.get("num_hidden_layers") is not None:
        out["mtp_layers"] = mtp.get("num_hidden_layers")
    elif pick("mtp_num_hidden_layers") is not None:
        out["mtp_layers"] = pick("mtp_num_hidden_layers")
    q = cfg.get("quantization") or cfg.get("quantization_config") or tc.get("quantization")
    if isinstance(q, dict):
        overrides = collections.Counter()
        for k, v in q.items():
            if isinstance(v, dict):
                overrides[f"{v.get('bits', '?')}-bit g{v.get('group_size', '?')}"] += 1
            elif v is False:
                overrides["not quantized"] += 1
        out["quantization"] = {"bits": q.get("bits"), "group_size": q.get("group_size"), "mode": q.get("mode"),
                               "per_module_overrides": dict(overrides)}
    if isinstance(cfg.get("engine_recipe"), str):
        out["engine_recipe"] = cfg["engine_recipe"][:400]
    try:
        gen = json.loads((p / "generation_config.json").read_text())
        out["generation"] = {k: gen[k] for k in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")
                             if k in gen}
    except (OSError, ValueError):
        out["generation"] = None
    try:
        tpl = (p / "chat_template.jinja").read_text(errors="replace")
        m = re.search(r"reasoning_effort\s+not\s+in\s*\(([^)]*)\)", tpl)
        out["reasoning_levels"] = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)) if m else None
        m = re.search(r"reasoning_effort\s*\|\s*default\(\s*['\"]([^'\"]+)['\"]", tpl)
        out["template_default_effort"] = m.group(1) if m else None
    except OSError:
        out["reasoning_levels"] = None
    weights, total, files = 0, 0, 0
    try:
        for e in os.scandir(p):
            try:
                size = e.stat().st_size if e.is_file() else 0
            except OSError:
                continue
            total += size
            if e.name.endswith(".safetensors"):
                weights += size
                files += 1
    except OSError:
        pass
    out["weights_bytes"], out["weight_files"], out["dir_bytes"] = weights, files, total
    _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = out
    return out


def _system_facts(now):
    """The machine: RAM, the GPU wired-memory sysctls, macOS, uptime. sysctl is re-read every 10 s at most."""
    if _SYSTEM_CACHE["facts"] is None or now - _SYSTEM_CACHE["at"] > 10:
        names = ["hw.memsize", "iogpu.wired_limit_mb", "iogpu.disable_wired_collector", "kern.boottime",
                 "kern.osversion", "hw.model", "machdep.cpu.brand_string", "hw.ncpu"]
        raw = {}
        for ln in _run(["sysctl"] + names).splitlines():   # a missing name is skipped (stderr), the rest still print
            k, sep, v = ln.partition(": ")
            if sep:
                raw[k.strip()] = v.strip()

        def as_int(k):
            try:
                return int(raw[k])
            except (KeyError, ValueError):
                return None
        m = re.search(r"sec\s*=\s*(\d+)", raw.get("kern.boottime", ""))
        _SYSTEM_CACHE["facts"] = {
            "ram_bytes": as_int("hw.memsize"),
            "iogpu_wired_limit_mb": as_int("iogpu.wired_limit_mb"),
            "iogpu_disable_wired_collector": as_int("iogpu.disable_wired_collector"),
            "macos": platform.mac_ver()[0] or None, "os_build": raw.get("kern.osversion"),
            "hw_model": raw.get("hw.model"), "cpu": raw.get("machdep.cpu.brand_string"), "cpus": as_int("hw.ncpu"),
            "boot_time": int(m.group(1)) if m else None,
        }
        _SYSTEM_CACHE["at"] = now
    out = dict(_SYSTEM_CACHE["facts"])
    out["uptime_s"] = (now - out["boot_time"]) if out.get("boot_time") else None
    try:
        out["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        out["load_avg"] = None
    return out


def _scrub_text(obj):
    if isinstance(obj, dict):
        return {k: _scrub_text(v) for k, v in obj.items()
                if not (k in _TEXT_KEYS and isinstance(v, (str, list)))}
    if isinstance(obj, list):
        return [_scrub_text(v) for v in obj]
    return obj


def _gb(b):
    try:
        return f"{float(b) / 1e9:.1f} GB"
    except (TypeError, ValueError):
        return None


def _int_or_none(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _limits(raw, proc, model, system):
    """A DERIVED, human-readable table of what the running engine allows, each row {name, value, source, note}. Built
    from the runtime block the engine reports first, then its flags, then `engine serve`'s documented defaults."""
    rows = []
    rt = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    hot = raw.get("hot_cache") if isinstance(raw.get("hot_cache"), dict) else {}
    ram = raw.get("ram_budget") if isinstance(raw.get("ram_budget"), dict) else {}
    mem = raw.get("memory") if isinstance(raw.get("memory"), dict) else {}
    flags = proc.get("flags") or {}
    env = proc.get("env") or {}
    have_flags = bool(proc.get("running"))

    def row(name, value, source, note=""):
        rows.append({"name": name, "value": "n/a" if value in (None, "") else value,
                     "source": source or "n/a", "note": note or ""})

    def flag(name):
        """(value, source-text) of a serve flag: the process's own, else the documented default."""
        if name in flags:
            v = flags[name]
            return v, f"{name} {v}" if v is not True else name
        d = _SERVE_DEFAULTS.get(name)
        if not have_flags:
            return d, f"{name}: engine process not found, serve default {d}"
        return d, f"{name} absent (serve default {d})"

    def fmt_tok(n):
        return f"{n:,} tokens" if isinstance(n, int) else None

    # context
    native = _int_or_none(model.get("max_position_embeddings"))
    mc = _int_or_none(rt.get("max_context"))
    if mc:
        note = f"checkpoint max_position_embeddings {native:,}" if native else ""
        if "--max-context" in flags:
            note += ("; " if note else "") + f"capped by --max-context {flags['--max-context']}"
        row("Context window", fmt_tok(mc), "runtime.max_context", note)
    elif native:
        mc = native
        row("Context window", fmt_tok(native), "config.json max_position_embeddings",
            "this engine build reports no runtime.max_context")
    else:
        row("Context window", None, "runtime.max_context", "")

    mtp_v, mtp_src = flag("--mtp")
    mtp = _int_or_none(mtp_v)
    if mtp is None:
        mtp = _int_or_none(raw.get("mtp_depth")) or 0
    if "--batch-mtp" in flags:
        bmtp, bmtp_src = _int_or_none(flags["--batch-mtp"]) or 0, f"--batch-mtp {flags['--batch-mtp']}"
    else:
        bmtp, bmtp_src = mtp, "--batch-mtp absent (= --mtp)"
    lookahead = max(mtp, bmtp) + 1
    row("Output room", f"prompt + max_tokens ≤ {mc - lookahead:,}" if mc else None,
        f"max_context − lookahead {lookahead} (max(--mtp {mtp}, --batch-mtp {bmtp}) + 1)",
        "admission rule: prompt + max_tokens + lookahead must fit max_context; the longest reply to a P-token "
        f"prompt is {mc - lookahead:,} − P tokens" if mc else "")

    tok_v, tok_src = flag("--tokens")
    tok = _int_or_none(tok_v)
    clamp_v, clamp_src = flag("--max-tokens-clamp")
    clamp = (_int_or_none(clamp_v) or 0) != 0
    decisions = raw.get("max_tokens_decisions")
    if tok == 0:
        row("Default max_tokens", "rest of the context", tok_src,
            "a request that names no max_tokens may run to max_context − prompt − lookahead")
    else:
        row("Default max_tokens", fmt_tok(tok), tok_src,
            "used when the client sends no max_tokens; " +
            ("cut to the room left when prompt + default would not fit" if clamp
             else "if prompt + default + lookahead exceeds max_context the request is refused (HTTP 400)"))
    note = ""
    if isinstance(decisions, dict):
        note = "decisions so far: " + (", ".join(f"{k} {v}" for k, v in sorted(decisions.items())) or "none")
    elif clamp:
        note = ("this engine build publishes no max_tokens_decisions -- it may predate P114, and then an oversized "
                "max_tokens is refused (HTTP 400) whatever the flag says")
    else:
        note = "prompt + max_tokens + lookahead > max_context is answered HTTP 400, with the numbers in the message"
    row("Oversized max_tokens", "clamped to the room left" if clamp else "refused with HTTP 400", clamp_src, note)

    eff_v, eff_src = flag("--reasoning-effort")
    levels = model.get("reasoning_levels")
    eff_note = f"template default {model['template_default_effort']}" if model.get("template_default_effort") else ""
    edec = raw.get("reasoning_effort_decisions")
    if isinstance(edec, dict):
        eff_note += ("; " if eff_note else "") + "asked->used so far: " + \
            (", ".join(f"{k} {v}" for k, v in sorted(edec.items())) or "none")
    row("Default reasoning effort", eff_v, eff_src + " (server default when a request names none)", eff_note)
    row("Reasoning levels", ", ".join(levels) if levels else None, "chat_template.jinja",
        ("P114 builds map high/max/maximum -> xhigh and minimal/none/off -> low; anything else gets the default"
         if isinstance(edec, dict) else "a level outside these falls back to the server default"))

    bmax_v, _ = flag("--think-bias-max")
    bmax = _float_or_none(bmax_v) or 0.0
    start = _int_or_none(flag("--think-bias-start")[0]) or 0
    full = _int_or_none(flag("--think-bias-full")[0]) or 0
    dl = _int_or_none(flag("--think-bias-deadline")[0]) or 0
    src = f"--think-bias-max {bmax:g} / -start {start} / -full {full} / -deadline {dl}"
    if bmax <= 0 or full <= start:
        row("Thinking soft stop", "off", src, "no ramp on </think>: thinking can run until max_tokens")
    else:
        val = f"bias 0 → {bmax:g} over {start:,}–{full:,} thinking tokens" + (f", +40 more by {dl:,}" if dl > full else "")
        note = (f"thinking is guaranteed to close by {dl:,} thinking tokens (it usually closes at its own sentence "
                "boundary long before)" if dl > full else "no deadline: thinking can in principle run to max_tokens")
        row("Thinking soft stop", val, src, note + "; a request may override with think_bias_max / think_bias_deadline")
    tb = _int_or_none(flag("--think-budget")[0]) or 0
    lg = _int_or_none(flag("--loop-guard")[0]) or 0
    row("Hard thinking guard", "off" if not (tb or lg) else
        ", ".join(x for x in (f"budget {tb:,} tokens" if tb else "", f"loop guard {lg} repeats" if lg else "") if x),
        f"--think-budget {tb} / --loop-guard {lg}",
        "a hard close that injects the wrap-up phrase; a request may set thinking_budget / loop_guard")

    mcr = _int_or_none(raw.get("max_concurrent"))
    mcr_src = "sessions.max_concurrent"
    if mcr is None:
        mcr_v, mcr_src = flag("--max-concurrent")
        mcr = _int_or_none(mcr_v)
    row("Max concurrent requests", f"{mcr} requests" if mcr else None, mcr_src,
        "over it the engine answers HTTP 503 + Retry-After; each request holds its own KV")

    bmin = _int_or_none(rt.get("batch_min"))
    bmin_src = "runtime.batch_min"
    if bmin is None:
        v, bmin_src = flag("--batch-min")
        bmin = _int_or_none(v)
    if "--batch-max-rows" in flags:
        bmr, bmr_src = _int_or_none(flags["--batch-max-rows"]), f"--batch-max-rows {flags['--batch-max-rows']}"
    else:
        bmr, bmr_src = max(bmin or 0, 8), "--batch-max-rows absent (serve default max(batch-min, 8))"
    win_v, win_src = flag("--batch-window-ms")
    row("Batch max rows", f"{bmr} rows" if bmr is not None else None, bmr_src,
        "the widest batched decode step")
    row("Batch start threshold", "off" if bmin == 0 else (f"{bmin} rows, window {win_v} ms" if bmin else None),
        f"{bmin_src}; {win_src}",
        "decode steps are batched once this many requests are eligible; a leader waits the window for peers")
    row("MTP depth", f"K={mtp} solo, K={bmtp} batched" if mtp or bmtp else "off (serial decode)",
        f"{mtp_src}; {bmtp_src}",
        (f"batched MTP policy: {rt['batch_mtp_policy']}" if rt.get("batch_mtp_policy") else "") +
        ("; ENGINE_MTP is not 1, so the head was not built at load" if mtp and env and env.get("ENGINE_MTP") != "1" else ""))

    ceiling = hot.get("ceiling_bytes")
    csrc = "hot_cache.ceiling_bytes"
    if ceiling is None and rt.get("hot_cache_gb") is not None:
        ceiling, csrc = (_float_or_none(rt.get("hot_cache_gb")) or 0) * 1e9, "runtime.hot_cache_gb"
    if ceiling is None:
        v, csrc = flag("--hot-cache-gb")
        ceiling = (_float_or_none(v) or 0) * 1e9
    charged = hot.get("charged_bytes")
    hnote = []
    if charged is not None and ceiling:
        hnote.append(f"now {_gb(charged)} charged ({100 * charged / ceiling:.0f}%)")
    if hot.get("entries") is not None:
        hnote.append(f"{hot['entries']} entries, {hot.get('rungs', 'n/a')} rungs")
    hnote.append(f"keeps {flag('--hot-keep-rungs')[0]} decode rungs per sequence")
    if ram.get("shared") or rt.get("shared_ram_budget"):
        hnote.append("shares one RAM budget with the KV reservations (an idle ceiling)")
    row("Hot prefix cache ceiling", "off" if ceiling == 0 else _gb(ceiling), csrc, "; ".join(hnote))

    kvb = rt.get("kv_budget_bytes")
    ksrc = "runtime.kv_budget_bytes"
    if kvb is None:
        kvb, ksrc = ram.get("capacity_bytes"), "ram_budget.capacity_bytes"
    knote = []
    if ram.get("live_limit_bytes") is not None:
        knote.append(f"live limit {_gb(ram['live_limit_bytes'])}")
    if ram.get("active_reservations") is not None:
        knote.append(f"active reservations {ram['active_reservations']} ({_gb(ram.get('active_reservation_bytes', 0))})")
    if "--kv-budget-gb" in flags:
        knote.append(f"--kv-budget-gb {flags['--kv-budget-gb']}")
    row("RAM KV budget", _gb(kvb), ksrc, "; ".join(knote))

    bpt = _float_or_none(rt.get("kv_bytes_per_token"))
    row("KV bytes per token", f"{bpt:,.0f} B ({bpt * 1000 / 1e6:.1f} MB per 1k tokens)" if bpt else None,
        "runtime.kv_bytes_per_token", "full-attention layers (+ the MTP layer) K/V plus the indexer")
    fc = _float_or_none(rt.get("full_context_history_capacity_bytes_per_row_derived"))
    fixed = _float_or_none(rt.get("fixed_bytes_per_sequence"))
    if fc:
        per = fc + (fixed or 0)
        val = _gb(fc) + (f" history + {_gb(fixed)} fixed" if fixed else " history")
        note = (f"derived: the KV budget holds at most {int(float(kvb) // per)} full-context sessions at once"
                if _float_or_none(kvb) else "")
        row("Bytes per full-context session", val,
            "runtime.full_context_history_capacity_bytes_per_row_derived" + (" + fixed_bytes_per_sequence" if fixed else ""),
            note)
    else:
        row("Bytes per full-context session", _gb(bpt * mc) if bpt and mc else None,
            "derived: kv_bytes_per_token × max_context", "this engine build reports no per-row capacity")

    pc = _int_or_none(rt.get("prefill_chunk"))
    pc_src = "runtime.prefill_chunk"
    if pc is None:
        v, pc_src = flag("--prefill-chunk")
        pc = _int_or_none(v)
    if pc == 0:
        alone, shared = rt.get("prefill_chunk_alone"), rt.get("prefill_chunk_shared")
        val = f"adaptive: {alone if alone is not None else 'n/a'} alone / {shared if shared is not None else 'n/a'} shared"
    else:
        val = fmt_tok(pc)
    pn = []
    if "batch_prefill" in rt:
        pn.append(f"batched prefill {'on' if rt['batch_prefill'] else 'off'}")
    for k, label in (("prefill_batch_max", "max rows"), ("prefill_batch_token_budget", "token budget"),
                     ("prefill_batch_decode_token_budget", "decode-token budget"),
                     ("prefill_batch_chunk_widths", "widths"), ("prefill_budget_cut", "budget cut")):
        if k in rt:
            v = rt[k]
            pn.append(f"{label} {('on' if v else 'off') if isinstance(v, bool) else v}")
    row("Prefill chunk policy", val, pc_src + ", runtime.prefill_*", "; ".join(pn))

    sc = flags.get("--state-cache")
    step_v, step_src = flag("--state-cache-step")
    row("Disk prefix cache", sc if sc else "off", "--state-cache" if sc else "--state-cache absent",
        f"rung step {step_v} tokens ({step_src})" +
        (f"; cap --state-cache-max-gb {flags['--state-cache-max-gb']}" if "--state-cache-max-gb" in flags else "") +
        (f"; ENGINE_DISK_CACHE={env['ENGINE_DISK_CACHE']}" if "ENGINE_DISK_CACHE" in env else ""))

    res = rt.get("model_resident_bytes") or rt.get("model_loaded_bytes")
    row("Model in memory", _gb(res), "runtime.model_resident_bytes",
        f"of {_gb(system.get('ram_bytes'))} RAM; weights on disk {_gb(model.get('weights_bytes'))}"
        if system.get("ram_bytes") else "")
    cl = env.get("ENGINE_CACHE_LIMIT_GB")
    row("MLX buffer cache limit", f"{cl} GB" if cl else ("32 GB" if env else None),
        "ENGINE_CACHE_LIMIT_GB" if cl else ("ENGINE_CACHE_LIMIT_GB unset (default 32)" if env else "environment not visible"),
        f"freed buffers MLX keeps for reuse; now {_gb(mem['cache_bytes'])}" if mem.get("cache_bytes") is not None else "")
    wc = rt.get("wired_configuration") if isinstance(rt.get("wired_configuration"), dict) else {}
    row("Wired memory", (f"{wc.get('state', 'n/a')} (requested {wc.get('requested_gb', 'n/a')} GB)" if wc else None),
        "runtime.wired_configuration",
        f"iogpu.wired_limit_mb {system.get('iogpu_wired_limit_mb', 'n/a')}, "
        f"iogpu.disable_wired_collector {system.get('iogpu_disable_wired_collector', 'n/a')}"
        + (f", ENGINE_WIRED_LIMIT_GB={env['ENGINE_WIRED_LIMIT_GB']}" if "ENGINE_WIRED_LIMIT_GB" in env else ""))
    if rt.get("template_cache_mb") is not None:
        row("Template cache", f"{rt['template_cache_mb']} MB", "runtime.template_cache_mb",
            f"{rt.get('template_cache_entries', 'n/a')} entries, "
            f"{(_float_or_none(rt.get('template_cache_bytes')) or 0) / 1e6:.1f} MB used, "
            f"hits {rt.get('template_cache_hits', 'n/a')} / misses {rt.get('template_cache_misses', 'n/a')}")
    host_v, host_src = flag("--host")
    port_v, port_src = flag("--port")
    host_v = rt.get("host") or host_v
    row("Listen address", f"{host_v}:{port_v}", f"{host_src}; {port_src}",
        "loopback only" if str(host_v).startswith("127.") else "reachable from the network (requires ENGINE_API_KEY)")
    if rt.get("scheduler"):
        row("Scheduler", f"{rt['scheduler']} (decode burst {rt.get('scheduler_decode_burst', 'n/a')})",
            "runtime.scheduler", "ENGINE_SCHEDULER=fifo selects the old FIFO")
    gen = model.get("generation") or {}
    if gen:
        row("Checkpoint sampling", ", ".join(f"{k} {v}" for k, v in gen.items()), "generation_config.json",
            "the checkpoint's recommendation; each client sends its own per request")
    return rows


def engine_overview():
    now = time.time()
    raw = _engine_sessions_raw()
    sessions = engine_sessions(raw)
    raw = _scrub_text(raw)
    sessions = _scrub_text(sessions)
    port = ENGINE.port or CFG.get("engine_port")     # _engine_sessions_raw() just refreshed ENGINE's status
    proc = _engine_process(now, port)
    env = proc.pop("env", None) if proc.get("running") else None
    env_note = proc.pop("env_note", None) if proc.get("running") else "no engine process"
    model_dir = (proc.get("flags") or {}).get("--model") or CFG.get("model_dir")
    model = _model_facts(model_dir, proc.get("cwd"))
    system = _system_facts(now)
    proc_view = dict(proc, env=env or {})     # _limits reads the env; the page gets it once, under "env"
    if not proc.get("running") and not raw.get("available"):
        # serve's defaults for an engine that does not exist would read as this machine's limits: say so instead
        limits = [{"name": "Engine", "value": "not running", "source": "pgrep -x engine; GET /v1/engine/sessions",
                   "note": "the limits are derived from the live engine (runtime block + its flags); none is running"}]
    else:
        try:
            limits = _limits(raw if raw.get("available") else {}, proc_view, model, system)
        except Exception as e:      # a malformed field from some engine build must not take the page down
            limits = [{"name": "limits", "value": "n/a", "source": "studio", "note": f"could not derive: {e!r}"}]
    return {
        "at": now, "available": bool(raw.get("available")), "reason": raw.get("reason"),
        "engine_port": port, "attached": bool(ENGINE.external),
        "sessions": sessions, "raw": raw, "process": proc,
        "env": {"available": bool(env), "vars": env or {}, "note": env_note,
                "filter": "only ENGINE_* and MLX* keys; secret-looking names redacted"},
        "model": model, "system": system, "limits": limits,
    }


# -------------------------------------------------------------------------- attachments / URLs

MAX_FETCH_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _is_public_host(hostname):
    """SSRF guard: refuse anything that resolves to a private, loopback, link-local or reserved
    address. This server is reachable through a tunnel (Cloudflare quick tunnel, see the README),
    so a URL-fetch feature is a URL-fetch-shaped SSRF surface unless it refuses to touch this
    machine's own network."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for family, _, _, _, sockaddr in infos:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def fetch_url_bytes(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError("only http:// and https:// URLs are supported")
    if not parsed.hostname or not _is_public_host(parsed.hostname):
        raise RuntimeError("refusing to fetch a private, loopback or link-local address")
    req = urllib.request.Request(url, headers={"User-Agent": "EngineStudio/0.1 (+local research tool)"})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not fetch this URL: {e}")
    with resp:
        final_host = urllib.parse.urlparse(resp.geturl()).hostname
        if not final_host or not _is_public_host(final_host):
            raise RuntimeError("refused: redirected to a private/loopback address")
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read(MAX_FETCH_BYTES + 1)
        if len(data) > MAX_FETCH_BYTES:
            raise RuntimeError(f"page is larger than {MAX_FETCH_BYTES // 1024 // 1024} MB, refusing to fetch")
        return content_type, data


def evict_state_cache_if_over_budget(dir_path):
    try:
        d = Path(dir_path)
        files = sorted(d.glob("st_*.safetensors"), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while total > STATE_CACHE_MAX_BYTES and files:
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink(missing_ok=True)
    except OSError:
        pass


# ------------------------------------------------------------------------------------- HTTP layer

MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".json": "application/json"}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "EngineStudio/0.1"

    def log_message(self, fmt, *args):
        pass  # keep stdout quiet; engine's own log goes to engine_serve.log

    def send_response(self, code, message=None):
        self._status_code = code   # captured for the traffic log regardless of which method sent it
        super().send_response(code, message)

    def _traffic_source(self):
        # cloudflared adds this on every request it proxies; without it every request (tunnel or
        # truly local) looks identical -- 127.0.0.1 -- since cloudflared itself connects locally.
        cf_ip = self.headers.get("Cf-Connecting-Ip")
        return (cf_ip, True) if cf_ip else (self.client_address[0], False)

    def _dispatch(self, method, fn):
        self._status_code = None
        t0 = time.time()
        try:
            if self.headers.get("Cf-Connecting-Ip") and not self.path.startswith("/v1/"):
                # Through the tunnel only the token-gated /v1/* passthrough is served. The UI and /api/* carry
                # no authentication -- /api/status returns the config, token included, and /api/server/stop
                # stops the engine -- so they stay local.
                return self._error(403, "only /v1/* (with the bearer token) is served through the tunnel")
            fn()
        finally:
            source, via_tunnel = self._traffic_source()
            # the sessions panel and /engine poll these every second; logging them would bury real traffic
            polled = self.path in ("/api/engine/sessions", "/api/engine/overview") and (self._status_code or 0) == 200
            if not polled:
                TRAFFIC.record(method, self.path, self._status_code or 0, (time.time() - t0) * 1000, source, via_tunnel)

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _serve_static(self, path):
        if path == "/":
            path = "/index.html"
        f = (WEB / path.lstrip("/")).resolve()
        if WEB not in f.parents and f != WEB:
            return self._error(404, "not found")
        if not f.exists() or not f.is_file():
            return self._error(404, "not found")
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(f.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- GET

    def do_GET(self):
        self._dispatch("GET", self._do_GET)

    def _do_GET(self):
        if self.path.startswith("/v1/"):
            return self._proxy_v1("GET")
        if self.path == "/api/status":
            s = ENGINE.status()
            s["config"] = CFG
            s["tunnel"] = TUNNEL.status()
            if s["running"]:
                s["api_url_local"] = f"http://127.0.0.1:{s['port']}/v1"
                s["api_url_lan"] = f"http://{lan_ip()}:{s['port']}/v1"
            return self._json(s)
        if self.path == "/api/stats":
            return self._json(STATS.snapshot())
        if self.path == "/api/traffic":
            return self._json(TRAFFIC.snapshot())
        if self.path == "/api/engine/sessions":
            return self._json(engine_sessions())
        if self.path == "/api/engine/overview":
            return self._json(engine_overview())
        if self.path.split("?", 1)[0] in ("/engine", "/engine/"):
            return self._serve_static("/engine.html")      # the full engine dashboard (web/engine.html)
        if self.path == "/api/conversations":
            return self._json(list_conversations())
        m = re.fullmatch(r"/api/conversations/([0-9a-f-]{8,36})", self.path)
        if m:
            try:
                return self._json(load_conversation(m.group(1)))
            except FileNotFoundError:
                return self._error(404, "no such conversation")
        return self._serve_static(self.path)

    # ---- POST / DELETE

    def do_POST(self):
        self._dispatch("POST", self._do_POST)

    def _do_POST(self):
        if self.path.startswith("/v1/"):
            return self._proxy_v1("POST")
        if self.path == "/api/server/start":
            body = self._read_json()
            cfg = dict(CFG)
            cfg.update({k: v for k, v in body.items() if k in cfg})
            try:
                ENGINE.start(cfg)
            except RuntimeError as e:
                return self._error(400, str(e))
            CFG.update(cfg)
            save_config(CFG)
            return self._json(ENGINE.status())
        if self.path == "/api/server/stop":
            try:
                ENGINE.stop()
            except RuntimeError as e:      # P103: an engine the Studio did not start is not the Studio's to stop
                return self._error(409, str(e))
            return self._json(ENGINE.status())
        if self.path == "/api/stats/reset":
            try:
                return self._json(reset_engine_sessions())
            except RuntimeError as e:
                return self._error(503, str(e))
        if self.path == "/api/tunnel/start":
            # With the UI on the LAN (external: true) /api/* has no authentication, so any device on the network could
            # otherwise put this machine's engine on the internet. Only a browser on this Mac may open the tunnel.
            if self.client_address[0] not in ("127.0.0.1", "::1"):
                return self._error(403, "the tunnel can only be started from this machine (http://127.0.0.1)")
            try:
                TUNNEL.start(CFG["studio_port"])
            except RuntimeError as e:
                return self._error(400, str(e))
            return self._json(TUNNEL.status())
        if self.path == "/api/tunnel/stop":
            TUNNEL.stop()
            return self._json(TUNNEL.status())
        if self.path == "/api/conversations":
            return self._json(new_conversation())
        if self.path == "/api/chat":
            return self._handle_chat()
        if self.path == "/api/attachments/upload":
            return self._handle_upload()
        if self.path == "/api/fetch_url":
            return self._handle_fetch_url()
        return self._error(404, "not found")

    def do_DELETE(self):
        self._dispatch("DELETE", self._do_DELETE)

    def _do_DELETE(self):
        m = re.fullmatch(r"/api/conversations/([0-9a-f-]{8,36})", self.path)
        if m:
            p = conv_path(m.group(1))
            p.unlink(missing_ok=True)
            return self._json({"ok": True})
        return self._error(404, "not found")

    # ---- /v1/*: an AUTHENTICATED raw passthrough to engine serve's own OpenAI-compatible API,
    # for exposing it (e.g. through a tunnel) to a client outside this machine. `/api/chat` above
    # is for THIS app's own UI and keeps history; this is for someone else's client code.

    def _proxy_v1(self, method):
        token = CFG.get("api_token")
        auth = self.headers.get("Authorization", "")
        if not token or auth != f"Bearer {token}":
            return self._error(401, "missing or invalid bearer token")
        st = ENGINE.status()
        if not st["running"]:
            return self._error(503, "engine serve is not running")
        port = st["port"]
        if method == "GET" and self.path == "/v1/models":
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("GET", "/v1/models")
            r = conn.getresponse()
            body = r.read()
            conn.close()
            self.send_response(r.status)
            self.send_header("Content-Type", r.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if method == "POST" and self.path == "/v1/chat/completions":
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n)
            try:
                req_obj = json.loads(data or b"{}")
            except json.JSONDecodeError:
                return self._error(400, "invalid JSON body")
            stream = bool(req_obj.get("stream"))
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
            ua = (self.headers.get("User-Agent") or "client")[:40]
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json", "X-Engine-Client": f"studio /v1 proxy ({ua})"})
            r = conn.getresponse()
            if stream:
                self.send_response(r.status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = r.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(("%x\r\n" % len(chunk)).encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            else:
                body = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            conn.close()
            return
        return self._error(404, "unknown endpoint (only GET /v1/models and POST /v1/chat/completions are proxied)")

    # ---- attachments: text extracted here, never touches the model directly

    def _handle_upload(self):
        body = self._read_json()
        filename = body.get("filename", "")
        data_b64 = body.get("data_base64", "")
        if not filename or not data_b64:
            return self._error(400, "filename and data_base64 are required")
        try:
            data = base64.b64decode(data_b64, validate=False)
        except Exception:
            return self._error(400, "data_base64 is not valid base64")
        if len(data) > MAX_UPLOAD_BYTES:
            return self._error(413, f"file is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
        try:
            text, truncated = extract.extract_file(filename, data)
        except RuntimeError as e:
            return self._error(422, str(e))
        return self._json({"filename": filename, "text": text, "truncated": truncated, "chars": len(text)})

    def _handle_fetch_url(self):
        body = self._read_json()
        url = (body.get("url") or "").strip()
        if not url:
            return self._error(400, "url is required")
        if "://" not in url:
            url = "https://" + url
        try:
            content_type, data = fetch_url_bytes(url)
            title, text, truncated = extract.extract_url(url, lambda u: (content_type, data))
        except RuntimeError as e:
            return self._error(422, str(e))
        return self._json({"url": url, "title": title, "text": text, "truncated": truncated, "chars": len(text)})

    # ---- chat: proxy to engine serve, stream SSE through, append to history

    def _handle_chat(self):
        body = self._read_json()
        cid = body.get("conversation_id")
        user_text = body.get("message", "")
        if not cid or not user_text:
            return self._error(400, "conversation_id and message are required")
        try:
            conv = load_conversation(cid)
        except FileNotFoundError:
            return self._error(404, "no such conversation")
        if not ENGINE.status()["running"]:
            return self._error(409, "engine serve is not running -- start it first")

        conv["messages"].append({"role": "user", "content": user_text})
        upstream_messages = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]
        payload = {
            "model": "engine", "messages": upstream_messages, "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": body.get("max_tokens", CFG["max_tokens"]),
            "temperature": body.get("temperature", CFG.get("temperature", 1.0)),
            "top_p": body.get("top_p", CFG.get("top_p", 0.95)),
            "top_k": body.get("top_k", CFG.get("top_k", 20)),
            "reasoning_effort": body.get("reasoning_effort", CFG["reasoning_effort"]),
        }
        data = json.dumps(payload).encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send_chunk(obj):
            frame = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(("%x\r\n" % len(frame)).encode() + frame + b"\r\n")

        reasoning, content, engine_stats, usage = "", "", {}, {}
        try:
            conn = http.client.HTTPConnection("127.0.0.1", ENGINE.status()["port"], timeout=600)
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json", "X-Engine-Client": "engine-studio"})
            resp = conn.getresponse()
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    line, buf = buf.split(b"\n\n", 1)
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_s = line[5:].strip()
                    if payload_s == "[DONE]":
                        continue
                    obj = json.loads(payload_s)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    if obj.get("engine_stats"):
                        engine_stats = obj["engine_stats"]
                    for ch in obj.get("choices", []):
                        d = ch.get("delta", {})
                        if d.get("reasoning_content"):
                            reasoning += d["reasoning_content"]
                        if d.get("content"):
                            content += d["content"]
                    send_chunk(obj)
            conn.close()
        except (OSError, ConnectionError) as e:
            send_chunk({"error": str(e)})
            self.wfile.write(b"0\r\n\r\n")
            return

        assistant = {"role": "assistant", "content": content}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        conv["messages"].append(assistant)
        if conv.get("title") is None:
            conv["title"] = user_text.strip().splitlines()[0][:60] if user_text.strip() else "New chat"
        conv["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_conversation(conv)

        if usage and engine_stats:
            STATS.record(usage, engine_stats)
        sc = ENGINE.status().get("state_cache_dir")
        if sc:
            evict_state_cache_if_over_budget(sc)

        self.wfile.write(b"0\r\n\r\n")


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # http.server.HTTPServer.server_bind() calls socket.getfqdn(host), a REVERSE DNS lookup
        # that can take tens of seconds (or longer) on a machine whose network policy makes that
        # kind of lookup slow -- measured here at 35s for '127.0.0.1'. Nothing in this app reads
        # server_name for anything but logging, so skip the lookup and use the bind address as-is.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def main():
    host = "0.0.0.0" if CFG.get("external") else "127.0.0.1"
    port = CFG.get("studio_port", 7860)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Engine Studio: http://127.0.0.1:{port}  (bind: {host})")
    if CFG.get("external"):
        print(f"  reachable on the LAN at: http://{lan_ip()}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        TUNNEL.stop()
        ENGINE.stop()


if __name__ == "__main__":
    main()
