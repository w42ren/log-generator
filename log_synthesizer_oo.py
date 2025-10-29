#!/usr/bin/env python3
"""
Object-oriented log synthesizer

- Extensible "line" hierarchy for different log types (auth.log, access.log, etc.)
- Pluggable generators that emit LogLine objects
- LineCollection handles ordering, shuffling, and writing to disk

Usage examples:
  python log_synthesizer_oo.py --type auth --out /mnt/data/auth_synthetic.log --days 3 --hosts web01,db01
  python log_synthesizer_oo.py --type access --out /mnt/data/access_synthetic.log --days 2 --hosts web01 --seed 42

This refactor keeps format-specific logic in dedicated classes and makes it easy to add new
log types by implementing a new LogLine subclass and a matching generator.
"""
from __future__ import annotations

import abc
import argparse
import dataclasses
import datetime as dt
import ipaddress
import os
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# =============================
# Core line abstractions
# =============================

@dataclasses.dataclass(order=True)
class LogLine(abc.ABC):
    """Abstract base for a *single* log entry.

    The `order=True` dataclass option lets us sort by the first field (`ts`) naturally.
    """
    ts: dt.datetime
    host: str

    @abc.abstractmethod
    def render(self) -> str:
        """Render the line exactly as it would appear in the target log file."""
        raise NotImplementedError

    # Optional: override in subclasses if there is additional structure to validate
    def validate(self) -> None:
        if not isinstance(self.ts, dt.datetime):
            raise TypeError("ts must be a datetime")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")


# =============================
# Auth.log implementation
# =============================

@dataclasses.dataclass(order=True)
class AuthLogLine(LogLine):
    process: str
    pid: Optional[int]
    message: str

    def render(self) -> str:
        # /var/log/auth.log canonical prefix: "Mon DD HH:MM:SS HOST PROC[PID]: message"
        # Note: auth.log omits the year and timezone.
        ts_str = self.ts.strftime("%b %d %H:%M:%S")
        proc = self.process
        if self.pid is not None:
            proc = f"{proc}[{self.pid}]"
        return f"{ts_str} {self.host} {proc}: {self.message}"


class AuthLogGenerator:
    """Generate synthetic SSH auth events for one or more hosts."""

    def __init__(
        self,
        hosts: Sequence[str],
        start: dt.datetime,
        days: int = 1,
        seed: Optional[int] = None,
        failure_rate: float = 0.8,
        mean_events_per_hour: float = 6.0,
    ) -> None:
        self.hosts = list(hosts)
        self.start = start.replace(minute=0, second=0, microsecond=0)
        self.days = max(1, days)
        self.rng = random.Random(seed)
        self.failure_rate = max(0.0, min(1.0, failure_rate))
        self.lambda_per_hour = max(0.0, mean_events_per_hour)

    def _choose_ip(self) -> str:
        # Mix of RFC1918 and public-ish looking IPs for realism
        candidate_nets = [
            ipaddress.ip_network("10.0.0.0/24"),
            ipaddress.ip_network("192.168.1.0/24"),
            ipaddress.ip_network("172.16.0.0/24"),
            ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
        ]
        net = self.rng.choice(candidate_nets)
        # Avoid network and broadcast
        host_int = self.rng.randrange(1, net.num_addresses - 1)
        return str(net.network_address + host_int)

    def _poisson(self, lam: float) -> int:
        # Knuth's algorithm for small integer counts per hour
        L = self.rng.expovariate(lam) if lam > 0 else float("inf")
        # But we just want a simple approximate Poisson draw; using Python's random module only,
        # we can approximate by summing exponentials. To keep it simple, use a bounded geometric-like draw.
        # For typical use, a rounded normal approximation works fine too; keep it simple:
        mean = lam
        std = lam ** 0.5
        k = int(round(self.rng.gauss(mean, std)))
        return max(0, k)

    def _message(self, is_failure: bool) -> str:
        if is_failure:
            user = self.rng.choice(["admin", "root", "test", "pi", "oracle", "guest", "ubuntu"])  
            return f"Failed password for {user} from {self._choose_ip()} port {self.rng.randrange(2000,65000)} ssh2"
        # success
        user = self.rng.choice(["web", "ansible", "backup", "deployer", "admin"]) 
        return f"Accepted password for {user} from {self._choose_ip()} port {self.rng.randrange(2000,65000)} ssh2"

    def generate(self) -> Iterable[AuthLogLine]:
        # Iterate per host per hour; within each hour, emit N events at random seconds
        end = self.start + dt.timedelta(days=self.days)
        cur = self.start
        while cur < end:
            for host in self.hosts:
                n = self._poisson(self.lambda_per_hour)
                for _ in range(n):
                    sec = self.rng.randrange(0, 3600)
                    ts = cur + dt.timedelta(seconds=sec)
                    is_failure = self.rng.random() < self.failure_rate
                    pid = self.rng.randrange(100, 10000)
                    yield AuthLogLine(
                        ts=ts,
                        host=host,
                        process="sshd",
                        pid=pid,
                        message=self._message(is_failure),
                    )
            cur += dt.timedelta(hours=1)


# =============================
# Web access.log (Common Log Format) implementation
# =============================

COMMON_LOG_TS_FMT = "%d/%b/%Y:%H:%M:%S %z"  # e.g., 10/Oct/2000:13:55:36 -0700

@dataclasses.dataclass(order=True)
class AccessLogLine(LogLine):
    ip: str
    method: str
    path: str
    status: int
    length: int
    user_ident: str = "-"
    user_auth: str = "-"
    referer: Optional[str] = None
    agent: Optional[str] = None
    tzinfo: dt.tzinfo = dt.timezone.utc

    def render(self) -> str:
        # Common Log Format with optional referer/agent as Combined Log Format
        # 127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326
        tsz = self.ts.astimezone(self.tzinfo).strftime("%d/%b/%Y:%H:%M:%S %z")
        req = f"{self.method} {self.path} HTTP/1.1"
        core = f"{self.ip} {self.user_ident} {self.user_auth} [{tsz}] \"{req}\" {self.status} {self.length}"
        if self.referer or self.agent:
            ref = self.referer or "-"
            ag = self.agent or "-"
            return f"{core} \"{ref}\" \"{ag}\""
        return core


class AccessLogGenerator:
    """Generate synthetic web access log lines in (Combined) Common Log Format."""

    AGENTS = [
        "Mozilla/5.0",
        "curl/8.4.0",
        "Chrome/120.0",
        "Safari/17.0",
        "PostmanRuntime/7.40.0",
    ]
    PATHS = ["/", "/index.html", "/login", "/api/v1/items", "/static/app.js", "/healthz", "/admin"]

    def __init__(
        self,
        host: str,
        start: dt.datetime,
        days: int = 1,
        seed: Optional[int] = None,
        mean_hits_per_minute: float = 5.0,
        error_rate: float = 0.02,
        tzinfo: dt.tzinfo = dt.timezone.utc,
    ) -> None:
        self.host = host
        self.start = start.replace(second=0, microsecond=0)
        self.days = max(1, days)
        self.rng = random.Random(seed)
        self.lambda_per_min = max(0.0, mean_hits_per_minute)
        self.error_rate = max(0.0, min(1.0, error_rate))
        self.tzinfo = tzinfo

    def _ip(self) -> str:
        net = ipaddress.ip_network("198.51.100.0/24")  # TEST-NET-2
        host_int = self.rng.randrange(1, net.num_addresses - 1)
        return str(net.network_address + host_int)

    def _status(self) -> int:
        if self.rng.random() < self.error_rate:
            return self.rng.choice([400, 401, 403, 404, 500, 502, 503])
        return 200

    def _method(self) -> str:
        return self.rng.choices(["GET", "POST", "PUT", "DELETE"], weights=[0.75, 0.2, 0.03, 0.02])[0]

    def generate(self) -> Iterable[AccessLogLine]:
        end = self.start + dt.timedelta(days=self.days)
        cur = self.start
        while cur < end:
            # Number of hits this minute
            mean = self.lambda_per_min
            std = mean ** 0.5
            n = max(0, int(round(self.rng.gauss(mean, std))))
            for _ in range(n):
                sec = self.rng.randrange(0, 60)
                ts = cur + dt.timedelta(seconds=sec)
                method = self._method()
                path = self.rng.choice(self.PATHS)
                status = self._status()
                size = 0 if status in (204, 304) else self.rng.randrange(128, 32768)
                yield AccessLogLine(
                    ts=ts,
                    host=self.host,
                    ip=self._ip(),
                    method=method,
                    path=path,
                    status=status,
                    length=size,
                    user_ident="-",
                    user_auth="-",
                    referer=self.rng.choice([None, "https://example.com/"] * 3),
                    agent=self.rng.choice(self.AGENTS),
                    tzinfo=self.tzinfo,
                )
            cur += dt.timedelta(minutes=1)


# =============================
# Line collection and I/O
# =============================

class LineCollection:
    """Container for LogLine objects with helpers for ordering and writing."""

    def __init__(self, lines: Optional[Iterable[LogLine]] = None) -> None:
        self._lines: List[LogLine] = list(lines) if lines else []

    def add(self, *lines: LogLine) -> None:
        self._lines.extend(lines)

    def extend(self, lines: Iterable[LogLine]) -> None:
        self._lines.extend(lines)

    def sort(self) -> None:
        # Relies on dataclass(order=True) which sorts by (ts, host, ...)
        self._lines.sort()

    def shuffle(self, seed: Optional[int] = None) -> None:
        rng = random.Random(seed)
        rng.shuffle(self._lines)

    def __len__(self) -> int:
        return len(self._lines)

    def __iter__(self):
        return iter(self._lines)

    def write(self, path: os.PathLike | str, encoding: str = "utf-8") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding=encoding, newline="\n") as f:
            for line in self._lines:
                f.write(line.render() + "\n")


# =============================
# CLI
# =============================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Synthetic log generator (OO)")
    ap.add_argument("--type", choices=["auth", "access"], default="auth", help="Log type to generate")
    ap.add_argument("--out", required=True, help="Output path for the log file")
    ap.add_argument("--hosts", default="web01", help="Comma-separated hostnames (auth uses all; access uses the first)")
    ap.add_argument("--days", type=int, default=1, help="Number of days of data to generate")
    ap.add_argument("--start", default=None, help="Start timestamp ISO (default: today 00:00:00)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle lines before sort (debugging)")
    ap.add_argument("--nosort", action="store_true", help="Do NOT sort chronologically before writing")
    ap.add_argument("--tz", default="+0000", help='Timezone offset for access logs in "+HHMM" or "-HHMM"')
    return ap.parse_args(argv)


def parse_tz(offset: str) -> dt.tzinfo:
    m = re.fullmatch(r"([+-])(\d{2})(\d{2})", offset)
    if not m:
        raise ValueError("Timezone offset must be like +0000 or -0530")
    sign, hh, mm = m.groups()
    delta = dt.timedelta(hours=int(hh), minutes=int(mm))
    if sign == "-":
        delta = -delta
    return dt.timezone(delta)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # Determine start
    if args.start:
        start = dt.datetime.fromisoformat(args.start)
    else:
        today = dt.date.today()
        start = dt.datetime.combine(today, dt.time(0, 0, 0))

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hosts:
        raise SystemExit("At least one host is required")

    coll = LineCollection()

    if args.type == "auth":
        gen = AuthLogGenerator(hosts=hosts, start=start, days=args.days, seed=args.seed)
        coll.extend(gen.generate())
    elif args.type == "access":
        tzinfo = parse_tz(args.tz)
        gen = AccessLogGenerator(host=hosts[0], start=start, days=args.days, seed=args.seed, tzinfo=tzinfo)
        coll.extend(gen.generate())
    else:
        raise SystemExit(f"Unknown log type: {args.type}")

    if args.shuffle:
        coll.shuffle(seed=args.seed)

    if not args.nosort:
        coll.sort()

    coll.write(args.out)
    print(f"Wrote {len(coll)} lines to {args.out}")


if __name__ == "__main__":
    main()
