# SPDX-License-Identifier: MIT
"""Centralize the container CLI and rootless runtime detection.

Rootless hosts may not have `docker` on PATH (or `docker` may be `podman`),
and Podman lacks `docker buildx bake`. This module picks the container CLI,
points `podman compose` at Docker Compose v2, and detects the container daemon
and subordinate ID ranges.

Without a subordinate ID range a rootless runtime has a single-UID user
namespace, where the kernel rejects `chown`/`lchown` and
`setuid`/`setgid`/`setgroups` to any non-zero ID with `EINVAL`/`EPERM`. That
breaks `apt-get`, `tar`, `useradd`, and `nix build` inside `docker build`. In
that mode this module generates:
1. A seccomp profile that returns `SCMP_ACT_ERRNO(0)` (no-op success) for the
   25 UID/GID-changing syscalls so ownership/user switches succeed as UID 0;
2. An apt config snippet (`APT::Sandbox::User "root";`) so `apt` does not fail
   its post-`setresgid` group-membership verification;
3. A nix config snippet (`build-users-group =` / `sandbox = false`) so
   `nix build` does not attempt to `kill(-1, SIGKILL)` as `nixbld1` while still
   running as UID 0.
"""

from __future__ import annotations

import functools
import getpass
import json
import os
import pwd
import shutil
import subprocess
from pathlib import Path

FAKE_OK_SYSCALLS: frozenset[str] = frozenset(
    {
        "chown",
        "chown32",
        "fchown",
        "fchown32",
        "fchownat",
        "lchown",
        "lchown32",
        "setuid",
        "setuid32",
        "setgid",
        "setgid32",
        "setreuid",
        "setreuid32",
        "setregid",
        "setregid32",
        "setresuid",
        "setresuid32",
        "setresgid",
        "setresgid32",
        "setfsuid",
        "setfsuid32",
        "setfsgid",
        "setfsgid32",
        "setgroups",
        "setgroups32",
    }
)

_SYSTEM_SECCOMP_CANDIDATES = (
    Path("/usr/share/containers/seccomp.json"),
    Path("/etc/containers/seccomp.json"),
)


def docker_bin() -> str:
    """Container CLI binary (`docker`, falling back to `podman` if `docker` is not on PATH)."""
    if shutil.which("docker") or not shutil.which("podman"):
        return "docker"
    return "podman"


# Cached: this runs the CLI and is checked before every compose command, and the
# CLI cannot change while oss-crs is running.
@functools.cache
def _cli_is_podman() -> bool:
    """Whether the container CLI is Podman, including a `docker` wrapper for it."""
    try:
        result = subprocess.run(
            [docker_bin(), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.lower().startswith("podman")


_COMPOSE_PLUGIN_DIRS = (
    Path.home() / ".docker" / "cli-plugins",
    Path("/usr/local/lib/docker/cli-plugins"),
    Path("/usr/local/libexec/docker/cli-plugins"),
    Path("/usr/lib/docker/cli-plugins"),
    Path("/usr/libexec/docker/cli-plugins"),
)


def _is_compose_v2(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "version", "--short"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    version = result.stdout.strip().lstrip("v")
    return result.returncode == 0 and bool(version) and not version.startswith("1.")


# Cached: this runs each candidate binary and is checked before every compose
# command, and the installed binaries cannot change while oss-crs is running.
@functools.cache
def compose_v2_provider() -> str | None:
    """Docker Compose v2 binary for `podman compose`, or None if there is none.

    Honours `PODMAN_COMPOSE_PROVIDER`; otherwise looks for the Docker CLI
    plugin and then `docker-compose` on PATH, skipping Compose v1.
    """
    configured = os.environ.get("PODMAN_COMPOSE_PROVIDER")
    if configured:
        return configured
    candidates = [str(d / "docker-compose") for d in _COMPOSE_PLUGIN_DIRS]
    if on_path := shutil.which("docker-compose"):
        candidates.append(on_path)
    for candidate in candidates:
        if os.access(candidate, os.X_OK) and _is_compose_v2(candidate):
            return candidate
    return None


def docker_cmd(*args: str) -> list[str]:
    """Build a container CLI invocation using the detected container binary.

    Raises:
        RuntimeError: for `compose` on Podman when no Docker Compose v2 is installed.
    """
    if args and args[0] == "compose" and _cli_is_podman():
        provider = compose_v2_provider()
        if provider is None:
            raise RuntimeError(
                "Podman needs Docker Compose v2 for `compose`; install it "
                "(see docs/setup.md) or set PODMAN_COMPOSE_PROVIDER to its path."
            )
        os.environ.setdefault("PODMAN_COMPOSE_PROVIDER", provider)
        os.environ.setdefault("PODMAN_COMPOSE_WARNING_LOGS", "false")
    return [docker_bin(), *args]


def is_podman_cli() -> bool:
    """Whether the container CLI is Podman, including a `docker` wrapper for it."""
    return _cli_is_podman()


def bake_supported() -> bool:
    """Whether the container CLI provides `buildx bake`.

    Podman's `buildx` is an alias for `podman build` and has no bake, so it is
    detected without running anything. For Docker, `buildx bake --help` exits
    nonzero when the buildx plugin (or its bake subcommand) is missing.
    """
    if _cli_is_podman():
        return False
    try:
        result = subprocess.run(
            docker_cmd("buildx", "bake", "--help"),
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


_SUBUID_PATH = Path("/etc/subuid")
_SUBGID_PATH = Path("/etc/subgid")
_NSSWITCH_PATH = Path("/etc/nsswitch.conf")


def _subid_file_has_entry(path: Path, names: tuple[str, ...]) -> bool:
    try:
        for line in path.read_text().splitlines():
            name, _, _rest = line.strip().partition(":")
            if name and name in names:
                return True
    except OSError:
        pass
    return False


def _nss_delegates_subid() -> bool:
    try:
        for line in _NSSWITCH_PATH.read_text().splitlines():
            if line.lstrip().startswith("subid:"):
                sources = line.partition(":")[2].split()
                return any(src != "files" for src in sources)
    except OSError:
        pass
    return False


def _getsubids_has_entry(user: str, group: bool) -> bool:
    cmd = ["getsubids", *(["-g"] if group else []), user]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def have_subuid_range() -> bool:
    """Whether the current user has both a subordinate UID and GID range.

    Checks /etc/subuid + /etc/subgid for the user/uid, and falls back to
    `getsubids` (which also sees NSS-provided ranges, e.g. SSSD/FreeIPA) when
    /etc/nsswitch.conf delegates `subid:` to a non-files source.
    """
    try:
        user = getpass.getuser()
    except Exception:
        try:
            user = pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            user = ""
    names = tuple(n for n in (user, str(os.getuid())) if n)
    uid_ok = _subid_file_has_entry(_SUBUID_PATH, names)
    gid_ok = _subid_file_has_entry(_SUBGID_PATH, names)
    if (uid_ok and gid_ok) or not user or not _nss_delegates_subid():
        return uid_ok and gid_ok
    uid_ok = uid_ok or _getsubids_has_entry(user, group=False)
    gid_ok = gid_ok or _getsubids_has_entry(user, group=True)
    return uid_ok and gid_ok


def describe_daemon() -> tuple[str, str]:
    """Identify the container daemon this process would talk to.

    Read-only: inspects `DOCKER_HOST` and well-known socket paths, never
    contacts a daemon. Returns `(kind, detail)` with kind one of
    `"rootless-docker"`, `"podman"`, `"rootful-docker"`, `"custom"` or
    `"none"`.
    """
    host = os.environ.get("DOCKER_HOST", "")
    if host:
        if "podman" in host:
            return "podman", host
        if "/run/user/" in host or "rootless" in host:
            return "rootless-docker", host
        return "custom", host
    xdg = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    rootless_sock = os.path.join(xdg, "docker.sock")
    if os.path.exists(rootless_sock):
        return "rootless-docker", f"unix://{rootless_sock}"
    podman_sock = os.path.join(xdg, "podman", "podman.sock")
    if os.path.exists(podman_sock):
        return "podman", f"unix://{podman_sock}"
    if os.path.exists("/var/run/docker.sock"):
        return "rootful-docker", "unix:///var/run/docker.sock"
    if _cli_is_podman():
        cli = shutil.which(docker_bin()) or docker_bin()
        return "podman", f"daemonless ({cli})"
    return "none", "no DOCKER_HOST and no daemon socket found"


def is_rootless_runtime() -> bool:
    """Whether containers run under a rootless runtime (rootless Docker or non-root Podman)."""
    if os.getuid() == 0:
        return False
    kind, _ = describe_daemon()
    return kind in ("podman", "rootless-docker") or _cli_is_podman()


def export_docker_host() -> None:
    """Point the Docker SDK at a rootless daemon socket when DOCKER_HOST is unset."""
    if os.environ.get("DOCKER_HOST"):
        return
    kind, detail = describe_daemon()
    if kind in ("podman", "rootless-docker") and detail.startswith("unix://"):
        os.environ["DOCKER_HOST"] = detail


# Cached: this probes the daemon and subordinate ID ranges and is checked before
# every build and run, and neither can change while oss-crs is running.
@functools.cache
def single_uid_mode_enabled() -> bool:
    """Whether builds and runs need the single-UID seccomp profile.

    True for a rootless runtime when the user has no subordinate ID range.
    """
    return is_rootless_runtime() and not have_subuid_range()


def _build_single_uid_seccomp_profile() -> dict:
    """Build an OCI seccomp profile that spoofs UID/GID syscalls with errno=0."""
    base: dict | None = None
    for cand in _SYSTEM_SECCOMP_CANDIDATES:
        if cand.is_file():
            try:
                base = json.loads(cand.read_text())
                break
            except Exception:
                continue
    if base is None:
        return {
            "defaultAction": "SCMP_ACT_ALLOW",
            "syscalls": [
                {
                    "names": sorted(FAKE_OK_SYSCALLS),
                    "action": "SCMP_ACT_ERRNO",
                    "args": [],
                    "errnoRet": 0,
                }
            ],
        }
    new_syscalls = []
    for entry in base.get("syscalls", []):
        if entry.get("action") == "SCMP_ACT_ALLOW":
            names = [n for n in entry.get("names", []) if n not in FAKE_OK_SYSCALLS]
            if names:
                new_syscalls.append({**entry, "names": names})
        else:
            new_syscalls.append(entry)
    new_syscalls.append(
        {
            "names": sorted(FAKE_OK_SYSCALLS),
            "action": "SCMP_ACT_ERRNO",
            "args": [],
            "comment": "Spoof UID/GID changes as no-op success in single-UID userns",
            "includes": {},
            "excludes": {},
            "errnoRet": 0,
        }
    )
    return {**base, "syscalls": new_syscalls}


def _single_uid_runtime_dir() -> Path:
    base = Path(__file__).resolve().parents[2] / ".oss-crs-workdir" / "single-uid"
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_single_uid_files() -> tuple[Path, Path, Path]:
    """Materialize (seccomp.json, apt.conf, nix.conf) for single-UID builds."""
    d = _single_uid_runtime_dir()
    seccomp_path = d / "seccomp.json"
    apt_conf = d / "apt-single-uid.conf"
    nix_conf = d / "nix.conf"
    if not seccomp_path.is_file():
        seccomp_path.write_text(json.dumps(_build_single_uid_seccomp_profile()))
    if not apt_conf.is_file():
        apt_conf.write_text('APT::Sandbox::User "root";\n')
    if not nix_conf.is_file():
        nix_conf.write_text("build-users-group =\nsandbox = false\n")
    return seccomp_path, apt_conf, nix_conf


def single_uid_build_flags(*, include_nix: bool = False) -> list[str]:
    """Extra `docker build` / `docker run` flags for single-UID rootless mode."""
    if not single_uid_mode_enabled():
        return []
    seccomp_path, apt_conf, nix_conf = ensure_single_uid_files()
    flags = [
        "--security-opt",
        f"seccomp={seccomp_path}",
        "-v",
        f"{apt_conf}:/etc/apt/apt.conf.d/99-oss-crs-single-uid.conf:ro",
    ]
    if include_nix:
        flags += ["-v", f"{nix_conf}:/etc/nix/nix.conf:ro"]
    return flags
