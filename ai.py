#!/usr/bin/env python3
"""AI Workflow Assistant for local Ollama models.

This is intentionally a single-file, standard-library-only CLI.  It keeps the
LLM on a short leash: the model may propose complete file contents, but this
program never writes a model proposal without an explicit human approval.

Install for a Termux shell (from a checkout or after copying this file):

    mkdir -p ~/bin
    cp ai.py ~/bin/ai.py
    chmod +x ~/bin/ai.py

The default endpoint is Ollama's local /api/generate endpoint.  See /help for
commands and the environment variables near the bottom of this file.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Small terminal helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def color(text: str, code: str, enabled: bool | None = None) -> str:
    if enabled is None:
        enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and not os.environ.get("AI_NO_COLOR")
    return f"{code}{text}{RESET}" if enabled else text


def now_iso() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat()


def clip(text: str, limit: int, marker: str = "\n… [kırpıldı] …\n") -> str:
    """Keep prompts and saved sessions bounded while preserving both ends."""
    text = str(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(marker) + 2:
        return text[:limit]
    left = max(1, (limit - len(marker)) // 2)
    right = max(1, limit - len(marker) - left)
    return text[:left] + marker + text[-right:]


def safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def prompt_yes_no(message: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(message + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in {"y", "yes", "e", "evet", "ok", "onay"}


def shorten_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


# ---------------------------------------------------------------------------
# Filesystem and project context
# ---------------------------------------------------------------------------

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".cache",
    ".idea",
    ".vscode",
}

TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".sh",
    ".ps1",
    ".sql",
    ".html",
    ".css",
    ".xml",
    ".csv",
    ".env.example",
}

SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "secrets.json",
}


def is_sensitive_path(path: Path) -> bool:
    if os.environ.get("AI_ALLOW_SENSITIVE", "").lower() in {"1", "true", "yes"}:
        return False
    name = path.name.lower()
    return (
        name in SENSITIVE_NAMES
        or name.endswith((".pem", ".key", ".p12", ".pfx"))
        or "secret" in name
        or "token" in name
    )


def is_probably_text(data: bytes) -> bool:
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def read_text(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes()
    if not is_probably_text(data):
        raise ValueError(f"ikili dosya okunamaz: {path}")
    text = data.decode("utf-8")
    return clip(text, limit) if limit is not None else text


def project_files(root: Path, max_files: int = 500) -> list[Path]:
    result: list[Path] = []
    try:
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".ai_"))
            for name in sorted(files):
                path = Path(current) / name
                if path.is_symlink() or path.is_file():
                    result.append(path)
                    if len(result) >= max_files:
                        return result
    except OSError:
        pass
    return result


def build_project_tree(root: Path, max_files: int = 350, max_chars: int = 10000) -> str:
    rows: list[str] = []
    files = project_files(root, max_files=max_files)
    for path in files:
        try:
            rel = path.relative_to(root).as_posix()
            size = path.stat().st_size
        except OSError:
            continue
        rows.append(f"{rel} ({size} B)")
    if len(files) >= max_files:
        rows.append("… [dosya listesi sınırı] …")
    return clip("\n".join(rows) or "(proje klasörü boş veya okunamıyor)", max_chars)


def collect_files_context(
    root: Path,
    paths: Iterable[str | Path],
    *,
    per_file: int = 9000,
    total: int = 28000,
) -> str:
    """Read explicitly selected files, never silently walking secrets."""
    chunks: list[str] = []
    used = 0
    for raw in paths:
        path = raw if isinstance(raw, Path) else root / raw
        try:
            path = path.resolve()
            rel = path.relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            continue
        if not path.is_file() or is_sensitive_path(path):
            continue
        try:
            content = read_text(path, per_file)
        except (OSError, ValueError):
            continue
        block = f"\n--- FILE: {rel} ---\n{content}\n--- END FILE ---\n"
        if used + len(block) > total:
            remaining = total - used
            if remaining > 100:
                chunks.append(clip(block, remaining))
            break
        chunks.append(block)
        used += len(block)
    return "".join(chunks) or "(seçili dosya içeriği yok)"


def discover_test_command(root: Path) -> str:
    configured = os.environ.get("AI_TEST_CMD") or os.environ.get("AI_TEST_COMMAND")
    if configured:
        return configured
    candidates = (
        ("main_test.py", "python3 main_test.py"),
        ("test.py", "python3 test.py"),
        ("pytest.ini", "python3 -m pytest -q"),
        ("pyproject.toml", "python3 -m pytest -q"),
        ("tests", "python3 -m unittest discover"),
    )
    for marker, command in candidates:
        if (root / marker).exists():
            return command
    return ""


def ensure_plan_checkboxes(content: str) -> str:
    """Make the plan contract usable even when a small model returns a list."""
    if re.search(r"^\s*-\s*\[[ xX]\]", content, flags=re.MULTILINE):
        return content
    lines = content.splitlines(keepends=True)
    changed = False
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:[-*]|\d+[.)])\s+(.*?)(\r?\n)?$", line)
        if not match:
            continue
        indent, body, ending = match.groups()
        lines[index] = f"{indent}- [ ] {body}{ending or ''}"
        changed = True
    if changed:
        return "".join(lines)
    # Do not discard a useful prose plan. Add a visible checkbox section so
    # /plan check has a stable target.
    suffix = "\n" if content and not content.endswith("\n") else ""
    return content + suffix + "\n## Adımlar\n\n- [ ] Yukarıdaki planı gözden geçir ve adımları netleştir.\n"


def relative_path(root: Path, raw: str) -> str:
    """Validate an LLM/user path and return a normalized project-relative path."""
    value = raw.strip().replace("\\", "/")
    if not value or "\x00" in value:
        raise ValueError("geçersiz dosya yolu")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        raise ValueError("mutlak dosya yolu kullanılamaz; proje içi göreli yol kullanın")
    # Resolve after joining so ../ cannot escape the project.
    resolved = (root / candidate).resolve()
    try:
        rel = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("dosya yolu proje klasörü dışına çıkıyor") from exc
    if rel == "." or rel.startswith(".git/") or rel == ".git":
        raise ValueError(".git içeriği değiştirilemez")
    if is_sensitive_path(resolved) and os.environ.get("AI_ALLOW_SENSITIVE", "").lower() not in {"1", "true", "yes"}:
        raise ValueError(f"korunan/gizli dosyaya erişim kapalı: {rel} (gerekirse AI_ALLOW_SENSITIVE=1)")
    return rel


# ---------------------------------------------------------------------------
# Ollama transport
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Very small urllib client for Ollama's generate API."""

    def __init__(
        self,
        endpoint: str | None = None,
        timeout: float | None = None,
        output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        base = endpoint or os.environ.get("AI_OLLAMA_URL", "http://localhost:11434/api/generate")
        self.endpoint = base.rstrip("/")
        self.timeout = timeout if timeout is not None else safe_float(os.environ.get("AI_OLLAMA_TIMEOUT"), 300.0)
        self.output_tokens = output_tokens if output_tokens is not None else safe_int(os.environ.get("AI_MAX_OUTPUT_TOKENS"), 1200)
        self.temperature = temperature if temperature is not None else safe_float(os.environ.get("AI_TEMPERATURE"), 0.15)

    def generate(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        temperature: float | None = None,
        output_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "keep_alive": os.environ.get("AI_OLLAMA_KEEP_ALIVE", "5m"),
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.output_tokens if output_tokens is None else output_tokens,
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise OllamaError(f"Ollama HTTP {exc.code}: {clip(detail, 500)}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Ollama'ya bağlanılamadı ({self.endpoint}). Ollama çalışıyor mu? Ayrıntı: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError("Ollama isteği zaman aşımına uğradı") from exc
        except OSError as exc:
            raise OllamaError(f"Ollama isteği başarısız: {exc}") from exc

        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama geçersiz JSON döndürdü: {clip(raw, 500)}") from exc
        if not isinstance(document, dict):
            raise OllamaError("Ollama yanıtı nesne değil")
        if document.get("error"):
            raise OllamaError(str(document["error"]))
        result = document.get("response")
        if not isinstance(result, str):
            raise OllamaError("Ollama yanıtında 'response' alanı yok")
        return result.strip()


# ---------------------------------------------------------------------------
# Session and proposal data
# ---------------------------------------------------------------------------

@dataclass
class ChangeProposal:
    path: str
    action: str
    content: str = ""
    summary: str = ""
    base_exists: bool = False
    base_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action,
            "content": self.content,
            "summary": self.summary,
            "base_exists": self.base_exists,
            "base_hash": self.base_hash,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ChangeProposal":
        if not isinstance(value, dict):
            raise ValueError("değişiklik nesne değil")
        return cls(
            path=str(value.get("path", "")),
            action=str(value.get("action", "modify")),
            content=str(value.get("content", value.get("new_content", ""))),
            summary=str(value.get("summary", value.get("description", ""))),
            base_exists=bool(value.get("base_exists", False)),
            base_hash=str(value.get("base_hash", "")),
        )


@dataclass
class SessionState:
    session_id: str = "default"
    project_root: str = ""
    model: str = "qwen2.5-coder:3b"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    summary: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    pending_changes: list[ChangeProposal] = field(default_factory=list)
    last_undo: list[dict[str, Any]] = field(default_factory=list)
    plan_path: str = "PLAN.md"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "project_root": self.project_root,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "messages": self.messages,
            "focus": self.focus,
            "pending_changes": [item.to_dict() for item in self.pending_changes],
            "last_undo": self.last_undo,
            "plan_path": self.plan_path,
        }

    @classmethod
    def from_dict(cls, value: Any, fallback_id: str = "default") -> "SessionState":
        if not isinstance(value, dict):
            raise ValueError("session JSON nesne değil")
        pending: list[ChangeProposal] = []
        for item in value.get("pending_changes", []) or []:
            try:
                pending.append(ChangeProposal.from_dict(item))
            except ValueError:
                continue
        messages = value.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        normalized_messages: list[dict[str, Any]] = []
        for item in messages:
            if isinstance(item, dict) and isinstance(item.get("content", ""), str):
                normalized_messages.append(
                    {
                        "role": str(item.get("role", "user")),
                        "content": item.get("content", ""),
                        "time": str(item.get("time", "")),
                    }
                )
        focus = value.get("focus", [])
        if not isinstance(focus, list):
            focus = []
        return cls(
            session_id=str(value.get("session_id", fallback_id)),
            project_root=str(value.get("project_root", "")),
            model=str(value.get("model", "qwen2.5-coder:3b")),
            created_at=str(value.get("created_at", now_iso())),
            updated_at=str(value.get("updated_at", now_iso())),
            summary=str(value.get("summary", "")),
            messages=normalized_messages,
            focus=[str(item) for item in focus],
            pending_changes=pending,
            last_undo=value.get("last_undo", []) if isinstance(value.get("last_undo", []), list) else [],
            plan_path=str(value.get("plan_path", "PLAN.md")),
        )


class SessionStore:
    """JSON session files under ~/.ai_sessions (or AI_SESSION_DIR)."""

    def __init__(self, directory: str | Path | None = None) -> None:
        configured = directory or os.environ.get("AI_SESSION_DIR", "~/.ai_sessions")
        self.directory = Path(configured).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_name(name: str) -> str:
        name = name.strip()
        if name.endswith(".json"):
            name = name[:-5]
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
        name = name.strip(".-")
        return name or "default"

    def path_for(self, name: str) -> Path:
        return self.directory / f"{self.safe_name(name)}.json"

    def save(self, state: SessionState, name: str | None = None) -> Path:
        if name:
            state.session_id = self.safe_name(name)
        else:
            state.session_id = self.safe_name(state.session_id)
        state.updated_at = now_iso()
        target = self.path_for(state.session_id)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(self.directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        return target

    def load(self, name: str) -> SessionState:
        path = self.path_for(name)
        with path.open("r", encoding="utf-8") as handle:
            return SessionState.from_dict(json.load(handle), fallback_id=self.safe_name(name))

    def list_sessions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                rows.append(
                    {
                        "session_id": str(data.get("session_id", path.stem)),
                        "title": str(data.get("title", data.get("session_id", path.stem))),
                        "project_root": str(data.get("project_root", "")),
                        "model": str(data.get("model", "")),
                        "updated_at": str(data.get("updated_at", "")),
                        "messages": len(data.get("messages", []) or []),
                        "path": str(path),
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError):
                rows.append({"session_id": path.stem, "title": "(bozuk JSON)", "path": str(path)})
        return sorted(rows, key=lambda row: str(row.get("updated_at", "")), reverse=True)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    reply: str
    changes: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    parse_note: str = ""


def _balanced_json_candidates(text: str) -> Iterable[str]:
    """Yield JSON-looking object slices without assuming markdown formatting."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield text[index : index + end]


def parse_file_blocks(text: str) -> list[dict[str, Any]]:
    """Fallback for small models that emit FILE headings instead of valid JSON."""
    pattern = re.compile(
        r"(?:^|\n)\s*(?:###?\s*)?(?:FILE|DOSYA)\s*:\s*([^\n]+)\n"
        r"(?:```[^\n]*\n)?(.*?)(?:\n```)?(?=\n\s*(?:###?\s*)?(?:FILE|DOSYA)\s*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    changes: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        path = match.group(1).strip().strip("`* ")
        content = match.group(2)
        if content.endswith("\n```"):
            content = content[:-4]
        changes.append({"path": path, "action": "modify", "content": content, "summary": "model FILE bloğu"})
    return changes


def parse_model_response(raw: str) -> ModelResult:
    text = raw.strip()
    for candidate in _balanced_json_candidates(text):
        try:
            document = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(document, dict):
            continue
        reply = document.get("reply", document.get("message", document.get("answer", "")))
        changes = document.get("changes", document.get("files", []))
        if not isinstance(reply, str):
            reply = str(reply)
        if not isinstance(changes, list):
            changes = []
        return ModelResult(reply=reply.strip(), changes=changes, raw=raw, parse_note="JSON")

    # Some models wrap JSON in a fenced block but emit a literal newline inside
    # a JSON string; the FILE fallback is safer than guessing an edit.
    blocks = parse_file_blocks(text)
    if blocks:
        reply = "Model JSON yerine dosya blokları döndürdü; öneri güvenli biçimde ayrıştırıldı."
        return ModelResult(reply=reply, changes=blocks, raw=raw, parse_note="FILE blocks")
    return ModelResult(reply=text, changes=[], raw=raw, parse_note="plain text")


# ---------------------------------------------------------------------------
# Git and command execution
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    command: str
    returncode: int
    output: str
    timed_out: bool = False


def run_shell_command(command: str, cwd: Path, timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return CommandResult(command, completed.returncode, output, False)
    except subprocess.TimeoutExpired as exc:
        output = ""
        if isinstance(exc.stdout, bytes):
            output += exc.stdout.decode("utf-8", errors="replace")
        elif exc.stdout:
            output += str(exc.stdout)
        if isinstance(exc.stderr, bytes):
            output += exc.stderr.decode("utf-8", errors="replace")
        elif exc.stderr:
            output += str(exc.stderr)
        return CommandResult(command, 124, output + "\n[timeout]", True)
    except OSError as exc:
        return CommandResult(command, 127, str(exc), False)


class GitManager:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, args: Sequence[str], timeout: float = 60.0) -> CommandResult:
        command = " ".join(shlex.quote(str(arg)) for arg in args)
        try:
            completed = subprocess.run(
                list(args),
                cwd=str(self.root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            return CommandResult(command, completed.returncode, (completed.stdout or "") + (completed.stderr or ""))
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(command, 127, str(exc))

    def is_repo(self) -> bool:
        return self.run(["git", "rev-parse", "--is-inside-work-tree"]).returncode == 0

    def status(self) -> str:
        return self.run(["git", "status", "--short"]).output.strip()

    def diff(self, cached: bool = False) -> str:
        args = ["git", "diff"]
        if cached:
            args.append("--cached")
        return self.run(args).output

    def branch(self) -> str:
        return self.run(["git", "branch", "--show-current"]).output.strip()

    def remote(self) -> str:
        return self.run(["git", "remote", "get-url", "origin"]).output.strip()


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """Sen yerel bir yazılım projesinde çalışan, dikkatli bir AI iş akışı asistanısın.
Kullanıcıya Türkçe, kısa ve uygulanabilir cevap ver; kod, dosya yolu ve terminal komutlarını İngilizce/orijinal haliyle bırak.

GÜVENLİK KURALLARI:
- Dosyayı, git'i veya komutu kendin çalıştırmış gibi asla iddia etme.
- Dosya değişikliği gerektiğinde yalnızca aşağıdaki JSON sözleşmesine göre tam yeni dosya içeriği öner.
- Kısmi diff, `...`, `TODO` veya "dosyanın geri kalanı" kullanma; tam içeriği yaz.
- Dosya yolları proje köküne göre göreli olmalı. `.git` veya proje dışına yazma önerme.
- Kullanıcı onayı olmadan hiçbir değişiklik uygulanmış sayılmaz.
- Emin olmadığın davranışı doğrulanmış gerçek gibi sunma; soru sor veya varsayımı belirt.

YANIT SÖZLEŞMESİ (çok önemli):
Her yanıtta markdown çiti olmadan tam olarak tek bir JSON nesnesi döndür:
{"reply":"Türkçe açıklama","changes":[]}
Değişiklik gerekiyorsa changes içine şu nesneleri koy:
{"path":"relative/path.py","action":"modify|create|delete","summary":"kısa açıklama","content":"dosyanın TAM yeni içeriği"}
Silme işleminde content boş olabilir. Soru/analiz yanıtlarında changes boş liste olsun.
JSON string kaçışlarını doğru yap. Cevap kısa ve net olsun.
"""

SUMMARY_SYSTEM_PROMPT = """Bir CLI kod asistanı oturumunu sıkıştırıyorsun. Türkçe kısa ama bilgi kaybını en aza indiren bir özet yaz.
Hedefi, kullanıcı kararlarını, kesin dosya yollarını, önerilen/uygulanan değişiklikleri, test çıktılarındaki hataları,
plan adımlarını ve açık soruları koru. Kodun tamamını kopyalama; önemli sembol ve komutları koru.
Yalnızca düz metin özet döndür, JSON veya markdown çiti kullanma."""

COMMIT_SYSTEM_PROMPT = """Git diff'ine bakarak tek satırlık Conventional Commit mesajı üret.
Yalnızca İngilizce mesajı döndür, başına/sonuna tırnak, markdown veya açıklama koyma.
72 karakteri geçmemeye çalış. Örnek: feat(auth): add retry timeout."""


class AIWorkflowAssistant:
    def __init__(
        self,
        project_root: str | Path = ".",
        *,
        model: str | None = None,
        session_name: str = "default",
        client: OllamaClient | None = None,
        store: SessionStore | None = None,
        auto_load: bool = False,
        auto_save: bool = True,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store or SessionStore()
        self.client = client or OllamaClient()
        self.auto_save = auto_save
        self.context_limit = safe_int(os.environ.get("AI_CONTEXT_CHARS"), 26000)
        self.summary_keep = max(4, safe_int(os.environ.get("AI_SUMMARY_KEEP_MESSAGES"), 8))
        self.test_timeout = safe_float(os.environ.get("AI_TEST_TIMEOUT"), 600.0)
        self.command_timeout = safe_float(os.environ.get("AI_COMMAND_TIMEOUT"), 120.0)
        self.project_tree = build_project_tree(self.root)
        self.git = GitManager(self.root)
        self.test_command = discover_test_command(self.root)
        self.last_proposal_note = ""
        self.session = SessionState(
            session_id=SessionStore.safe_name(session_name),
            project_root=str(self.root),
            model=model or os.environ.get("AI_MODEL", "qwen2.5-coder:3b"),
        )
        self.loaded = False
        if auto_load and self.store.path_for(session_name).exists():
            try:
                self.load_session(session_name, quiet=True)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(color(f"[uyarı] varsayılan session yüklenemedi: {exc}", YELLOW))

    # ----- state and context -------------------------------------------------

    @property
    def model(self) -> str:
        return self.session.model

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.session.messages

    @property
    def pending_changes(self) -> list[ChangeProposal]:
        return self.session.pending_changes

    def save_session(self, name: str | None = None, quiet: bool = False) -> Path:
        self.session.project_root = str(self.root)
        path = self.store.save(self.session, name)
        if not quiet:
            print(color(f"Session kaydedildi: {path}", GREEN))
        return path

    def maybe_save(self) -> None:
        if self.auto_save:
            try:
                self.save_session(quiet=True)
            except OSError as exc:
                print(color(f"[uyarı] otomatik session kaydı başarısız: {exc}", YELLOW))

    def load_session(self, name: str, quiet: bool = False) -> None:
        state = self.store.load(name)
        saved_root = Path(state.project_root).expanduser().resolve() if state.project_root else self.root
        if saved_root != self.root:
            if state.pending_changes:
                print(color("[uyarı] session başka bir projeden geliyor; bekleyen değişiklikler temizlendi.", YELLOW))
                state.pending_changes = []
            if not quiet:
                print(color(f"[uyarı] session proje yolu farklı: {saved_root}", YELLOW))
        state.project_root = str(self.root)
        self.session = state
        self.session.session_id = SessionStore.safe_name(name)
        self.loaded = True
        if not quiet:
            print(f"Session yüklendi: {self.session.session_id} ({len(self.messages)} mesaj)")
        self.maybe_save()

    def new_session(self, name: str | None = None) -> None:
        session_id = SessionStore.safe_name(name or f"session-{_datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.session = SessionState(
            session_id=session_id,
            project_root=str(self.root),
            model=self.model,
        )
        print(f"Yeni session: {session_id}")
        self.maybe_save()

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        item: dict[str, Any] = {"role": role, "content": content, "time": now_iso()}
        item.update(extra)
        self.messages.append(item)

    def estimate_context_chars(self) -> int:
        messages = sum(len(str(item.get("content", ""))) + 32 for item in self.messages)
        pending = sum(len(item.content) + len(item.path) + len(item.summary) + 80 for item in self.pending_changes)
        return len(BASE_SYSTEM_PROMPT) + len(self.project_tree) + len(self.session.summary) + messages + pending

    def focus_context(self) -> str:
        if not self.session.focus:
            return "(odak modu kapalı; yalnızca istenen bağlamı kullan)"
        return "\n".join(f"- {path}" for path in self.session.focus)

    def current_plan_context(self) -> str:
        path = self.root / self.session.plan_path
        if not path.is_file() or is_sensitive_path(path):
            return "(PLAN.md yok)"
        try:
            return clip(read_text(path), 6000)
        except (OSError, ValueError):
            return "(PLAN.md okunamadı)"

    def build_system_prompt(self) -> str:
        focused_contents = "(odak dosyası içeriği yok)"
        if self.session.focus:
            focused_contents = collect_files_context(
                self.root,
                self.session.focus,
                per_file=7000,
                total=14000,
            )
        focus_rule = (
            "Focus açıksa yalnızca bu dosyalar hakkında inceleme/değişiklik öner; başka dosyaları "
            "kullanıcı açıkça istemedikçe kapsam dışı say."
            if self.session.focus
            else "Focus kapalı; yalnızca istekte sağlanan dosya içeriğini kullan."
        )
        extra = f"""
PROJE KÖKÜ: {self.root}
MODEL: {self.model}
PROJE DOSYA HARİTASI:
{self.project_tree}
ODAK DOSYALARI:
{self.focus_context()}
ODAK KURALI: {focus_rule}
ODAK DOSYA İÇERİKLERİ:
{focused_contents}
AKTİF PLAN:
{self.current_plan_context()}
"""
        return BASE_SYSTEM_PROMPT + extra

    def build_prompt(self) -> str:
        sections: list[str] = [
            "Aşağıdaki bağlam önceki oturumdan gelir. En yeni kullanıcı isteğine yanıt ver.",
        ]
        if self.session.summary:
            sections.append("\n[ÖNCEKİ OTURUM ÖZETİ]\n" + self.session.summary)
        if self.messages:
            transcript: list[str] = []
            for item in self.messages:
                role = str(item.get("role", "user")).upper()
                transcript.append(f"[{role}]\n{item.get('content', '')}")
            sections.append("\n[OTURUM MESAJLARI]\n" + "\n\n".join(transcript))
        if self.pending_changes:
            pending = "\n".join(
                f"- {change.action} {change.path}: {change.summary or 'öneri beklemede'}"
                for change in self.pending_changes
            )
            sections.append("\n[UYGULANMAMIŞ ÖNERİLER - bunlar henüz dosyaya yazılmadı]\n" + pending)
        return "\n".join(sections)

    def compact_if_needed(self) -> None:
        if self.estimate_context_chars() <= self.context_limit:
            return
        if len(self.messages) <= 2:
            # A single gigantic test/file message cannot be sent whole to a
            # 3B model. Preserve both ends (the end normally contains the
            # error) and keep every very recent message represented.
            per_message = max(700, self.context_limit // max(2, len(self.messages)))
            for item in self.messages:
                item["content"] = clip(str(item.get("content", "")), per_message)
            self.maybe_save()
            return
        # For a very small configured context, compact even before the normal
        # eight-message window is full. Otherwise recent turns stay intact.
        keep_count = self.summary_keep if len(self.messages) > self.summary_keep else max(2, len(self.messages) // 2)
        old = self.messages[:-keep_count]
        keep = self.messages[-keep_count:]
        transcript = "\n\n".join(
            f"[{item.get('role', 'user').upper()}]\n{item.get('content', '')}" for item in old
        )
        if self.session.summary:
            transcript = "ÖNCEKİ ÖZET:\n" + self.session.summary + "\n\n" + transcript
        prompt = clip(
            "Oturum bağlamını sıkıştır. Önemli kararları ve açık işleri koru:\n\n" + transcript,
            max(self.context_limit * 2, 18000),
        )
        try:
            summary = self.client.generate(
                model=self.model,
                system=SUMMARY_SYSTEM_PROMPT,
                prompt=prompt,
                temperature=0.0,
                output_tokens=min(900, self.client.output_tokens),
            )
            self.session.summary = clip(summary, max(3000, self.context_limit // 3))
        except OllamaError:
            # Context management must not make the CLI unusable when Ollama is
            # temporarily unavailable. This fallback is deterministic.
            self.session.summary = clip(transcript, max(3000, self.context_limit // 3))
        self.session.messages = keep
        self.maybe_save()
        print(color("[context] Eski mesajlar kısa özete dönüştürüldü.", DIM))

    # ----- model requests ---------------------------------------------------

    def ask_model(
        self,
        user_text: str,
        *,
        extra_context: str = "",
        show_reply: bool = True,
        add_user: bool = True,
    ) -> ModelResult | None:
        if add_user:
            self.add_message("user", user_text)
        if extra_context:
            # Tool/file context is part of the conversation and is persisted;
            # it therefore remains available after a session reload.
            self.add_message("context", extra_context)
        self.compact_if_needed()
        try:
            raw = self.client.generate(
                model=self.model,
                system=self.build_system_prompt(),
                prompt=self.build_prompt(),
            )
        except OllamaError as exc:
            print(color(f"[Ollama hata] {exc}", RED))
            self.add_message("tool", f"Ollama hatası: {exc}")
            self.maybe_save()
            return None
        result = parse_model_response(raw)
        proposal_note = ""
        if result.changes:
            proposal_note = "\n[Dosya önerileri: " + ", ".join(str(item.get("path", "?")) for item in result.changes) + "]"
        self.add_message("assistant", result.reply + proposal_note)
        if show_reply and result.reply:
            print(color("AI: ", BOLD + CYAN), end="")
            print(result.reply)
        if result.parse_note == "plain text" and result.changes == []:
            # Not an error for questions, but visible for edit commands.
            self.last_proposal_note = "Model JSON sözleşmesine uymadı; dosya yazılmadı."
        self.maybe_save()
        return result

    def stage_model_result(self, result: ModelResult, expected_paths: Iterable[str] = ()) -> list[ChangeProposal]:
        staged: list[ChangeProposal] = []
        expected = set(expected_paths)
        for raw_change in result.changes:
            if not isinstance(raw_change, dict):
                continue
            raw_path = raw_change.get("path", raw_change.get("file", ""))
            try:
                path = relative_path(self.root, str(raw_path))
            except ValueError as exc:
                print(color(f"[öneri reddedildi] {exc}", RED))
                continue
            action = str(raw_change.get("action", "modify")).lower().strip()
            if action in {"write", "edit", "update"}:
                action = "modify"
            if action in {"new", "add"}:
                action = "create"
            if action not in {"modify", "create", "delete"}:
                print(color(f"[öneri reddedildi] bilinmeyen action: {action}", RED))
                continue
            content_value = raw_change.get("content", raw_change.get("new_content", raw_change.get("contents", "")))
            if content_value is None:
                content_value = ""
            if not isinstance(content_value, str):
                content_value = str(content_value)
            full_path = self.root / path
            exists = full_path.exists()
            if exists and full_path.is_dir():
                print(color(f"[öneri reddedildi] yol klasör: {path}", RED))
                continue
            try:
                current = read_text(full_path) if exists else ""
            except (OSError, ValueError) as exc:
                print(color(f"[öneri reddedildi] {path}: {exc}", RED))
                continue
            if action == "create" and exists:
                # Do not silently turn a create into an overwrite.
                print(color(f"[öneri reddedildi] create hedefi zaten var: {path}", RED))
                continue
            if action == "modify" and not exists:
                # A weak model often omits `action:create` for a new file. It
                # is safe to normalize only in this direction: a missing path
                # can never overwrite existing user data.
                action = "create"
                print(color(f"[öneri] hedef yok; modify -> create: {path}", YELLOW))
            if action == "delete" and not exists:
                print(color(f"[öneri atlandı] zaten yok: {path}", YELLOW))
                continue
            summary = str(raw_change.get("summary", raw_change.get("description", ""))).strip()
            if expected and path not in expected:
                summary = (summary + " [ek dosya]").strip()
            proposal = ChangeProposal(
                path=path,
                action=action,
                content="" if action == "delete" else content_value,
                summary=summary,
                base_exists=exists,
                base_hash=sha256_text(current) if exists else "",
            )
            if action != "delete" and exists and current == proposal.content:
                print(color(f"[öneri atlandı] değişiklik yok: {path}", YELLOW))
                continue
            staged.append(proposal)

        # A later proposal for a path replaces an older pending proposal, which
        # keeps /plan followed by a refinement comprehensible.
        if staged:
            by_path = {item.path: item for item in self.pending_changes}
            for item in staged:
                by_path[item.path] = item
            self.session.pending_changes = list(by_path.values())
            self.last_proposal_note = ""
        return staged

    # ----- diff and approval ------------------------------------------------

    def _contents_for_diff(self, proposal: ChangeProposal) -> tuple[str, str]:
        path = self.root / proposal.path
        try:
            old = read_text(path) if path.exists() else ""
        except (OSError, ValueError):
            old = "[ikili veya okunamayan dosya]\n"
        new = "" if proposal.action == "delete" else proposal.content
        return old, new

    def diff_for(self, proposal: ChangeProposal) -> str:
        old, new = self._contents_for_diff(proposal)
        # Use line strings without terminators and let difflib terminate its
        # headers.  With keepends=True + lineterm="" the first header and the
        # next header are printed as ``--- old+++ new`` on narrow terminals.
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        if proposal.action == "create":
            from_name = "/dev/null"
            to_name = proposal.path
        elif proposal.action == "delete":
            from_name = proposal.path
            to_name = "/dev/null"
        else:
            from_name = proposal.path
            to_name = proposal.path
        return "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=from_name,
                tofile=to_name,
                lineterm="\\n",
            )
        )

    def print_diff(self, changes: Sequence[ChangeProposal] | None = None) -> None:
        items = list(changes if changes is not None else self.pending_changes)
        if not items:
            print("Bekleyen değişiklik yok.")
            return
        shown = 0
        max_chars = safe_int(os.environ.get("AI_MAX_DIFF_CHARS"), 60000)
        enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and not os.environ.get("AI_NO_COLOR")
        for proposal in items:
            diff = self.diff_for(proposal)
            if shown + len(diff) > max_chars:
                print(color("… diff sınırı nedeniyle devamı gösterilmedi …", YELLOW, enabled))
                break
            for line in diff.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    print(color(line, BOLD + CYAN, enabled))
                elif line.startswith("+"):
                    print(color(line, GREEN, enabled))
                elif line.startswith("-"):
                    print(color(line, RED, enabled))
                elif line.startswith("@@"):
                    print(color(line, CYAN, enabled))
                else:
                    print(line)
            shown += len(diff)

    @staticmethod
    def change_stats(proposal: ChangeProposal) -> tuple[int, int]:
        old_lines = proposal.content.splitlines() if proposal.action == "create" else []
        # This method is replaced with an accurate calculation in summary();
        # keeping it independent makes it safe for missing files.
        return (len(old_lines), 0)

    def summary_for_changes(self, changes: Sequence[ChangeProposal] | None = None) -> str:
        items = list(changes if changes is not None else self.pending_changes)
        if not items:
            return "Bekleyen değişiklik yok."
        lines = [f"{len(items)} dosyada öneri bekliyor:"]
        for item in items:
            old, new = self._contents_for_diff(item)
            diff_lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""))
            added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++") )
            removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
            verb = {"create": "yeni", "delete": "sil", "modify": "güncelle"}.get(item.action, item.action)
            detail = f" — {item.summary}" if item.summary else ""
            lines.append(f"  {item.path:<42} {verb} (+{added}/-{removed}){detail}")
        return "\n".join(lines)

    def check_proposal_conflicts(self, changes: Sequence[ChangeProposal]) -> list[str]:
        conflicts: list[str] = []
        for item in changes:
            path = self.root / item.path
            exists = path.exists()
            if exists:
                try:
                    current = read_text(path)
                except (OSError, ValueError) as exc:
                    conflicts.append(f"{item.path}: okunamıyor ({exc})")
                    continue
                current_hash = sha256_text(current)
            else:
                current_hash = ""
            if exists != item.base_exists or (exists and current_hash != item.base_hash):
                conflicts.append(f"{item.path}: öneri hazırlanırken dosya değişmiş")
        return conflicts

    def review_pending(self) -> bool:
        if not self.pending_changes:
            print("Bekleyen değişiklik yok.")
            return False
        print(color(self.summary_for_changes(), BOLD))
        while True:
            try:
                choice = input("[y] hepsini uygula [n] iptal [d] diff [s] tek tek: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if choice in {"y", "yes", "e", "evet"}:
                return self.apply_changes(list(self.pending_changes))
            if choice in {"n", "no", "h", "hayır", "hayir"}:
                self.session.pending_changes = []
                self.maybe_save()
                print("Öneri iptal edildi; dosyalara dokunulmadı.")
                return False
            if choice in {"d", "diff"}:
                self.print_diff()
                continue
            if choice in {"s", "select", "tek"}:
                selected: list[ChangeProposal] = []
                for item in self.pending_changes:
                    print(f"\n{item.path}")
                    self.print_diff([item])
                    if prompt_yes_no("Bu dosyayı uygula?", False):
                        selected.append(item)
                if not selected:
                    print("Hiçbir dosya seçilmedi.")
                    return False
                if self.apply_changes(selected):
                    selected_paths = {item.path for item in selected}
                    self.session.pending_changes = [item for item in self.pending_changes if item.path not in selected_paths]
                    self.maybe_save()
                    return True
                return False
            print("Seçim: y, n, d veya s")

    def stage_and_review(self, result: ModelResult, expected_paths: Iterable[str] = ()) -> bool:
        staged = self.stage_model_result(result, expected_paths=expected_paths)
        if result.changes and not staged:
            print(color("Model değişiklik önerdi ama güvenlik doğrulamasından geçmedi; yazma yapılmadı.", RED))
            self.maybe_save()
            return False
        if not staged:
            if result.reply:
                print(color("Değişiklik önerisi yok; dosyalara dokunulmadı.", DIM))
            self.maybe_save()
            return False
        print(color(self.summary_for_changes(staged), BOLD + YELLOW))
        return self.review_pending()

    # ----- safe application and undo ----------------------------------------

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        old_mode: int | None = None
        try:
            old_mode = path.stat().st_mode & 0o777
        except OSError:
            pass
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            # mkstemp creates 0600 files. Preserve an existing executable bit
            # and make newly generated source files normally usable.
            os.chmod(path, old_mode if old_mode is not None else 0o644)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def apply_changes(self, changes: Sequence[ChangeProposal]) -> bool:
        changes = list(changes)
        if not changes:
            return False
        conflicts = self.check_proposal_conflicts(changes)
        if conflicts:
            print(color("Çakışma: " + "; ".join(conflicts), RED))
            print("Dosya dışarıdan değişmiş. Yeni bir öneri üretin; eski öneri zorla uygulanmadı.")
            return False

        undo_rows: list[dict[str, Any]] = []
        backups: list[tuple[Path, bytes]] = []
        created: list[Path] = []
        try:
            # Back up everything before the first write. A .bak is intentionally
            # plain and portable on Android/Termux.
            for item in changes:
                path = self.root / item.path
                if path.exists():
                    before = path.read_bytes()
                    backup = Path(str(path) + ".bak")
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
                    backups.append((path, before))
                    undo_rows.append({"path": item.path, "existed": True, "backup": str(backup)})
                else:
                    undo_rows.append({"path": item.path, "existed": False, "backup": ""})

            for item in changes:
                path = self.root / item.path
                if item.action == "delete":
                    if path.exists():
                        path.unlink()
                else:
                    if not path.exists():
                        created.append(path)
                    self._atomic_write(path, item.content)
        except (OSError, UnicodeError) as exc:
            print(color(f"Uygulama başarısız, geri alınıyor: {exc}", RED))
            for path in created:
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass
            for path, before in backups:
                try:
                    self._atomic_write(path, before.decode("utf-8"))
                except (OSError, UnicodeDecodeError):
                    try:
                        path.write_bytes(before)
                    except OSError:
                        pass
            return False

        self.session.last_undo = undo_rows
        applied_paths = {item.path for item in changes}
        self.session.pending_changes = [item for item in self.pending_changes if item.path not in applied_paths]
        self.project_tree = build_project_tree(self.root)
        self.maybe_save()
        print(color(f"{len(changes)} dosya güncellendi. Yedekler .bak olarak saklandı.", GREEN))
        return True

    def undo(self) -> bool:
        rows = list(self.session.last_undo)
        if not rows:
            print("Geri alınacak son uygulama yok.")
            return False
        print("Son uygulama geri alınacak:")
        for row in rows:
            print(f"  - {row.get('path')}")
        if not prompt_yes_no("Devam?", False):
            print("Geri alma iptal edildi.")
            return False
        restored = 0
        for row in rows:
            try:
                rel = relative_path(self.root, str(row.get("path", "")))
                path = self.root / rel
            except ValueError:
                continue
            existed = bool(row.get("existed", False))
            backup_name = str(row.get("backup", ""))
            if existed and backup_name:
                backup = Path(backup_name)
                if backup.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, path)
                    restored += 1
            elif not existed and path.exists():
                path.unlink()
                restored += 1
        self.session.last_undo = []
        self.project_tree = build_project_tree(self.root)
        self.maybe_save()
        print(color(f"{restored} dosya geri alındı.", GREEN))
        return True

    # ----- command implementations -----------------------------------------

    def print_banner(self) -> None:
        plan_state = "var" if (self.root / self.session.plan_path).exists() else "yok"
        test = self.test_command or "ayarlanmadı (AI_TEST_CMD)"
        print(color("AI Workflow Assistant", BOLD + CYAN) + f" | model: {self.model} | session: {self.session.session_id}")
        print(f"Proje: {self.root}")
        print(f"Aktif plan: {plan_state} | test: {test}")
        if self.loaded:
            print(color(f"Session devam ediyor: {len(self.messages)} mesaj", DIM))

    def print_help(self) -> None:
        print(
            """\nKomutlar:
  /help                         Bu yardım
  /model [ad]                   Modeli göster/değiştir (AI_MODEL varsayılanı)
  /read <dosya> [soru]          Dosyayı oku ve soru sor
  /scan <soru>                  Projeyi/dosyaları tarayıp soru sor
  /edit <dosya> <talimat>       Tam içerik önerisi üret; onaysız yazmaz
  /new <dosya> <talimat>        Yeni dosya önerisi üret; onaysız yazmaz
  /plan [talimat]               PLAN.md öner/güncelle; onay kapısından geçer
  /plan show                    Planı göster
  /plan check <no...>           Plan maddelerini [x] yap (sonra onay)
  /apply                        Bekleyen önerileri tekrar gözden geçir/uygula
  /diff                         Bekleyen önerilerin renkli diff'i
  /test [komut]                 Test komutunu çalıştır, çıktıyı hafızaya ekle
  /diagnostic <talimat>         Diagnostic Python dosyası öner
  /diagnostic-run <dosya> ...   Diagnostic'i onayla ve çalıştır
  /run <komut>                  Komut çalıştır (önce onay), çıktıyı hafızaya ekle
  /focus [dosyalar...]          Odak dosyaları ayarla; /focus off ile kapat
  /save [ad]                    Session'ı ~/.ai_sessions/<ad>.json kaydet
  /load <ad>                    Session yükle
  /new-session [ad]             Yeni konuşma başlat
  /list-sessions                Kayıtlı session'ları listele
  /commit [mesaj]               AI commit mesajı öner; onayla ve commit et
  /push                         origin/<branch> için ayrıca onay iste
  /undo                         Son uygulanan dosya değişikliklerini geri al
  /status                       Git/session/diff durumunu göster
  /quit                         Kaydet ve çık

Doğal dil istekleri de mümkündür. Model dosya önerirse önce özet ve diff gösterilir;
[y/n/d/s] onayından önce hiçbir model içeriği çalışma ağacına yazılmaz.
"""
        )

    def cmd_model(self, args: str) -> None:
        name = args.strip()
        if not name:
            print(f"Model: {self.model}")
            print("Hızlı seçenekler: qwen2.5-coder:3b (hızlı), qwen2.5-coder:7b (derin)")
            return
        if any(char.isspace() for char in name) or len(name) > 200:
            print(color("Geçersiz model adı.", RED))
            return
        self.session.model = name
        mode = "hızlı" if "3b" in name.lower() else "derin" if "7b" in name.lower() else "özel"
        print(f"Model değiştirildi: {name} ({mode} mod)")
        self.maybe_save()

    def cmd_read(self, args: str) -> None:
        try:
            tokens = shlex.split(args)
        except ValueError as exc:
            print(color(f"Komut ayrıştırılamadı: {exc}", RED))
            return
        if not tokens:
            print("Kullanım: /read <dosya> [soru]")
            return
        try:
            rel = relative_path(self.root, tokens[0])
        except ValueError as exc:
            print(color(str(exc), RED))
            return
        path = self.root / rel
        if not path.is_file():
            print(color(f"Dosya yok: {rel}", RED))
            return
        try:
            content = read_text(path, safe_int(os.environ.get("AI_READ_MAX_CHARS"), 18000))
        except (OSError, ValueError) as exc:
            print(color(str(exc), RED))
            return
        question = " ".join(tokens[1:]) or "Bu dosyanın amacı, akışı ve riskleri nedir?"
        self.ask_model(
            f"{rel} dosyasını incele ve şu soruyu yanıtla: {question}",
            extra_context=f"[READ FILE {rel}]\n{content}",
        )

    def cmd_scan(self, args: str) -> None:
        question = args.strip() or "Projeyi kısaca tara; önemli bileşenleri, riskleri ve açık sorunları belirt."
        files = [path for path in project_files(self.root, 500) if path.suffix.lower() in TEXT_SUFFIXES and not is_sensitive_path(path)]
        context = collect_files_context(self.root, files, per_file=4500, total=safe_int(os.environ.get("AI_SCAN_MAX_CHARS"), 30000))
        self.ask_model(f"Proje taraması yap ve şu soruyu yanıtla: {question}", extra_context=context)

    def cmd_edit_or_new(self, args: str, create: bool) -> None:
        try:
            tokens = shlex.split(args)
        except ValueError as exc:
            print(color(f"Komut ayrıştırılamadı: {exc}", RED))
            return
        if len(tokens) < 2:
            print("Kullanım: /new <dosya> <talimat> veya /edit <dosya> <talimat>")
            return
        try:
            rel = relative_path(self.root, tokens[0])
        except ValueError as exc:
            print(color(str(exc), RED))
            return
        path = self.root / rel
        if create and path.exists():
            print(color(f"Yeni dosya zaten var: {rel}; /edit kullanın.", RED))
            return
        if not create and not path.is_file():
            print(color(f"Düzenlenecek dosya yok: {rel}", RED))
            return
        instruction = " ".join(tokens[1:])
        try:
            old = "(dosya yeni; mevcut içerik yok)" if create else read_text(
                path,
                safe_int(os.environ.get("AI_EDIT_MAX_CHARS"), 24000),
            )
        except (OSError, ValueError) as exc:
            print(color(str(exc), RED))
            return
        verb = "oluştur" if create else "düzenle"
        prompt = (
            f"{rel} dosyasını {verb}. Talimat: {instruction}\n"
            "Sadece gerekli dosya için tam içerik içeren güvenli JSON değişiklik önerisi döndür."
        )
        result = self.ask_model(prompt, extra_context=f"[TARGET FILE {rel}]\n{old}")
        if result:
            self.stage_and_review(result, expected_paths=[rel])

    def cmd_plan(self, args: str) -> None:
        tokens = args.strip().split()
        if tokens and tokens[0].lower() == "show":
            path = self.root / self.session.plan_path
            if not path.is_file():
                print("PLAN.md yok.")
                return
            try:
                print(read_text(path))
            except (OSError, ValueError) as exc:
                print(color(str(exc), RED))
            return
        if tokens and tokens[0].lower() in {"check", "done", "uncheck"}:
            self.cmd_plan_check(tokens[0].lower(), tokens[1:])
            return
        instruction = args.strip() or "Konuşma, test çıktısı ve proje bağlamına göre uygulanabilir bir PLAN.md oluştur veya güncelle."
        existing = self.current_plan_context()
        git_context = f"[GIT STATUS]\n{clip(self.git.status(), 4000)}\n[GIT DIFF]\n{clip(self.git.diff(), 10000)}"
        result = self.ask_model(
            f"{instruction}\nPLAN.md zorunlu format: amaç, kapsam, riskler, doğrulama komutu ve checkbox'lı adımlar (- [ ] adım).",
            extra_context=f"[CURRENT PLAN]\n{existing}\n{git_context}",
        )
        if result:
            for raw_change in result.changes:
                if isinstance(raw_change, dict) and str(raw_change.get("path", "")) == self.session.plan_path:
                    raw_change["content"] = ensure_plan_checkboxes(
                        str(raw_change.get("content", raw_change.get("new_content", "")))
                    )
            self.stage_and_review(result, expected_paths=[self.session.plan_path])

    def cmd_plan_check(self, mode: str, values: Sequence[str]) -> None:
        path = self.root / self.session.plan_path
        if not path.is_file():
            print("PLAN.md yok; önce /plan çalıştırın.")
            return
        try:
            content = read_text(path)
        except (OSError, ValueError) as exc:
            print(color(str(exc), RED))
            return
        requested: set[int] = set()
        for value in values:
            if "-" in value:
                try:
                    start, end = value.split("-", 1)
                    requested.update(range(int(start), int(end) + 1))
                except ValueError:
                    pass
            else:
                try:
                    requested.add(int(value))
                except ValueError:
                    pass
        if not requested:
            print("Kullanım: /plan check 1 2 veya /plan uncheck 2")
            return
        index = 0
        lines = content.splitlines(keepends=True)
        changed = False
        for number, line in enumerate(lines):
            if not re.match(r"^\s*-\s*\[[ xX]\]", line):
                continue
            index += 1
            if index not in requested:
                continue
            marker = "[x]" if mode in {"check", "done"} else "[ ]"
            updated = re.sub(r"^([ \t]*-[ \t]*)\[[ xX]\]", rf"\1{marker}", line, count=1)
            if updated != line:
                lines[number] = updated
                changed = True
        if not changed:
            print("İstenen plan maddesinde değişiklik yok veya madde numarası bulunamadı.")
            return
        proposal = ChangeProposal(
            path=self.session.plan_path,
            action="modify",
            content="".join(lines),
            summary=f"PLAN.md maddeleri güncellendi: {', '.join(map(str, sorted(requested)))}",
            base_exists=True,
            base_hash=sha256_text(content),
        )
        self.session.pending_changes = [item for item in self.pending_changes if item.path != proposal.path] + [proposal]
        self.review_pending()

    def cmd_apply(self) -> None:
        self.review_pending()

    def cmd_test(self, args: str) -> None:
        command = args.strip() or self.test_command
        if not command:
            print("Test komutu ayarlı değil. AI_TEST_CMD='python3 main_test.py' ayarlayın veya /test <komut> kullanın.")
            return
        print(f"Çalıştırılıyor: {command}")
        result = run_shell_command(command, self.root, self.test_timeout)
        output = result.output or "(çıktı yok)"
        shown = clip(output, safe_int(os.environ.get("AI_TEST_DISPLAY_CHARS"), 50000))
        print(shown)
        lines = len(output.splitlines())
        status = "başarılı" if result.returncode == 0 else f"başarısız (exit {result.returncode})"
        if result.timed_out:
            status = "timeout"
        print(color(f"Test {status}; {lines} satır çıktı conversation'a eklendi.", GREEN if result.returncode == 0 else RED))
        self.add_message(
            "tool",
            f"TEST KOMUTU: {command}\nEXIT: {result.returncode}\nOUTPUT:\n{clip(output, safe_int(os.environ.get('AI_TEST_CONTEXT_CHARS'), 30000))}",
        )
        self.maybe_save()

    def cmd_diagnostic(self, args: str) -> None:
        instruction = args.strip()
        if not instruction:
            candidates = [
                shorten_path(path, self.root)
                for path in project_files(self.root, 500)
                if "diagnostic" in path.name.lower() and path.suffix == ".py"
            ]
            print("Diagnostic dosyaları:")
            print("\n".join(f"  - {path}" for path in candidates) or "  (yok)")
            print("Kullanım: /diagnostic <ne ölçülsün?>")
            return
        result = self.ask_model(
            "Tamamı standart Python kütüphanesiyle çalışan bir diagnostic scripti hazırla. "
            f"İstenen teşhis: {instruction}. Dosya, güvenli ve tekrar çalıştırılabilir olsun; kullanıcı girdisi alma.",
            extra_context="[DIAGNOSTIC CONSTRAINTS]\nDosya önerisini diagnostics/diagnostic_<konu>.py gibi göreli bir yola koy; tam Python içeriği üret.",
        )
        if result:
            self.stage_and_review(result)

    def cmd_diagnostic_run(self, args: str) -> None:
        try:
            tokens = shlex.split(args)
        except ValueError as exc:
            print(color(str(exc), RED))
            return
        if not tokens:
            print("Kullanım: /diagnostic-run <dosya.py> [argümanlar]")
            return
        try:
            rel = relative_path(self.root, tokens[0])
        except ValueError as exc:
            print(color(str(exc), RED))
            return
        path = self.root / rel
        if not path.is_file() or path.suffix != ".py":
            print(color(f"Python diagnostic bulunamadı: {rel}", RED))
            return
        command = " ".join([shlex.quote(sys.executable), shlex.quote(rel), *[shlex.quote(item) for item in tokens[1:]]])
        if not prompt_yes_no(f"Çalıştır: {command}?", False):
            print("Diagnostic iptal edildi.")
            return
        result = run_shell_command(command, self.root, self.command_timeout)
        print(result.output or "(çıktı yok)")
        self.add_message(
            "tool",
            f"DIAGNOSTIC: {command}\nEXIT: {result.returncode}\nOUTPUT:\n{clip(result.output, 30000)}",
        )
        self.maybe_save()

    def cmd_run(self, args: str) -> None:
        command = args.strip()
        if not command:
            print("Kullanım: /run <komut>")
            return
        if not prompt_yes_no(f"Komut çalıştırılsın mı: {command}?", False):
            print("Komut iptal edildi.")
            return
        result = run_shell_command(command, self.root, self.command_timeout)
        print(result.output or "(çıktı yok)")
        self.add_message("tool", f"KOMUT: {command}\nEXIT: {result.returncode}\nOUTPUT:\n{clip(result.output, 30000)}")
        self.maybe_save()

    def cmd_focus(self, args: str) -> None:
        try:
            tokens = shlex.split(args)
        except ValueError as exc:
            print(color(str(exc), RED))
            return
        if not tokens:
            print("Odak: " + (", ".join(self.session.focus) if self.session.focus else "kapalı"))
            return
        if tokens[0].lower() in {"off", "none", "clear", "kapat"}:
            self.session.focus = []
            print("Odak modu kapatıldı.")
            self.maybe_save()
            return
        focus: list[str] = []
        for token in tokens:
            try:
                rel = relative_path(self.root, token)
            except ValueError as exc:
                print(color(f"Odak atlandı: {exc}", RED))
                continue
            if not (self.root / rel).is_file():
                print(color(f"Odak dosyası yok: {rel}", YELLOW))
                continue
            focus.append(rel)
        self.session.focus = focus
        print("Odak dosyaları: " + (", ".join(focus) if focus else "kapalı"))
        self.maybe_save()

    def cmd_status(self) -> None:
        print(f"Model: {self.model}")
        print(f"Session: {self.session.session_id} | mesaj: {len(self.messages)} | context yaklaşık: {self.estimate_context_chars()} karakter")
        print(f"Focus: {', '.join(self.session.focus) if self.session.focus else 'kapalı'}")
        print(f"Bekleyen öneri: {len(self.pending_changes)}")
        if self.git.is_repo():
            branch = self.git.branch() or "(detached)"
            print(f"Git: {branch}")
            print(self.git.status() or "working tree temiz")
        else:
            print("Git deposu değil.")

    def cmd_commit(self, args: str) -> None:
        if not self.git.is_repo():
            print(color("Bu klasör git deposu değil.", RED))
            return
        status = self.git.status()
        diff = self.git.diff()
        cached = self.git.diff(cached=True)
        if not status and not cached:
            print("Commit edilecek değişiklik yok.")
            return
        supplied = args.strip()
        if supplied:
            message = supplied
        else:
            prompt = clip(f"STATUS:\n{status}\nDIFF:\n{diff}\nCACHED DIFF:\n{cached}", 24000)
            try:
                message = self.client.generate(
                    model=self.model,
                    system=COMMIT_SYSTEM_PROMPT,
                    prompt=prompt,
                    temperature=0.0,
                    output_tokens=80,
                ).strip().splitlines()[0]
            except (OllamaError, IndexError) as exc:
                print(color(f"Commit mesajı üretilemedi: {exc}", RED))
                return
        message = message.strip().strip('"`')
        print(f"Commit mesajı önerim:\n  {message}")
        print("Commit kapsamı: çalışma ağacındaki tüm değişiklikler (git add -A)")
        try:
            answer = input("[y] onayla [n] iptal [e] mesajı düzenle: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if answer in {"e", "edit", "düzenle"}:
            try:
                message = input("Yeni commit mesajı: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not message:
                print("Boş mesajla commit yapılmadı.")
                return
            if not prompt_yes_no("Bu mesajla commit atılsın mı?", False):
                print("Commit iptal edildi.")
                return
        elif answer not in {"y", "yes", "e", "evet"}:
            print("Commit iptal edildi.")
            return
        added = self.git.run(["git", "add", "-A"])
        if added.returncode != 0:
            print(color(f"git add başarısız:\n{added.output}", RED))
            return
        committed = self.git.run(["git", "commit", "-m", message], timeout=120.0)
        print(committed.output)
        if committed.returncode == 0:
            print(color("Commit tamamlandı.", GREEN))

    def cmd_push(self) -> None:
        if not self.git.is_repo():
            print(color("Bu klasör git deposu değil.", RED))
            return
        branch = self.git.branch()
        remote = self.git.remote()
        if not branch:
            print(color("Aktif branch yok; push yapılmadı.", RED))
            return
        if not remote:
            print(color("origin remote yok; push yapılmadı.", RED))
            return
        print(f"Push hedefi: origin/{branch} ({remote})")
        if not prompt_yes_no("Push onaylıyor musun?", False):
            print("Push iptal edildi.")
            return
        result = self.git.run(["git", "push", "origin", branch], timeout=300.0)
        print(result.output)
        print(color("Push tamamlandı." if result.returncode == 0 else "Push başarısız.", GREEN if result.returncode == 0 else RED))

    # ----- command dispatcher and loop -------------------------------------

    def dispatch(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if not stripped.startswith("/"):
            result = self.ask_model(stripped)
            if result:
                self.stage_and_review(result)
            return True
        try:
            parts = shlex.split(stripped)
        except ValueError as exc:
            print(color(f"Komut ayrıştırılamadı: {exc}", RED))
            return True
        command = parts[0].lower()
        args = stripped[len(parts[0]) :].strip()
        if command in {"/quit", "/exit", "/q"}:
            self.maybe_save()
            print("Session kaydedildi. Görüşürüz.")
            return False
        if command in {"/help", "/?"}:
            self.print_help()
        elif command == "/model":
            self.cmd_model(args)
        elif command == "/read":
            self.cmd_read(args)
        elif command == "/scan":
            self.cmd_scan(args)
        elif command == "/edit":
            self.cmd_edit_or_new(args, create=False)
        elif command == "/new":
            self.cmd_edit_or_new(args, create=True)
        elif command == "/plan":
            self.cmd_plan(args)
        elif command == "/apply":
            self.cmd_apply()
        elif command == "/diff":
            self.print_diff()
        elif command == "/test":
            self.cmd_test(args)
        elif command == "/diagnostic":
            self.cmd_diagnostic(args)
        elif command in {"/diagnostic-run", "/run-diagnostic"}:
            self.cmd_diagnostic_run(args)
        elif command == "/run":
            self.cmd_run(args)
        elif command == "/focus":
            self.cmd_focus(args)
        elif command == "/save":
            self.save_session(args or None)
        elif command == "/load":
            if not args:
                print("Kullanım: /load <ad>")
            else:
                try:
                    self.load_session(args)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    print(color(f"Session yüklenemedi: {exc}", RED))
        elif command in {"/new-session", "/newsession"}:
            self.new_session(args or None)
        elif command in {"/list-sessions", "/sessions"}:
            rows = self.store.list_sessions()
            if not rows:
                print("Kayıtlı session yok.")
            for row in rows:
                print(f"{row.get('session_id', '?'):<24} {row.get('updated_at', ''):<25} {row.get('messages', 0):>3} mesaj  {row.get('project_root', '')}")
        elif command == "/commit":
            self.cmd_commit(args)
        elif command == "/push":
            self.cmd_push()
        elif command == "/undo":
            self.undo()
        elif command == "/status":
            self.cmd_status()
        else:
            print(color(f"Bilinmeyen komut: {command}. /help yazın.", YELLOW))
        return True

    def run(self) -> None:
        self.print_banner()
        print("Yardım: /help | çıkış: /quit")
        while True:
            try:
                line = input(color("\nSen> ", BOLD + MAGENTA))
            except EOFError:
                print()
                self.maybe_save()
                break
            except KeyboardInterrupt:
                print("\nCtrl-C: çıkılıyor; session kaydediliyor.")
                self.maybe_save()
                break
            if not self.dispatch(line):
                break


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ollama tabanlı güvenli AI iş akışı asistanı")
    parser.add_argument("--project", default=".", help="Proje kökü (varsayılan: .)")
    parser.add_argument("--session", default=os.environ.get("AI_SESSION", "default"), help="Başlangıç session adı")
    parser.add_argument("--model", default=None, help="Başlangıç modeli; yoksa AI_MODEL veya qwen2.5-coder:3b")
    parser.add_argument("--resume", action="store_true", help="Başlangıç session'ı varsa otomatik yükle")
    parser.add_argument("--no-autosave", action="store_true", help="Her işlem sonrası otomatik session kaydını kapat")
    args = parser.parse_args(argv)
    assistant = AIWorkflowAssistant(
        args.project,
        model=args.model,
        session_name=args.session,
        auto_load=args.resume or os.environ.get("AI_AUTO_LOAD", "").lower() in {"1", "true", "yes"},
        auto_save=not args.no_autosave,
    )
    assistant.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
