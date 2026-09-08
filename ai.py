#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai.py — Ollama tabanlı yerel AI İş Akışı Asistanı (stdlib only)

Test → Analiz → Soru-cevap → Diagnostic → Plan → Onaylı uygulama → Git döngüsünü
yönetir. Termux / Android (8GB RAM, 3B model) hedefiyle yazılmıştır.

Kullanım:
    python3 ai.py                       # etkileşimli REPL
    python3 ai.py -m qwen2.5-coder:7b   # farklı model
    python3 ai.py -s oturum1            # farklı session
    python3 ai.py -d /path/to/project   # proje klasörü
    python3 ai.py -q "soru"             # tek atımlık soru (session'a kaydedilir)

Gereksinim: Python 3.8+ ve çalışan bir Ollama (http://localhost:11434).
Bağımlılık: YOK (sadece Python stdlib).
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------

VERSION = "2.0.0"
HOME = os.path.expanduser("~")
SESSION_DIR = os.path.join(HOME, ".ai_sessions")
CONFIG_FILE = os.path.join(HOME, ".ai_config.json")

DEFAULT_MODEL = "qwen2.5-coder:3b"
DEFAULT_OLLAMA = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_CTX = 8192

RESPONSE_RESERVE = 1400      # modelin yanitı için ayrılan token bütçesi
KEEP_RECENT = 6              # özetlemede her zaman tutulan son mesaj sayısı
SUMMARIZE_MIN_TOKENS = 700   # bu kadar token'lık eski geçmiş varsa özetle
SUMMARY_INPUT_CAP = 3500     # özetleme çağrısına verilen maksimum token
SYSTEM_TOKEN_CAP = 1500      # system prompt için yumuşak üst sınır

TOOL_HEAD = 50               # test/komut çıktısında tutulacak baş satır
TOOL_TAIL = 70               # ... ve kuyruk satırı
FOCUS_FILE_LINES = 150       # /focus dosyalarından dosya başına satır
FILE_READ_CAP = 250          # /read ile eklenen dosya satır limiti
DIFF_MAX_LINES = 240         # ekranda gösterilecek maksimum diff satırı
TREE_MAX_ENTRIES = 120       # proje ağacında gösterilecek maksimum dosya
DIAG_FILE = ".ai_diagnostic.py"
BACKUP_KEEP = 20             # session'da tutulacak maksimum undo yığını

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".idea", ".vscode", "dist", "build", ".tox", ".eggs",
    ".gradle", ".next", ".nuxt", "coverage", ".ruff_cache", ".cache",
}

USE_COLOR = (os.environ.get("AI_COLOR") == "1") or (
    sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
)


def C(code, s):
    """ANSI renk sarmalayıcı (renk kapalıysa düz metin döner)."""
    if USE_COLOR:
        return "\033[%sm%s\033[0m" % (code, s)
    return s


def bold(s):    return C("1", s)
def dim(s):     return C("2", s)
def red(s):     return C("31", s)
def green(s):   return C("32", s)
def yellow(s):  return C("33", s)
def blue(s):    return C("34", s)
def magenta(s): return C("35", s)
def cyan(s):    return C("36", s)


# ----------------------------------------------------------------------------
# Küçük yardımcılar
# ----------------------------------------------------------------------------

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def est_tokens(text):
    """Kaba token tahmini (Türkçe + kod karışımı için ~3 karakter/token)."""
    return max(1, len(text) // 3)


def kfmt(n):
    return ("%.1fk" % (n / 1000.0)) if n >= 1000 else str(n)


def slugify(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name.strip("-") or "default"


def atomic_write(path, content):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def read_text_safe(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def truncate_lines(text, head=TOOL_HEAD, tail=TOOL_TAIL):
    """Uzun çıktıları baş+kuyruk olarak kısaltır (ara satırları işaretler)."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 3:
        return text
    skipped = len(lines) - head - tail
    return "\n".join(lines[:head] +
                     ["... [%d satır atlandı] ..." % skipped] +
                     lines[-tail:])


def truncate_chars(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit // 2] + "\n...[kesildi]...\n" + text[-limit // 2:]


def ask(prompt, default="n"):
    """Kullanıcıdan tek satır onay/girdi alır; Ctrl-C/EOF → default."""
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return default


def ask_yn(prompt, default=False):
    suffix = "[Y/n]" if default else "[y/n]"
    a = ask("%s %s " % (prompt, dim(suffix))).lower()
    if not a:
        return default
    return a in ("y", "yes", "e", "evet")


def read_multiline(end_word="END"):
    """Çok satırlı yapıştırma: kullanıcı tek başına END yazınca biter."""
    print(dim("  (Yapıştır/düzenle. Bitirmek için tek başına %s yaz)" % end_word))
    lines = []
    while True:
        try:
            ln = input()
        except EOFError:
            break
        if ln.strip() == end_word:
            break
        lines.append(ln)
    return "\n".join(lines)


def info(msg):
    print(cyan("ℹ ") + msg)


def ok(msg):
    print(green("✅ ") + msg)


def warn(msg):
    print(yellow("⚠️  ") + msg)


def err(msg):
    print(red("❌ ") + msg)


# ----------------------------------------------------------------------------
# Ollama istemcisi
# ----------------------------------------------------------------------------

class OllamaError(Exception):
    pass


class Ollama:
    """Minimum Ollama REST istemcisi (urllib tabanlı, stream destekli)."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def _request(self, path, payload=None, timeout=600):
        url = self.base + path
        if payload is None:
            req = urllib.request.Request(url)
        else:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise OllamaError("Ollama HTTP %s: %s" % (e.code, body))
        except urllib.error.URLError as e:
            raise OllamaError(
                "Ollama'ya ulaşılamadı (%s).\n"
                "  Termux'ta başlat:  ollama serve\n"
                "  Sonra başka terminalde: ollama run %s" % (e.reason, DEFAULT_MODEL))
        except OSError as e:
            raise OllamaError("Bağlantı hatası: %s" % e)

    def list_models(self):
        with self._request("/api/tags", timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        return sorted(m.get("name", "?") for m in data.get("models", []))

    def chat(self, messages, model, num_ctx, stream=True, temperature=0.2,
             num_predict=None, on_delta=None):
        """
        /api/chat çağrısı. Döner: (tam_metin, kesildi_mi)
        stream=True ise on_delta(parça) ile token'ları canlı basar.
        """
        options = {"num_ctx": num_ctx, "temperature": temperature}
        if num_predict:
            options["num_predict"] = num_predict
        payload = {"model": model, "messages": messages,
                   "stream": bool(stream), "options": options}

        if not stream:
            with self._request("/api/chat", payload, timeout=900) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data.get("message", {}).get("content", ""), False

        chunks = []
        interrupted = False
        try:
            with self._request("/api/chat", payload, timeout=900) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = obj.get("message", {}).get("content", "")
                    if piece:
                        chunks.append(piece)
                        if on_delta:
                            on_delta(piece)
                    if obj.get("done"):
                        break
        except KeyboardInterrupt:
            interrupted = True
        except OllamaError:
            # Kısmi yanıt varsa onu kullan, yoksa hatayı yükselt
            if not chunks:
                raise
            interrupted = True
        return "".join(chunks), interrupted


# ----------------------------------------------------------------------------
# Model çıktısından yapı çıkarımı
# ----------------------------------------------------------------------------

FILE_BLOCK_RE = re.compile(
    r'(?:^|\n)\s*(?:\*\*)?\s*(?:DOSYA|FILE|Dosya|dosya)\s*[:：]\s*(?:\*\*)?\s*'
    r'[`"\']*([^\s`"\'*]+)[`"\']*\s*(?:\*\*)?[^\n]*'
    r'\n+```[\w.+-]*[^\n]*\n(.*?)(?:```|\Z)',
    re.S)

PYTHON_BLOCK_RE = re.compile(r'```python[^\n]*\n(.*?)(?:```|\Z)', re.S)


def parse_file_blocks(text):
    """Model yanıtındaki 'DOSYA: yol + kod bloğu' örneklerini çıkarır."""
    changes, warnings = [], []
    for m in FILE_BLOCK_RE.finditer(text or ""):
        path, content = m.group(1), m.group(2)
        p = path.strip().replace("\\", "/").lstrip("./").strip("`\"' ")
        if not p or p.startswith("/") or ".." in p.split("/"):
            warnings.append("Güvenli olmayan yol atlandı: %s" % path)
            continue
        content = content.rstrip("\n") + "\n"
        changes.append({"path": p, "content": content})
    return changes, warnings


def last_python_block(text):
    blocks = PYTHON_BLOCK_RE.findall(text or "")
    return blocks[-1].rstrip("\n") if blocks else None


# ----------------------------------------------------------------------------
# Session (kalıcı oturum) yönetimi
# ----------------------------------------------------------------------------

def new_session(name, model, num_ctx, root):
    return {
        "version": 2,
        "name": name,
        "created": now_iso(),
        "updated": now_iso(),
        "model": model,
        "num_ctx": num_ctx,
        "project_root": root,
        "test_cmd": "",
        "focus": [],
        "plan_file": "PLAN.md",
        "tree": "",
        "messages": [],        # {"role","content","kind","ts"}
        "pending": [],         # [{"path","content"}] onay bekleyen değişiklikler
        "last_applied": [],    # son /apply'de yazılan dosyalar
        "backups": [],         # undo yığını: liste listeleri
        "stats": {"turns": 0},
    }


def session_path(name):
    return os.path.join(SESSION_DIR, slugify(name) + ".json")


def save_session(s):
    os.makedirs(SESSION_DIR, exist_ok=True)
    s["updated"] = now_iso()
    atomic_write(session_path(s["name"]), json.dumps(s, ensure_ascii=False, indent=1))


def load_session(name):
    p = session_path(name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def list_sessions():
    if not os.path.isdir(SESSION_DIR):
        return []
    out = []
    for fn in os.listdir(SESSION_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSION_DIR, fn), "r", encoding="utf-8") as f:
                d = json.load(f)
            out.append((d.get("name", fn[:-5]), d.get("updated", "?"),
                        len(d.get("messages", [])), d.get("model", "?")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda x: x[1], reverse=True)


def load_config():
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            cfg = {}
    return cfg


# ----------------------------------------------------------------------------
# Asistan uygulaması
# ----------------------------------------------------------------------------

BASE_SYSTEM = """\
SEN: Kıdemli yazılım geliştirme asistanısın. Kullanıcının projesinde \
test → teşhis → plan → onaylı uygulama → git döngüsünü yönetirsin.
DİL: Türkçe yaz; kod, komut ve dosya adları İngilizce kalır.
STİL: Kısa ve net yaz, madde kullan, gereksiz giriş/kapanış yapma.
KURALLAR:
1) Dosya içeriğini görmeden kesin iddia uydurma; gerekirse /read ile paylaşmasını iste.
2) Emin olmadığında en fazla 3 kısa ve net soru sor, sonra öneri ver.
3) Hiçbir dosyaya yazdığını ASLA söyleme; değişiklikleri sadece ÖNER.
4) Dosya değişikliği önerisi SADECE şu formatta olur:
DOSYA: goreli/yol.py
```python
<dosyanın TAM yeni içeriği>
```
5) Küçük teşhis scripti için: önce "DIAGNOSTIC:" yaz, sonra tek ```python bloğu ver. Kullanıcı /diag ile çalıştırır.
6) Plan istenirse "# PLAN:" ile başlayan Markdown üret ve adımları "- [ ]" ile listele."""


class App:
    def __init__(self, root, model, num_ctx, ollama_url, session_name):
        self.root = os.path.abspath(root)
        self.ollama = Ollama(ollama_url)
        s = load_session(session_name)
        if s is None:
            s = new_session(session_name, model, num_ctx, self.root)
        else:
            if model:
                s["model"] = model
            if num_ctx:
                s["num_ctx"] = num_ctx
            s["project_root"] = self.root
        self.session = s
        if not self.session.get("tree"):
            self.session["tree"] = self.scan_tree(show=False)

    # ------------------------------------------------------------------ genel

    def save(self):
        save_session(self.session)

    def msg(self, role, content, kind="chat"):
        return {"role": role, "content": content, "kind": kind, "ts": now_iso()}

    def prompt(self):
        n = len(self.session.get("pending", []))
        tag = " " + magenta("[%d 📥 bekleyen]" % n) if n else ""
        return "\n" + bold(blue("👤 >")) + tag + " "

    def banner(self):
        s = self.session
        print()
        print(bold("🤖 AI Workflow Assistant v%s" % VERSION),
              dim("| model: %s | session: %s | ctx: %s tok"
                  % (s["model"], s["name"], kfmt(s["num_ctx"]))))
        print("📁 Proje:", self.root)
        pl = self.plan_progress()
        if pl:
            print("📋 Aktif plan:", self.session.get("plan_file"), "(%s)" % pl)
        else:
            print("📋 Aktif plan:", dim("yok"))
        try:
            self.ollama.list_models()
        except OllamaError as e:
            warn("Ollama erişilemiyor — %s" % str(e).splitlines()[0])
        print(dim("/help → komut listesi"))

    # ------------------------------------------------------- proje taraması

    def scan_tree(self, show=True):
        entries = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
            for fn in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, fn), self.root)
                entries.append(rel.replace(os.sep, "/"))
                if len(entries) >= TREE_MAX_ENTRIES:
                    break
            if len(entries) >= TREE_MAX_ENTRIES:
                break
        entries.sort()
        tree = "\n".join(entries)
        if len(entries) >= TREE_MAX_ENTRIES:
            tree += "\n... (daha fazla dosya var, /focus ile daralt)"
        self.session["tree"] = tree
        if show:
            print(cyan("📂 Proje ağacı (%d dosya):" % len(entries)))
            print(dim(tree))
        return tree

    def autodetect_test_cmd(self):
        for cand in ("main_test.py", "run_tests.py", "test_all.py", "main.py"):
            if os.path.exists(os.path.join(self.root, cand)):
                return "python3 %s" % cand
        make = read_text_safe(os.path.join(self.root, "Makefile")) or ""
        if re.search(r"^test\s*:", make, re.M):
            return "make test"
        if os.path.isdir(os.path.join(self.root, "tests")):
            return "python3 -m pytest tests -q"
        return ""

    # -------------------------------------------------------- system prompt

    def build_system_prompt(self):
        parts = [BASE_SYSTEM]
        tree = self.session.get("tree") or ""
        parts.append("## PROJE: %s\n### Dosya ağacı\n%s" % (self.root, tree))
        focus = self.session.get("focus") or []
        if focus:
            blocks, total = [], 0
            for rel in focus:
                content = read_text_safe(os.path.join(self.root, rel)) or ""
                lines = content.splitlines()
                if len(lines) > FOCUS_FILE_LINES:
                    content = "\n".join(lines[:FOCUS_FILE_LINES]) + \
                        "\n... [%d satır daha]" % (len(lines) - FOCUS_FILE_LINES)
                blocks.append("--- %s ---\n%s" % (rel, content))
                total += est_tokens(content)
                if total > 2600:
                    blocks.append("(odak dosyalarının geri kalanı bütçe nedeniyle kırpıldı)")
                    break
            parts.append("## ODAK DOSYALAR (sadece bunlar üzerinde çalış)\n" +
                         "\n".join(blocks))
        md = self.read_plan()
        if md:
            steps = plan_steps(md)
            todo = [t for d, t in steps if not d][:10]
            if todo:
                parts.append("## AKTİF PLAN (sıradaki adımlar)\n" +
                             "\n".join("- [ ] " + t for t in todo))
        sysprompt = "\n\n".join(parts)
        if est_tokens(sysprompt) > SYSTEM_TOKEN_CAP and tree:
            # ağacı kırparak yeniden dene
            short_tree = "\n".join(tree.splitlines()[:60]) + "\n...(kırpıldı)"
            sysprompt = sysprompt.replace(tree, short_tree)
        return sysprompt

    # --------------------------------------------------------- bağlam bütçesi

    def context_usage(self):
        sys_t = est_tokens(self.build_system_prompt())
        msg_t = sum(est_tokens(m["content"]) + 8
                    for m in self.session["messages"])
        return sys_t + msg_t, self.session["num_ctx"] - RESPONSE_RESERVE

    def auto_compact(self, force=False):
        """Bütçe aşılırsa eski mesajları LLM ile özetler (aider/cline taktiği:
        önce büyük araç çıktılarını kırp, sonra özetle, son mesajları koru)."""
        used, budget = self.context_usage()
        msgs = self.session["messages"]
        if not force and used <= budget:
            return
        if len(msgs) <= KEEP_RECENT:
            # son mesajlar bile sığmıyorsa büyük olanları sert kırp
            self._hard_trim(budget)
            return
        head, tail = msgs[:-KEEP_RECENT], msgs[-KEEP_RECENT:]
        head_tok = sum(est_tokens(m["content"]) for m in head)

        if head_tok >= SUMMARIZE_MIN_TOKENS or force:
            transcript = "\n".join(
                "%s: %s" % (m["role"].upper(), truncate_chars(m["content"], 900))
                for m in head)
            transcript = truncate_chars(transcript, SUMMARY_INPUT_CAP * 3)
            prompt = ("[ÖZET İSTEĞİ]\nAşağıdaki konuşma bölümünü Türkçe ve en fazla "
                      "12 satırla özetle.\nMUTLAKA KORU: dosya yolları, alınan "
                      "kararlar, açık sorular, sayısal değerler, test/commit "
                      "sonuçları.\n---\n" + transcript)
            try:
                summary, _ = self.ollama.chat(
                    [{"role": "user", "content": prompt}],
                    self.session["model"], min(4096, self.session["num_ctx"]),
                    stream=False, num_predict=320, temperature=0.1)
            except OllamaError as e:
                warn("Özetleme başarısız, eski mesajlar düşürüldü: %s" % e)
                summary = "(eski konuşma özeti alınamadı)"
            self.session["messages"] = \
                [self.msg("summary", summary.strip(), kind="summary")] + tail
            info("🧹 Bağlam sıkıştırıldı: %d eski mesaj özetlendi "
                 "(~%s tok → ~%s tok)" % (len(head), kfmt(head_tok),
                                          kfmt(est_tokens(summary))))
        else:
            self.session["messages"] = tail
            info("🧹 %d eski mesaj bütçe için düşürüldü" % len(head))
        if self.context_usage()[0] > budget:
            self._hard_trim(budget)

    def _hard_trim(self, budget):
        """Son çare: araç çıktılarını ve uzun mesajları sert kırp."""
        for m in self.session["messages"]:
            if self.context_usage()[0] <= budget:
                break
            if m["kind"] in ("tool", "event"):
                m["content"] = truncate_lines(m["content"], 15, 20)
        for m in self.session["messages"]:
            if self.context_usage()[0] <= budget:
                break
            if len(m["content"]) > 800:
                m["content"] = truncate_chars(m["content"], 800)

    # ------------------------------------------------------------- sohbet

    def build_api_messages(self):
        out = [{"role": "system", "content": self.build_system_prompt()}]
        for m in self.session["messages"]:
            if m["role"] == "assistant":
                out.append({"role": "assistant", "content": m["content"]})
            else:
                prefix = "[ÖNCEKİ KONUŞMANIN ÖZETİ]\n" if m["kind"] == "summary" else ""
                out.append({"role": "user", "content": prefix + m["content"]})
        return out

    def stream_reply(self, api_msgs, temperature=0.2, num_predict=None):
        print("\n" + bold(magenta("🤖 ")) , end="")
        first = [True]

        def on_delta(piece):
            if first[0]:
                first[0] = False
            sys.stdout.write(piece)
            sys.stdout.flush()

        text, interrupted = self.ollama.chat(
            api_msgs, self.session["model"], self.session["num_ctx"],
            stream=True, temperature=temperature, num_predict=num_predict,
            on_delta=on_delta)
        print()
        if interrupted:
            print(dim("  (✂ yanıt yarıda kesildi — devam edebilirsin)"))
        return text, interrupted

    def chat_turn(self, text):
        self.session["messages"].append(self.msg("user", text))
        self.session["stats"]["turns"] += 1
        self.auto_compact()
        try:
            api = self.build_api_messages()
        except Exception as e:
            err("Bağlam hazırlanamadı: %s" % e)
            return
        try:
            reply, interrupted = self.stream_reply(api)
        except OllamaError as e:
            err(str(e))
            self.session["messages"].pop()
            return
        content = reply + ("\n(yanıt yarıda kesildi)" if interrupted else "")
        self.session["messages"].append(self.msg("assistant", content))
        changes, warnings_ = parse_file_blocks(reply)
        for w in warnings_:
            warn(w)
        if changes:
            self.add_pending(changes)
            n = len(self.session["pending"])
            print(green("📥 %d dosya için değişiklik önerildi → incelemek için: /apply" % n))

    def add_pending(self, changes):
        by_path = {c["path"]: c for c in self.session.get("pending", [])}
        for c in changes:
            by_path[c["path"]] = c
        self.session["pending"] = list(by_path.values())

    # ------------------------------------------------------- dosya değişikliği

    def change_diff(self, ch):
        full = os.path.join(self.root, ch["path"])
        old = read_text_safe(full)
        is_new = old is None
        old = old or ""
        diff = list(difflib.unified_diff(
            old.splitlines(), ch["content"].splitlines(),
            fromfile="a/" + ch["path"], tofile="b/" + ch["path"], n=2,
            lineterm=""))
        adds = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        return diff, adds, dels, is_new

    def print_colored_diff(self, ch):
        diff, adds, dels, is_new = self.change_diff(ch)
        if is_new:
            print(bold(cyan("➕ %s (yeni dosya)" % ch["path"])))
        else:
            print(bold(cyan("📝 %s (+%d/-%d)" % (ch["path"], adds, dels))))
        shown = 0
        for ln in diff:
            if shown >= DIFF_MAX_LINES:
                print(dim("... (diff kırpıldı)"))
                break
            if ln.startswith("+++") or ln.startswith("---"):
                out = dim(ln)
            elif ln.startswith("@@"):
                out = cyan(ln)
            elif ln.startswith("+"):
                out = green(ln)
            elif ln.startswith("-"):
                out = red(ln)
            else:
                out = dim(ln)
            print(out)
            shown += 1

    def apply_changes(self, paths=None):
        pending = self.session.get("pending", [])
        todo = [c for c in pending if paths is None or c["path"] in paths]
        applied, batch = [], []
        for ch in todo:
            full = os.path.join(self.root, ch["path"])
            old = read_text_safe(full)
            was_new = old is None
            if not was_new:
                atomic_write(full + ".bak", old)
            atomic_write(full, ch["content"])
            batch.append({"path": ch["path"], "was_new": was_new})
            applied.append(ch["path"])
            print(green("  ✍ %s %s" % (ch["path"],
                                       "(yeni)" if was_new else "(güncellendi, .bak alındı)")))
        self.session["pending"] = [c for c in pending
                                   if c["path"] not in set(applied)]
        if applied:
            self.session["last_applied"] = applied
            self.session["backups"].append(batch)
            self.session["backups"] = self.session["backups"][-BACKUP_KEEP:]
            self.mark_plan_steps(applied)
            self.session["messages"].append(self.msg(
                "event",
                "[DURUM] Kullanıcı değişiklikleri onayladı ve uygulandı: " +
                ", ".join("%s (%s)" % (b["path"],
                                       "yeni" if b["was_new"] else "güncellendi")
                          for b in batch) +
                ". Şimdi test etmeyi önerebilirsin (/test).",
                kind="event"))
        return applied

    # ------------------------------------------------------------------ plan

    def plan_path(self):
        return os.path.join(self.root, self.session.get("plan_file") or "PLAN.md")

    def read_plan(self):
        return read_text_safe(self.plan_path())

    def plan_progress(self):
        md = self.read_plan()
        if not md:
            return None
        steps = plan_steps(md)
        if not steps:
            return "adım yok"
        done = sum(1 for d, _ in steps if d)
        return "%d/%d adım" % (done, len(steps))

    def mark_plan_steps(self, paths):
        md = self.read_plan()
        if not md:
            return
        out, changed = [], False
        for line in md.splitlines():
            m = re.match(r"^(\s*)- \[ \](.*)$", line)
            if m and any(p in line for p in paths):
                out.append(m.group(1) + "- [x]" + m.group(2))
                changed = True
            else:
                out.append(line)
        if changed:
            atomic_write(self.plan_path(), "\n".join(out) + "\n")

    # ------------------------------------------------------------------- git

    def git(self, *args, timeout=60):
        p = subprocess.run(["git"] + list(args), cwd=self.root,
                           capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        # DİKKAT: porcelain çıktısında satır başındaki boşluk anlamlıdır;
        # sadece sondaki boşluk/satır sonu kırpılır, baş strip edilmez.
        return p.returncode, (p.stdout or "").rstrip(), (p.stderr or "").strip()

    def is_repo(self):
        rc, out, _ = self.git("rev-parse", "--is-inside-work-tree")
        return rc == 0 and out == "true"

    # =========================================================== KOMUTLAR ===

    def cmd_help(self, arg):
        print(bold("\nKomutlar:"))
        rows = [
            ("/test [cmd | set <cmd>]", "Test komutunu çalıştır, çıktıyı conversation'a ekle"),
            ("/run <komut>",             "Tek seferlik komut çalıştır (onaylı)"),
            ("/diag",                    "Modelin son diagnostic kodunu çalıştır"),
            ("/read <dosya> [soru]",     "Dosyayı conversation'a ekle (+opsiyonel soru)"),
            ("/scan [soru]",             "Proje ağacını tara/güncelle"),
            ("/focus <d1> <d2> | off",   "Modeli belli dosyalara kilitle"),
            ("/plan [hedef]",            "PLAN.md üret/güncelle (onaylı)"),
            ("/apply",                   "Bekleyen değişiklikleri göster → onayla → uygula"),
            ("/discard",                 "Bekleyen değişiklikleri çöpe at"),
            ("/undo",                    "Son /apply'yi geri al (.bak'tan)"),
            ("/commit",                  "AI commit mesajı önersin → onayla → commit"),
            ("/push",                    "Onaylı git push"),
            ("/status",                  "Oturum, bağlam, plan ve git durumu"),
            ("/model [ad]",              "Model listele / değiştir"),
            ("/ctx <n>",                 "Context window boyutu (token)"),
            ("/compact",                 "Konuşmayı zorla özetle"),
            ("/save [ad] / /load <ad>",  "Session kaydet / yükle"),
            ("/new-session <ad>",        "Yeni session başlat"),
            ("/list-sessions",           "Session'ları listele"),
            ("/quit",                    "Çıkış"),
        ]
        for c_, d_ in rows:
            print("  " + cyan("%-26s" % c_) + dim(d_))
        print(dim("\nSerbest metin → doğrudan AI'ya gider. Onay mekanizması: "
                  "[y] onayla [n] iptal [d] diff [s] tek tek."))

    # /test ---------------------------------------------------------------
    def cmd_test(self, arg):
        arg = (arg or "").strip()
        if arg.startswith("set "):
            self.session["test_cmd"] = arg[4:].strip()
            ok("Test komutu kaydedildi: %s" % bold(self.session["test_cmd"]))
            return
        if arg == "show":
            print("test_cmd =", bold(self.session["test_cmd"] or "(atanmadı)"))
            return
        cmd = arg or self.session.get("test_cmd") or ""
        if not cmd:
            cmd = self.autodetect_test_cmd()
            if cmd:
                info("Test komutu otomatik bulundu: %s" % cmd)
                self.session["test_cmd"] = cmd
            else:
                err("Test komutu atanmadı. Örnek: /test set python3 main_test.py")
                return
        print(cyan("🔧 Çalıştırılıyor: %s" % bold(cmd)))
        try:
            p = subprocess.run(cmd, shell=True, cwd=self.root,
                               capture_output=True, text=True,
                               errors="replace", timeout=900)
            out = (p.stdout or "") + (p.stderr or "")
            rc = p.returncode
        except subprocess.TimeoutExpired:
            out, rc = "ZAMAN AŞIMI (900s)", 124
        shown = truncate_lines(out.strip(), 100, 100)
        print(dim(shown if shown else "(çıktı boş)"))
        status = green("exit=0 ✅") if rc == 0 else red("exit=%d ❌" % rc)
        nlines = len(out.splitlines())
        print("✅ Çıktı conversation'a eklendi (%d satır, %s)" % (nlines, status))
        self.session["messages"].append(self.msg(
            "event",
            "[TEST ÇIKTISI] $ %s (%s)\n```\n%s\n```" % (cmd, status, truncate_lines(out.strip())),
            kind="tool"))
        info("Şimdi örn: 'bu çıktıyı analiz et, sorunları bul'")

    # /run ----------------------------------------------------------------
    def cmd_run(self, arg):
        cmd = (arg or "").strip()
        if not cmd:
            err("Kullanım: /run <komut>")
            return
        if not ask_yn("🔧 Çalıştırılacak: %s — onaylıyor musun?" % bold(cmd)):
            print("İptal edildi.")
            return
        try:
            p = subprocess.run(cmd, shell=True, cwd=self.root,
                               capture_output=True, text=True,
                               errors="replace", timeout=300)
            out, rc = ((p.stdout or "") + (p.stderr or "")), p.returncode
        except subprocess.TimeoutExpired:
            out, rc = "ZAMAN AŞIMI (300s)", 124
        print(dim(truncate_lines(out.strip(), 80, 80) or "(çıktı boş)"))
        print((green("exit=0 ✅") if rc == 0 else red("exit=%d ❌" % rc)))
        self.session["messages"].append(self.msg(
            "event", "[KOMUT ÇIKTISI] $ %s (exit=%d)\n```\n%s\n```"
            % (cmd, rc, truncate_lines(out.strip())), kind="tool"))

    # /diag ---------------------------------------------------------------
    def cmd_diag(self, arg):
        last = next((m for m in reversed(self.session["messages"])
                     if m["role"] == "assistant"), None)
        code = last_python_block(last["content"]) if last else None
        if not code:
            err("Son AI yanıtında Python kod bloğu yok.")
            return
        print(cyan("🧪 Diagnostic kodu:"))
        print(dim(truncate_lines(code, 30, 30)))
        if not ask_yn("Çalıştırayım mı?"):
            print("İptal edildi.")
            return
        path = os.path.join(self.root, DIAG_FILE)
        atomic_write(path, code)
        try:
            p = subprocess.run([sys.executable, DIAG_FILE], cwd=self.root,
                               capture_output=True, text=True,
                               errors="replace", timeout=300)
            out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
        except subprocess.TimeoutExpired:
            out, rc = "ZAMAN AŞIMI (300s)", 124
        print(cyan("📊 Diagnostic çıktısı:"))
        print(out.strip() or "(boş)")
        self.session["messages"].append(self.msg(
            "event",
            "[DIAGNOSTIC ÇIKTISI] (%s)\n```\n%s\n```" % (
                "exit=0" if rc == 0 else "exit=%d" % rc,
                truncate_lines(out.strip())),
            kind="tool"))
        info("Çıktı conversation'a eklendi. Kod: %s (istemezsen sil)" % DIAG_FILE)

    # /read ---------------------------------------------------------------
    def cmd_read(self, arg):
        parts = (arg or "").strip().split(None, 1)
        if not parts:
            err("Kullanım: /read <dosya> [soru]")
            return
        rel = parts[0].replace("\\", "/").lstrip("./")
        full = os.path.join(self.root, rel)
        content = read_text_safe(full)
        if content is None:
            err("Dosya okunamadı: %s" % rel)
            return
        lines = content.splitlines()
        shown = "\n".join(lines[:FILE_READ_CAP])
        if len(lines) > FILE_READ_CAP:
            shown += "\n... [%d satır daha]" % (len(lines) - FILE_READ_CAP)
        self.session["messages"].append(self.msg(
            "event", "[DOSYA: %s — %d satır]\n```\n%s\n```" % (rel, len(lines), shown),
            kind="tool"))
        ok("%s conversation'a eklendi (%d satır%s)"
           % (rel, len(lines), ", kırpıldı" if len(lines) > FILE_READ_CAP else ""))
        if len(parts) > 1:
            self.chat_turn("%s dosyası eklendi. Soru: %s" % (rel, parts[1]))

    # /scan ---------------------------------------------------------------
    def cmd_scan(self, arg):
        self.scan_tree(show=True)
        self.save()
        ok("Proje ağacı güncellendi ve system prompt'a eklendi.")
        if (arg or "").strip():
            self.chat_turn("Proje ağacı güncellendi. Soru: " + arg.strip())

    # /focus --------------------------------------------------------------
    def cmd_focus(self, arg):
        arg = (arg or "").strip()
        if not arg:
            f = self.session.get("focus") or []
            print("Odak dosyaları:", bold(", ".join(f)) if f else "(yok)")
            return
        if arg == "off":
            self.session["focus"] = []
            ok("Odak modu kapatıldı.")
            return
        valid = []
        for a in arg.split():
            rel = a.replace("\\", "/").lstrip("./")
            if os.path.isfile(os.path.join(self.root, rel)):
                valid.append(rel)
            else:
                warn("Bulunamadı, atlandı: %s" % rel)
        if valid:
            self.session["focus"] = valid
            ok("Odak modu: %s → AI artık sadece bu dosyalara odaklanacak."
               % bold(", ".join(valid)))

    # /plan ---------------------------------------------------------------
    def cmd_plan(self, arg):
        goal = (arg or "").strip()
        if not goal:
            goal = ask("🎯 Hedef: ")
        if not goal:
            print("İptal edildi.")
            return
        md = self.read_plan()
        context_note = ""
        summaries = [m["content"] for m in self.session["messages"]
                     if m["kind"] == "summary"]
        if summaries:
            context_note = "Konuşma özeti (bağlam):\n" + summaries[-1][:1200] + "\n"
        prompt = ("[PLAN İSTEĞİ]\nHedef: %s\n\n%s\nŞu Markdown formatında plan üret "
                  "(başka hiçbir şey yazma):\n"
                  "# PLAN: <kısa başlık>\n\n## Hedef\n<1-2 cümle>\n\n## Adımlar\n"
                  "- [ ] <dosya/yolu> → <yapılacak iş>\n...\n\n## Notlar\n<risk/sıralama, "
                  "kısa>\n\nKurallar: en fazla 8 adım, her adım TEK satır ve ilgili "
                  "dosya yolunu içersin." % (goal, context_note))
        try:
            reply, _ = self.stream_reply([{"role": "system",
                                           "content": self.build_system_prompt()},
                                          {"role": "user", "content": prompt}],
                                         temperature=0.1)
        except OllamaError as e:
            err(str(e))
            return
        a = ask("\n📋 Bu plan kaydedilsin mi? [y]evet [n]hayır [e]elle düzenle: ").lower()
        if a == "e":
            reply = read_multiline()
        elif a not in ("y", "yes", "evet", ""):
            print("İptal edildi (plan kaydedilmedi).")
            return
        if "- [ ]" not in reply:
            warn("Plan formatında '- [ ]' adımı bulunamadı; yine de kaydediliyor.")
        atomic_write(self.plan_path(), reply.rstrip() + "\n")
        self.session["plan_file"] = os.path.basename(self.plan_path())
        self.session["messages"].append(self.msg(
            "event", "[DURUM] PLAN.md güncellendi. Hedef: %s" % goal, kind="event"))
        ok("%s kaydedildi (%s)" % (bold(self.session["plan_file"]),
                                   self.plan_progress()))
        info("Uygulamak için: /apply")

    # /apply --------------------------------------------------------------
    def cmd_apply(self, arg):
        if not self.session.get("pending"):
            md = self.read_plan()
            steps = plan_steps(md) if md else []
            todo = [t for d, t in steps if not d]
            if not todo:
                info("Bekleyen değişiklik yok. Önce AI'dan değişiklik iste "
                     "veya /plan ile plan üret.")
                return
            info("Bekleyen değişiklik yok — plandan üretiliyor (%d adım kaldı)..."
                 % len(todo))
            self.generate_from_plan(todo)
            if not self.session.get("pending"):
                return
        print(bold("\n🤖 Şu değişiklikler uygulanacak:"))
        for ch in self.session["pending"]:
            _, adds, dels, is_new = self.change_diff(ch)
            if is_new:
                print("  ➕ %s %s" % (ch["path"], dim("(yeni, %d satır)"
                                                     % len(ch["content"].splitlines()))))
            else:
                print("  📝 %s %s" % (ch["path"], dim("(+%d/-%d)" % (adds, dels))))
        choice = ask("\n[y]hepsini onayla [n]iptal [d]diff göster [s]tek tek: ").lower()
        if choice == "d":
            for ch in self.session["pending"]:
                self.print_colored_diff(ch)
                print()
            choice = ask("[y]hepsini onayla [n]iptal [s]tek tek: ").lower()
        if choice == "s":
            selected = []
            for ch in self.session["pending"]:
                if ask_yn("  %s uygulansın mı?" % bold(ch["path"])):
                    selected.append(ch["path"])
            if not selected:
                print("Hiçbiri seçilmedi (bekleyenler korundu).")
                return
            paths = selected
        elif choice in ("y", "yes", "evet", ""):
            paths = None
        else:
            print("İptal edildi (bekleyen değişiklikler korundu, /discard ile atabilirsin).")
            return
        applied = self.apply_changes(paths)
        ok("%d dosya güncellendi. Yedekler <dosya>.bak olarak saklandı → /undo"
           % len(applied))

    def generate_from_plan(self, todo):
        paths = []
        for t in todo:
            for tok in re.findall(r"[\w./-]+\.\w+", t):
                if tok not in paths and len(paths) < 4:
                    paths.append(tok)
        file_blocks = []
        total = 0
        for rel in paths + (self.session.get("focus") or []):
            content = read_text_safe(os.path.join(self.root, rel))
            if content is None:
                continue
            total += est_tokens(content)
            if total > 3500:
                break
            file_blocks.append("--- %s ---\n```\n%s\n```" % (rel, content))
        prompt = ("[PLAN UYGULAMA İSTEĞİ]\nAktif planın tamamlanmamış adımları:\n" +
                  "\n".join("- " + t for t in todo) +
                  "\n\nİlgili dosyaların güncel içeriği:\n" +
                  ("\n".join(file_blocks) if file_blocks else "(dosya içeriği verilmedi)") +
                  "\n\nGörev: Bu adımları uygula. Yanıtta AÇIKLAMAYI en fazla 3 cümlede "
                  "tut ve değişecek HER dosya için şu formatı kullan:\n"
                  "DOSYA: goreli/yol.py\n```python\n<dosyanın TAM yeni içeriği>\n```")
        try:
            reply, _ = self.stream_reply([{"role": "system",
                                           "content": self.build_system_prompt()},
                                          {"role": "user", "content": prompt}],
                                         temperature=0.15)
        except OllamaError as e:
            err(str(e))
            return
        self.session["messages"].append(self.msg("assistant", reply))
        changes, warns = parse_file_blocks(reply)
        for w in warns:
            warn(w)
        if changes:
            self.add_pending(changes)

    # /discard, /undo -----------------------------------------------------
    def cmd_discard(self, arg):
        n = len(self.session.get("pending", []))
        if not n:
            info("Bekleyen değişiklik yok.")
            return
        if ask_yn("%d bekleyen değişiklik atılsın mı?" % n):
            self.session["pending"] = []
            ok("Bekleyen değişiklikler atıldı.")

    def cmd_undo(self, arg):
        batches = self.session.get("backups") or []
        if not batches:
            info("Geri alınacak değişiklik yok.")
            return
        batch = batches.pop()
        for b in reversed(batch):
            full = os.path.join(self.root, b["path"])
            try:
                if b["was_new"]:
                    if os.path.exists(full):
                        os.remove(full)
                    print("  🗑 %s silindi (yeni dosyaydı)" % b["path"])
                else:
                    bak = read_text_safe(full + ".bak")
                    if bak is not None:
                        atomic_write(full, bak)
                        print("  ↩ %s geri yüklendi (.bak)" % b["path"])
                    else:
                        warn("%s için .bak bulunamadı!" % b["path"])
            except OSError as e:
                err("%s geri yüklenemedi: %s" % (b["path"], e))
        self.session["backups"] = batches
        ok("Son uygulama geri alındı. (PLAN.md işaretleri değiştirilmedi)")

    # /commit, /push ------------------------------------------------------
    def cmd_commit(self, arg):
        if not self.is_repo():
            err("Bu klasör bir git deposu değil. İlk once: git init")
            return
        rc, out, _ = self.git("status", "--porcelain", "-uall")
        if not out:
            info("Commitlenecek değişiklik yok (çalışma alanı temiz).")
            return
        changed = [l[3:].strip().replace('"', "") for l in out.splitlines()]
        changed = [p for p in changed
                   if not p.endswith(".bak") and p != DIAG_FILE
                   and "__pycache__/" not in p]
        applied = [p for p in self.session.get("last_applied", [])
                   if p in changed]
        stage = applied if applied else changed
        if applied:
            others = [p for p in changed if p not in applied]
            info("Asistanın değiştirdiği dosyalar stage'lenecek: %s"
                 % ", ".join(applied))
            if others:
                info("(stage dışı kalanlar: %s)" % ", ".join(others))
        else:
            info("Stage'lenecek dosyalar: %s" % ", ".join(changed))
        if not ask_yn("Bu dosyalar stage'lensin mi?"):
            print("İptal edildi.")
            return
        if stage:
            self.git("add", *stage)
        else:
            self.git("add", "-A")
        rc, stat, _ = self.git("diff", "--cached", "--stat")
        print(dim(stat or "(stat yok)"))
        prompt = ("[COMMIT MESAJI İSTEĞİ]\nGit değişiklikleri:\n%s\n\n"
                  "Tek satırlık conventional commit mesajı yaz "
                  "(feat/fix/chore/refactor/test/docs: ...). SADECE mesajı yaz, "
                  "başka hiçbir şey yazma." % (stat or ", ".join(stage)))
        try:
            msg, _ = self.ollama.chat(
                [{"role": "user", "content": prompt}], self.session["model"],
                min(4096, self.session["num_ctx"]), stream=False,
                num_predict=64, temperature=0.2)
        except OllamaError as e:
            warn("Model öneremedi: %s" % str(e).splitlines()[0])
            msg = ""
        msg = msg.strip().strip("`\"'").splitlines()[0].strip() if msg.strip() else ""
        suggestion = bold(msg) if msg else dim("(öneri alınamadı)")
        a = ask("\n🤖 Commit mesajı önerim:\n   %s\nOnaylıyor musun? [y/n/e-edit]: "
                % suggestion).lower()
        if a == "e":
            msg = ask("Mesaj: ").strip()
        elif a not in ("y", "yes", "evet", "") or not msg:
            print("İptal edildi.")
            return
        rc, cout, cerr = self.git("commit", "-m", msg)
        if rc != 0:
            err("Commit başarısız: %s" % (cerr or cout or "(bilinmeyen)"))
            return
        _, short, _ = self.git("rev-parse", "--short", "HEAD")
        ok("Commit atıldı: %s" % bold(short))
        self.session["last_applied"] = []
        self.session["messages"].append(self.msg(
            "event", "[DURUM] Git commit: %s \"%s\"" % (short, msg), kind="event"))

    def cmd_push(self, arg):
        if not self.is_repo():
            err("Bu klasör bir git deposu değil.")
            return
        rc, branch, _ = self.git("rev-parse", "--abbrev-ref", "HEAD")
        rc2, remote, _ = self.git("config", "--get", "branch.%s.remote" % branch)
        if rc2 == 0 and remote:
            cmd = ["push"]
            target = "%s/%s" % (remote, branch)
        else:
            rrc, remotes, _ = self.git("remote")
            if "origin" in remotes.split():
                cmd = ["push", "-u", "origin", branch]
                target = "origin/%s (yeni upstream)" % branch
            else:
                err("Remote bulunamadı. Önce: git remote add origin <url>")
                return
        if not ask_yn("🤖 %s üzerine push edilecek. Onaylıyor musun?" % bold(target)):
            print("İptal edildi.")
            return
        try:
            p = subprocess.run(["git"] + cmd, cwd=self.root,
                               capture_output=True, text=True,
                               errors="replace", timeout=300)
        except subprocess.TimeoutExpired:
            err("Push zaman aşımına uğradı (ağ?).")
            return
        out = (p.stdout or "") + (p.stderr or "")
        print(dim(out.strip() or "(çıktı yok)"))
        if p.returncode == 0:
            ok("Push tamamlandı → %s" % target)
        else:
            err("Push başarısız (yukarıdaki çıktıya bak).")

    # /status -------------------------------------------------------------
    def cmd_status(self, arg):
        s = self.session
        used, budget = self.context_usage()
        print(bold("\n📊 Durum"))
        print("  model      :", bold(s["model"]))
        print("  session    :", s["name"], dim("(%d mesaj, %s turn)"
                                             % (len(s["messages"]), s["stats"]["turns"])))
        print("  bağlam     : %s / %s tok %s" % (
            kfmt(used), kfmt(budget),
            yellow("(dolmak üzere)" if used > budget * 0.9 else "")))
        print("  proje      :", self.root)
        print("  plan       :", s.get("plan_file") or "-",
              ("(%s)" % self.plan_progress()) if self.plan_progress() else "")
        print("  focus      :", ", ".join(s.get("focus") or []) or "-")
        print("  test_cmd   :", s.get("test_cmd") or "-")
        print("  bekleyen   :", "%d değişiklik (/apply)" % len(s.get("pending", []))
              if s.get("pending") else "-")
        if self.is_repo():
            _, branch, _ = self.git("rev-parse", "--abbrev-ref", "HEAD")
            _, st, _ = self.git("status", "--porcelain")
            print("  git        : %s, %d değişiklik" % (branch, len(st.splitlines())))
        else:
            print("  git        : (depo değil)")

    # /model, /ctx --------------------------------------------------------
    def cmd_model(self, arg):
        arg = (arg or "").strip()
        if not arg:
            print("Aktif model:", bold(self.session["model"]))
            try:
                models = self.ollama.list_models()
                if models:
                    print("Kurulu modeller:")
                    for m in models:
                        mark = green(" ●") if m == self.session["model"] else "  "
                        print("  %s %s" % (mark, m))
            except OllamaError as e:
                err(str(e))
            return
        self.session["model"] = arg
        ok("Model değişti → %s" % bold(arg))

    def cmd_ctx(self, arg):
        try:
            n = int((arg or "").strip())
        except ValueError:
            err("Kullanım: /ctx 8192")
            return
        self.session["num_ctx"] = n
        ok("Context window → %d token" % n)

    # /compact ------------------------------------------------------------
    def cmd_compact(self, arg):
        n0 = len(self.session["messages"])
        self.auto_compact(force=True)
        ok("Sıkıştırma tamamlandı (%d → %d mesaj)"
           % (n0, len(self.session["messages"])))

    # session komutları ----------------------------------------------------
    def cmd_save(self, arg):
        name = (arg or "").strip() or self.session["name"]
        self.session["name"] = slugify(name)
        self.save()
        ok("Session kaydedildi: %s" % session_path(self.session["name"]))

    def cmd_load(self, arg):
        name = (arg or "").strip()
        if not name:
            self.cmd_sessions("")
            return
        s = load_session(name)
        if s is None:
            err("Session bulunamadı: %s (/list-sessions)" % name)
            return
        self.save()
        self.session = s
        self.banner()

    def cmd_new_session(self, arg):
        name = slugify((arg or "").strip() or ("s-" + datetime.now().strftime("%H%M%S")))
        self.save()
        old = self.session
        self.session = new_session(name, old["model"], old["num_ctx"], self.root)
        self.session["test_cmd"] = old.get("test_cmd", "")
        self.save()
        ok("Yeni session: %s" % bold(name))

    def cmd_sessions(self, arg):
        rows = list_sessions()
        if not rows:
            info("Kayıtlı session yok (~/.ai_sessions/).")
            return
        print(bold("\nSession'lar (~/.ai_sessions/):"))
        for name, updated, nmsg, model in rows:
            cur = green(" ●") if name == self.session["name"] else "  "
            print("  %s %-18s %s  %3d mesaj  %s" % (cur, name, updated, nmsg, model))

    # ======================================================================
    def dispatch(self, line):
        parts = line.split(None, 1)
        cmd = parts[0].lower().lstrip("/")
        arg = parts[1] if len(parts) > 1 else ""
        table = {
            "help": self.cmd_help, "?": self.cmd_help,
            "test": self.cmd_test, "run": self.cmd_run, "diag": self.cmd_diag,
            "read": self.cmd_read, "scan": self.cmd_scan, "focus": self.cmd_focus,
            "plan": self.cmd_plan, "apply": self.cmd_apply,
            "discard": self.cmd_discard, "undo": self.cmd_undo,
            "commit": self.cmd_commit, "push": self.cmd_push,
            "status": self.cmd_status,
            "model": self.cmd_model, "ctx": self.cmd_ctx,
            "compact": self.cmd_compact,
            "save": self.cmd_save, "load": self.cmd_load,
            "new-session": self.cmd_new_session, "new": self.cmd_new_session,
            "list-sessions": self.cmd_sessions, "sessions": self.cmd_sessions,
            "quit": None, "exit": None, "q": None,
        }
        if cmd in ("quit", "exit", "q"):
            return False
        fn = table.get(cmd)
        if fn is None:
            warn("Bilinmeyen komut: /%s — /help ile listeye bak." % cmd)
            return True
        try:
            fn(arg)
        except OllamaError as e:
            err(str(e))
        except KeyboardInterrupt:
            print(dim("\n  (kesildi)"))
        return True

    def repl(self):
        self.banner()
        while True:
            try:
                line = input(self.prompt()).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print(dim("  (çıkış için /quit)"))
                continue
            if not line:
                continue
            if line.startswith("/"):
                if not self.dispatch(line):
                    break
            else:
                try:
                    self.chat_turn(line)
                except OllamaError as e:
                    err(str(e))
                except KeyboardInterrupt:
                    print(dim("\n  (kesildi)"))
            self.save()
        self.save()
        print(cyan("\n👋 Görüşürüz! Session kaydedildi: %s" % self.session["name"]))


# ----------------------------------------------------------------------------
# PLAN.md yardımcıları
# ----------------------------------------------------------------------------

def plan_steps(md):
    """PLAN.md metninden (done?, metin) adım listesi çıkarır."""
    steps = []
    for line in (md or "").splitlines():
        m = re.match(r"^\s*-\s*\[( |x|X)\]\s*(.+)$", line)
        if m:
            steps.append((m.group(1).lower() == "x", m.group(2).strip()))
    return steps


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description="AI Workflow Assistant — Ollama tabanlı yerel kod asistanı")
    ap.add_argument("-m", "--model", default=cfg.get("model", DEFAULT_MODEL))
    ap.add_argument("-s", "--session", default=cfg.get("session", "default"))
    ap.add_argument("-d", "--dir", default=os.getcwd(),
                    help="proje klasörü (varsayılan: bulunulan dizin)")
    ap.add_argument("-c", "--ctx", type=int,
                    default=cfg.get("num_ctx", DEFAULT_CTX))
    ap.add_argument("-o", "--ollama", default=cfg.get("ollama_url", DEFAULT_OLLAMA))
    ap.add_argument("-q", "--once", metavar="SORU",
                    help="tek atımlık soru: sor, yanıtı bas, çık")
    ap.add_argument("--version", action="version",
                    version="ai.py %s" % VERSION)
    args = ap.parse_args()

    app = App(root=args.dir, model=args.model, num_ctx=args.ctx,
              ollama_url=args.ollama, session_name=slugify(args.session))

    if args.once:
        app.session["messages"].append(app.msg("user", args.once))
        app.auto_compact()
        try:
            reply, _ = app.stream_reply(app.build_api_messages())
            app.session["messages"].append(app.msg("assistant", reply))
        except OllamaError as e:
            err(str(e))
            sys.exit(1)
        app.save()
        return

    app.repl()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Çıkıldı.")
