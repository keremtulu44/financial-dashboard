#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mock_ollama.py — ai.py'yi modelsiz test etmek için sahte Ollama sunucusu.

Gerçek Ollama API'sini taklit eder: POST /api/chat (stream + non-stream),
POST /api/generate, GET /api/tags. Yanıtlar anahtar kelimelere göre seçilir.
ai.py'nin HTTP/NDJSON katmanını ve tüm onay akışını gerçek bir model
çalıştırmadan uçtan uca denemek içindir.

Kullanım:
    python3 tools/mock_ollama.py [port]        # varsayılan 11435
    AI_OLLAMA_URL=http://127.0.0.1:11435 python3 ai.py
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11435

PLAN_MD = """# PLAN: Login akışına timeout ve retry ekle

## Hedef
Login fonksiyonuna 30 saniyelik timeout ve 3 denemelik retry mekanizması eklenecek.

## Adımlar
- [ ] auth/login.py → timeout=30 parametresi ekle
- [ ] utils/retry.py → retry decorator'ı içeren yeni dosya oluştur
- [ ] tests/test_login.py → timeout ve retry için yeni testler
- [ ] main_test.py → test listesine yeni testleri ekle

## Notlar
- Önce utils/retry.py yazılmalı, login onu kullanacak.
- Geriye dönük uyumluluk: timeout parametresi opsiyonel olmalı.
"""

LOGIN_NEW = '''"""auth/login.py — oturum açma yardımcıları."""
import time

from utils.retry import retry


@retry(times=3, delay=1.0)
def login(username, password, timeout=30):
    """Kullanıcı girişi. timeout saniye içinde yanıt gelmezse hata verir."""
    start = time.time()
    resp = _do_request(username, password)
    elapsed = time.time() - start
    if elapsed > timeout:
        raise TimeoutError("login %ss icinde yanit vermedi" % timeout)
    if resp.get("status") != "ok":
        raise ValueError("giriss basarisiz: %s" % resp.get("reason"))
    return resp["token"]


def _do_request(username, password):
    # mevcut istek mantigi (ayni kaliyor)
    return {"status": "ok", "token": "demo-token"}
'''

RETRY_PY = '''"""utils/retry.py — basit retry decorator'ı."""
import functools
import time


def retry(times=3, delay=1.0, exceptions=(Exception,)):
    """Fonksiyonu hata halinde `times` kez, `delay` saniye arayla tekrar dener."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last = exc
                    time.sleep(delay)
            raise last
        return wrapper
    return decorator
'''

TEST_LOGIN = '''"""tests/test_login.py — timeout ve retry testleri."""
import unittest


class TestLogin(unittest.TestCase):
    def test_login_success(self):
        from auth.login import login
        self.assertEqual(login("u", "p"), "demo-token")

    def test_login_has_timeout_default(self):
        import inspect
        from auth import login as m
        sig = inspect.signature(m.login)
        self.assertEqual(sig.parameters["timeout"].default, 30)


if __name__ == "__main__":
    unittest.main()
'''

DIAG_CODE = '''DIAGNOSTIC: Sistemdeki timeout/retry durumunu kontrol eden küçük script:

```python
import inspect
import auth.login as m

sig = inspect.signature(m.login)
print("timeout parametresi:", sig.parameters.get("timeout", "YOK"))
src = inspect.getsource(m.login)
print("retry kullanimi:", "VAR" if "retry" in src else "YOK")
```

Sonuca göre planı konuşalım.
'''

ANALYSIS = """Test çıktısını inceledim. 3 potansiyel sorun buldum:

1. **Timeout eksik** — `auth/login.py` içindeki `login()` hiçbir timeout değeri kullanmıyor; ağ takılması durumunda sonsuza kadar bekleyebilir.
2. **Retry yok** — geçici ağ hatalarında tek deneme ile vazgeçiliyor.
3. **Loglama yetersiz** — başarısız denemeler hiçbir yere kaydedilmiyor.

Sana sorularım:
1. Sistem şu an production'da mı çalışıyor?
2. Timeout kaç saniye olmalı?
3. Retry'da kaç deneme uygundur?
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._send_json({"models": [
                {"name": "qwen2.5-coder:3b"},
                {"name": "qwen2.5-coder:7b"},
            ]})
        else:
            self._send_json({"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path.startswith("/api/generate"):
            self._handle(req, gen=True)
        else:
            self._handle(req, gen=False)

    def _pick_reply(self, req):
        text = ""
        msgs = req.get("messages") or []
        if msgs:
            text = (msgs[-1].get("content") or "") + "\n" + \
                   "\n".join((m.get("content") or "") for m in msgs[-3:])
        else:
            text = req.get("prompt", "")
        if "[PLAN İSTEĞİ]" in text:
            return PLAN_MD
        if "[PLAN UYGULAMA İSTEĞİ]" in text:
            return ("Plandaki ilk üç adımı uyguluyorum.\n\n"
                    "DOSYA: utils/retry.py\n```python\n" + RETRY_PY + "```\n\n"
                    "DOSYA: auth/login.py\n```python\n" + LOGIN_NEW + "```\n\n"
                    "DOSYA: tests/test_login.py\n```python\n" + TEST_LOGIN + "```\n")
        if "[COMMIT MESAJI İSTEĞİ]" in text:
            return "feat(auth): add timeout and retry mechanism to login"
        if "[ÖZET İSTEĞİ]" in text:
            return ("Kullanıcı login akışına 30s timeout ve retry eklemeye karar verdi. "
                    "İlgili dosyalar: auth/login.py, utils/retry.py (yeni), "
                    "tests/test_login.py (yeni). Plan PLAN.md'de.")
        if "diagnostic" in text.lower() and ("yaz" in text or "hazırla" in text):
            return DIAG_CODE
        if "[DIAGNOSTIC" in text or "diagnostic yaz" in text.lower():
            return DIAG_CODE
        if "timeout 30" in text or ("evet" in text.lower() and "retry" in text.lower()):
            return ("Anladım: production, timeout=30s, retry eklenecek. "
                    "Teşhisi netleştirmek için bir diagnostic yazayım mı? "
                    "(Hazırlandı — /diag ile çalıştırabilirsin)\n\n" + DIAG_CODE)
        if "analiz" in text.lower() or "[TEST ÇIKTISI]" in text:
            return ANALYSIS
        return "Anladım: " + text.strip().splitlines()[-1][:120] if text.strip() else "Tamam."

    def _handle(self, req, gen=False):
        reply = self._pick_reply(req)
        stream = bool(req.get("stream", True))
        if not stream:
            if gen:
                self._send_json({"response": reply, "done": True})
            else:
                self._send_json({"message": {"role": "assistant",
                                             "content": reply}, "done": True})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        step = 24
        for i in range(0, len(reply), step):
            chunk = reply[i:i + step]
            obj = {"message": {"role": "assistant", "content": chunk}, "done": False}
            self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.01)
        final = {"message": {"role": "assistant", "content": ""}, "done": True}
        self.wfile.write((json.dumps(final) + "\n").encode("utf-8"))
        self.wfile.flush()


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("mock ollama → http://127.0.0.1:%d" % PORT)
    srv.serve_forever()
