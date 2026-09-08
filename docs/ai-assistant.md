# AI Workflow Assistant (`ai.py`)

Ollama tabanlı, **sıfır bağımlılıklı** (yalnızca Python stdlib) terminal yapay zekâ
asistanı. Termux / Android (8 GB RAM, 3B model) hedefiyle tasarlandı; her POSIX
ortamda çalışır.

```
test → analiz → soru-cevap → diagnostic → plan → onaylı uygulama → git → tekrar test
```

## Kurulum (Termux)

```bash
pkg install python git ollama          # ollama paketi yoksa: curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:3b
cp ai.py ~/bin/ai.py && chmod +x ~/bin/ai.py   # ~/bin yoksa: mkdir -p ~/bin
```

Çalıştırma:

```bash
cd ~/otomasyon                          # proje klasörün
python3 ~/bin/ai.py                     # veya ai.py
python3 ai.py -m qwen2.5-coder:7b -c 8192 -s oturum2
python3 ai.py -q "config.py ne yapıyor?"   # tek atımlık soru
```

Opsiyonel ayar dosyası `~/.ai_config.json`:

```json
{"model": "qwen2.5-coder:3b", "ollama_url": "http://localhost:11434", "num_ctx": 8192}
```

Modelsiz denemek: `python3 tools/mock_ollama.py 11435` ardından
`AI_OLLAMA_URL=http://127.0.0.1:11435 python3 ai.py`.

## Komutlar

| Komut | İş |
|---|---|
| `/test [cmd \| set <cmd>]` | Test komutunu çalıştırır, çıktısı (baş+kuyruk kırpılarak) conversation'a eklenir |
| `/run <komut>` | Tek seferlik komut (onaylı) |
| `/diag` | Modelin son diagnostic kodunu `.ai_diagnostic.py`'ye yazıp çalıştırır, çıktıyı ekler |
| `/read <dosya> [soru]` | Dosyayı conversation'a ekler (250 satıra kadar) |
| `/scan [soru]` | Proje ağacını tarayıp system prompt'a gömer |
| `/focus <d1> <d2> \| off` | Modeli seçili dosyalara kilitler (içerikler otomatik eklenir) |
| `/plan [hedef]` | Checkbox'lı PLAN.md üretir (onaylı) |
| `/apply` | Bekleyen değişiklikleri listeler → [y/n/d/s] → toplu uygular, `.bak` alır |
| `/discard` | Bekleyen değişiklikleri atar |
| `/undo` | Son `/apply`'yi `.bak`'tan geri alır |
| `/commit` | AI commit mesajı önerir → [y/n/e] → commit |
| `/push` | Onaylı `git push` (upstream yoksa `-u` önerir) |
| `/status` | Model, bağlam kullanımı, plan, focus, git durumu |
| `/model [ad]`, `/ctx <n>` | Model / context window değiştir |
| `/compact` | Konuşmayı zorla özetler |
| `/save`, `/load`, `/new-session`, `/list-sessions` | Session yönetimi (`~/.ai_sessions/`) |
| `/quit` | Çıkış (otomatik kaydeder) |

Serbest metin doğrudan modele gider. Model `DOSYA:` bloklarıyla değişiklik
önerirse bunlar **pending** olarak tutulur; hiçbir dosya `/apply` onayı olmadan
değişmez.

## Session dosyası şeması (`~/.ai_sessions/<ad>.json`)

```json
{
  "version": 2,
  "name": "demo1",
  "created": "2026-09-08T10:24:36",
  "updated": "2026-09-08T10:24:38",
  "model": "qwen2.5-coder:3b",
  "num_ctx": 8192,
  "project_root": "/tmp/ai-demo",
  "test_cmd": "python3 main_test.py",
  "focus": [],
  "plan_file": "PLAN.md",
  "tree": "auth/login.py\nmain_test.py\n...",
  "messages": [
    {"role": "user|assistant|summary|event", "content": "...",
     "kind": "chat|tool|summary|event", "ts": "2026-09-08T10:24:37"}
  ],
  "pending":   [{"path": "auth/login.py", "content": "<tam yeni içerik>"}],
  "last_applied": ["auth/login.py"],
  "backups":   [[{"path": "auth/login.py", "was_new": false}]],
  "stats": {"turns": 2}
}
```

## PLAN.md formatı

```markdown
# PLAN: Login akışına timeout ve retry ekle

## Hedef
Login fonksiyonuna 30 saniyelik timeout ve 3 denemelik retry eklenecek.

## Adımlar
- [ ] auth/login.py → timeout=30 parametresi ekle
- [ ] utils/retry.py → retry decorator'ı içeren yeni dosya oluştur
- [ ] tests/test_login.py → yeni testler

## Notlar
- Önce utils/retry.py yazılmalı.
```

`/apply` ile bir dosya yazıldığında o dosyanın yolunu içeren adımlar otomatik
`- [x]` olur.

## Bağlam (8k token) yönetimi

1. Test/komut çıktıları **eklenirken** kırpılır (ilk 50 + son 70 satır).
2. System prompt (rol + ağaç + focus + sıradaki plan adımları) ~1.5k token ile sınırlı.
3. Bütçe aşılırsa eski mesajlar tek bir Türkçe özete indirgenir (kararlar, dosya
   yolları, sayısal değerler korunur); son 6 mesaj her zaman birebir tutulur.
4. Hâlâ taşarsa araç çıktıları ve uzun mesajlar sert kırpılır (`_hard_trim`).
5. `/status` anlık kullanımı, `/compact` zorlu özetleme verir.

## 3B model için prompt stratejisi

- Değişiklik formatı bilinçli olarak **tüm-dosya-yeniden-yazma** (`DOSYA:` +
  tek fenced blok): 3B modeller SEARCH/REPLACE diff'lerinde çok hata yapar
  (aider de zayıf modellerde `whole` formatına düşer).
- System prompt ~15 satır; kurallar numaralı, format örneği literal.
- Alt görevler (commit mesajı, özet, plan) ayrı küçük istekler olarak
  gönderilir; `num_predict` ile sınırlanır.
- `temperature` 0.1–0.2 (kod için deterministik çıktı).
- Modeli uydurma yapmaması için dosya içeriği olmadan iddia yasak; `/read`
  ile içerik istemesi yönlendirilir.
