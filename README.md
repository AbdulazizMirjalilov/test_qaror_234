# Qaror 234 RAG

RAG tizimi: Oʻzbekiston Respublikasi Vazirlar Mahkamasining 234-son qarori boʻyicha savollarga javob berish uchun.

Manba hujjat: [Vazirlar Mahkamasining 2026-yil 11-maydagi 234-son qarori](https://lex.uz/uz/docs/-8193120) (lex.uz).

## Gallyusinatsiyani jilovlash yondashuvi

1. **Score threshold** — retrieval bosqichida eng yaxshi natija belgilangan chegaradan (default 0.55) past boʻlsa, LLM chaqirilmaydi, darhol "Hujjatda bu haqida ma'lumot yo'q" qaytariladi (kirillcha savolga — kirillcha javob).
2. **Qatʼiy system prompt** — LLM faqat berilgan kontekstdan foydalanishga majburlanadi; kontekst mavjud boʻlsa-da, unda aniq javob boʻlmasa, LLM buni tan olishi va rad etishi kerak.
3. **Manba dasturiy yoʻl bilan aniqlanadi** — LLM manba (ilova/bob/band) koʻrsatishga ishonilmaydi (xato qilishi mumkin edi), buning oʻrniga top-1 retrieval natijasining metadatasi dasturiy ravishda javobga qoʻshiladi.
4. **`answer_grounded` maydoni** — retrieval nimadir topgan (`found_in_document: true`) boʻlsa-da, LLM haqiqatan ham mazmunli javob berdimi yoki "ma'lumot yo'q" deb rad etdimi — bu ikki holat API javobida alohida maydonlar orqali aniq ajratiladi.
5. **Toʻliq gap nazorati** — model faqat raqam ("25") yoki savolni takrorlamaydigan parcha ("Yigirma besh ish kuni ichida") qaytarsa, bitta tuzatuvchi soʻrov yuboriladi va javob toʻliq gapga aylantiriladi. Tekshiruv javob uzunligiga emas, savoldagi asosiy soʻzlar javobda takrorlanganiga qaraydi — parcha javob uzun boʻlishi ham mumkin.

Qoʻshimcha: foydalanuvchi savoli indeksdagi matn bilan **bir xil normalizatsiyadan** oʻtkaziladi (apostrof variantlari, kirill → lotin transliteratsiyasi), shuning uchun `o'tkazish`, `oʻtkazish` va `ўтказиш` yozilishlari bir xil natija beradi.

## Savol turlari

Tizim uch xil savolni ajratadi:

1. **Hujjat mazmuni boʻyicha savollar** — retrieval + LLM orqali, manba (ilova/bob/band) koʻrsatilgan holda javob beriladi.
2. **Qarorning oʻzi haqidagi savollar** ("Qaror raqami nechchi?", "Qarorni kim imzolagan?") — hujjatda bu maʼlumotlar faqat imzo blokida (`Toshkent sh.` / `2026-yil 11-may,` / `234-son`) uchraydi va tabiiy savolga semantik jihatdan deyarli oʻxshamaydi. Shu sababli indeksga hujjat rekvizitlari uchun alohida chunk qoʻshiladi (raqam, sana, nom, imzolagan shaxs, kuchga kirishi, ilovalar soni). U javobda `"source": "Qaror rekvizitlari"` deb koʻrsatiladi va `chunker.py` da hujjat matnidan avtomatik yigʻiladi — qoʻlda yozilmaydi, hujjat almashtirilsa oʻzi yangilanadi.
3. **Yordamchining oʻzi haqidagi savollar** ("Nima qila olasan?", "Salom") — bunday maʼlumot hujjatda yoʻq, shuning uchun ular retrievalgacha ushlanadi: LLM chaqirilmasdan tizim imkoniyatlari haqida qisqa javob qaytariladi (`found_in_document: false`).

## Lokal muhitda ishga tushirish

### 1. Talablar
- Python 3.10 – 3.14 (tavsiya: 3.14)
- [Poetry](https://python-poetry.org) (dependency manager)
- [Ollama](https://ollama.com) oʻrnatilgan va ishga tushirilgan
- (ixtiyoriy) `make` — qulay buyruqlar uchun

### 2. Ollama modelini yuklab olish
```bash
ollama pull qwen2.5:7b-instruct
```
Ollama fon rejimida ishlab turishi kerak (tekshirish uchun: `ollama list`).

### 3. Kutubxonalarni oʻrnatish
```bash
make install        # yoki: poetry install
```
Virtual muhit avtomatik ravishda loyiha ichida (`.venv/`) yaratiladi (`poetry.toml`).

Faqat testlar uchun yengil muhit (torch/chromadb'siz):
```bash
make install-light  # yoki: poetry install --without ml
```

**Poetry oʻrnatilmagan boʻlsa (pip fallback):** loyihaning asosiy dependency
manager'i — Poetry (`pyproject.toml` + `poetry.lock`). Qulaylik uchun
`requirements.txt` ham repoda bor, lekin u **qoʻlda tahrirlanmaydi** —
`poetry.lock`'dan avtomatik eksport qilinadi (`make export-reqs`). Pip bilan
oʻrnatish:
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Dependency'lar oʻzgarganda (`pyproject.toml` tahrirlangach) ikkala faylni
yangilang: `make lock && make export-reqs`.

### 4. Sozlamalar (ixtiyoriy)
Barcha sozlamalar `app/core/config.py` da jamlangan va `QAROR_` prefiksli environment oʻzgaruvchilari yoki `.env` fayli orqali oʻzgartiriladi:
```bash
cp .env.example .env
```
Asosiy oʻzgaruvchilar: `QAROR_SCORE_THRESHOLD`, `QAROR_OLLAMA_MODEL`, `QAROR_EMBEDDING_MODEL`, `QAROR_TOP_K` va boshqalar — koʻp ishlatiladiganlari `.env.example` faylida, toʻliq roʻyxat esa `app/core/config.py` da.

**Model tanlash:** default `qwen2.5:7b-instruct` (~8 GB RAM yetarli). Kattaroq model
(masalan `qwen2.5:14b-instruct`) `QAROR_OLLAMA_MODEL` orqali almashtiriladi, lekin unga
kamida 24 GB RAM tavsiya etiladi — 16 GB mashinada sinovda u sekin (~40-50 s/javob),
beqaror (paging tufayli timeout'lar) va sifat jihatidan barqaror ustunlik bermadi.

**Model xotirada qancha turadi:** har bir soʻrov bilan Ollama'ga `keep_alive`
yuboriladi (default `30m`), yaʼni model soʻnggi savoldan keyin 30 daqiqa RAM'da
qoladi. Ollama'ning oʻz defaulti 5 daqiqa — undan keyin model diskdan qayta
yuklanadi va bu CPU'da keyingi savolga ~20–100 s (kesh holatiga qarab) qoʻshimcha kutish qoʻshadi, yaʼni
koʻpincha javob vaqtiga teng yoki undan uzunroq. Buning evaziga model RAM'ni uzoqroq band qiladi
(model ~4.7 GB, shu davr mobaynida ushlab turiladi). Xotira tor boʻlsa
`QAROR_OLLAMA_KEEP_ALIVE` ni qisqartiring (`5m`, yoki `0` — darhol boʻshatish);
boʻsh qoldirilsa `keep_alive` umuman yuborilmaydi va Ollama serverining oʻz
sozlamasi kuchda qoladi — Ollama boshqa dasturlar bilan birga ishlatilsa shu maʼqul.

### 5. Manba hujjatni joylashtirish
Qaror matni ([lex.uz/uz/docs/-8193120](https://lex.uz/uz/docs/-8193120)) allaqachon
`data/234_11.05.2026_ozb.doc` sifatida repoda mavjud — bu qadam faqat hujjat
yangilanganda kerak boʻladi.

### 6. Hujjatni indekslash (bir marta bajariladi)
```bash
make ingest
# yoki qoʻlda:
poetry run python -m app.ingestion.loader
poetry run python -m app.ingestion.chunker
poetry run python -m app.ingestion.embedder
```
`embedder.py` birinchi ishga tushirilganda BGE-M3 modelini (~2.3 GB) yuklab oladi — internet tezligiga qarab vaqt talab qilishi mumkin.

### 7. API'ni ishga tushirish
```bash
make dev            # yoki: poetry run uvicorn app.main:app --reload
```

### 8. Sinash
```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Aeroportlar uchun ekspertiza oʻtkazish muddati qancha?"}'
```

Yoki Swagger UI: `http://localhost:8000/docs`

## Arxitektura

Loyiha qatlamli (layered) arxitekturada tashkil etilgan:

```
app/
├── api/v1/          # HTTP endpointlar (yupqa qatlam, faqat routing)
├── core/            # config, constants, dependencies (DI), exceptions, logging
├── services/        # biznes mantiq (rag.py) va tashqi servis klientlari
│                    # (retriever.py — Chroma+embedding, llm.py — Ollama)
├── schemas/         # Pydantic request/response modellari
├── utils/           # matn normalizatsiyasi va translit, manba formati,
│                    # javob shakli (rad javobi / parcha gap aniqlash)
└── ingestion/       # offline indekslash pipeline (loader → chunker → embedder)
```

Xatolar yagona formatda qaytariladi (`{"success": false, "message": ..., "data": ...}`),
har bir soʻrov loguru orqali loglanadi (`logs/` papkasiga ham yoziladi).

## API endpointlari

| Endpoint | Tavsif |
|---|---|
| `POST /v1/ask` | Savolga toʻliq javob (manba va score bilan) |
| `POST /v1/ask/stream` | Xuddi shu, lekin javob token-token NDJSON oqimida (`meta` → `token`... → `done`) |
| `GET /health` | Readiness tekshiruvi: Chroma indeksi va Ollama holati alohida koʻrsatiladi (nosozlikda 503) |
| `GET /` | Servis nomi va versiyasi |

Ollama ishlamayotgan boʻlsa `/v1/ask` tushunarli xabar bilan `503` qaytaradi (ichki 500 emas).

### Javob formati

```json
{
  "answer": "Aeroportlar uchun ekspertiza oʻtkazish muddati 25 ish kuni boʻladi.",
  "source": "1-ilova",
  "score": 0.639,
  "found_in_document": true,
  "answer_grounded": true
}
```

- `found_in_document` — retrieval bosqichida hujjatdan tegishli boʻlak topildimi (score threshold asosida). Diqqat: yordamchining oʻzi haqidagi savollarda ham `false` boʻladi — javob foydali, lekin u hujjatdan olinmagan.
- `answer_grounded` — LLM haqiqatan ham mazmunli javob berdimi, yoki kontekst topilgan boʻlsa ham "ma'lumot yo'q" deb javob berdimi (kirillcha rad javobi ham aniqlanadi).
- `source` — odatda `1-ilova` yoki `2-ilova, 3-bob, 28-band` koʻrinishida. Qarorning oʻzi (raqami, sanasi, imzolagan shaxs) haqidagi savollarda `Qaror rekvizitlari`, javob hujjatdan olinmaganda esa `null` boʻladi.

## Tillar va alifbolar

Hujjat lotin alifbosida indekslangan (`data/234_11.05.2026_ozb.doc`), lekin savollarni **lotin** yoki **oʻzbek kirill** alifbosida berish mumkin.

Kirill yozuvidagi savol kelsa, retrievaldan oldin u avtomatik lotinga transliteratsiya qilinadi (`app/utils/text.py`). LLM esa foydalanuvchi savolini asl koʻrinishida oladi va javobni **xuddi shu alifboda** qaytaradi — lotin savolga lotin, kirill savolga kirill.

Kirill alifbosi bilan sinash:

```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Аэропортлар учун экспертиза муддати қанча?"}'
```

## Docker orqali ishga tushirish

> **Muhim:** Chroma indeksi git'da saqlanmaydi (`make ingest` natijasi). Docker image
> `data/` katalogini oʻzida koʻchiradi, shuning uchun `docker build`dan **oldin** hostda
> `make ingest` ni bajaring — aks holda konteyner ishga tushishda `IndexNotReadyError`
> bilan toʻxtaydi.

**1-rejim — toʻliq mustaqil stack** (hech narsa oʻrnatilmagan mashina uchun; Ollama ham konteynerda):
```bash
docker compose up --build
# birinchi martada LLM modelini konteynerga yuklab olish:
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```
Eslatma: konteynerlar hostdagi Ollama va model keshlarini koʻrmaydi — bu rejimda Ollama image (~2 GB), Qwen modeli (~4.7 GB) va BGE-M3 (~2.3 GB) konteyner volume'lariga qaytadan yuklab olinadi.

**2-rejim — hostdagi Ollama'dan foydalanish** (Ollama va modellar allaqachon lokal oʻrnatilgan boʻlsa — qayta yuklab olinmaydi):
```bash
make docker-up-host
# yoki:
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build
```
Bu rejimda faqat API konteynerda ishlaydi, u hostdagi Ollama (`host.docker.internal:11434`) va hostdagi HuggingFace keshidan foydalanadi.

## Development

Kundalik buyruqlar `Makefile` da jamlangan:

| Buyruq | Tavsif |
|---|---|
| `make install` / `make install-light` | Barcha / yengil (ML'siz) kutubxonalar |
| `make dev` | API'ni auto-reload bilan ishga tushirish |
| `make test` | Test toʻplami |
| `make lint` / `make format` | Ruff tekshiruv / avtoformat |
| `make precommit` | Git pre-commit hook'larini oʻrnatish |
| `make ingest` / `make eval` | Indekslash / retrieval sifatini baholash |
| `make lock` | `poetry.lock` ni qayta yozish |
| `make export-reqs` | `requirements.txt` ni `poetry.lock`'dan qayta eksport qilish |

**Pre-commit** (`.pre-commit-config.yaml`): har bir commit oldidan ruff (lint+format) va
umumiy tekshiruvlar (trailing whitespace, YAML/TOML/JSON, katta fayllar) avtomatik ishlaydi.
Bir marta oʻrnatiladi:
```bash
make precommit      # yoki: poetry run pre-commit install
```

## Testlar

Test toʻplami ogʻir ML kutubxonalarsiz ishlaydi (retriever test ichida almashtiriladi yoki mock qilinadi, Ollama talab qilinmaydi):
```bash
make test           # yoki: poetry run pytest -q
```
CI (GitHub Actions) har bir push/PR da lint va testlarni ishga tushiradi.

## Retrieval sifatini baholash

`data/eval_questions.json` da kutilgan manbalari koʻrsatilgan savollar toʻplami bor. Toʻliq muhit oʻrnatilgandan soʻng:
```bash
make eval           # yoki: poetry run python -m scripts.evaluate_retrieval
```
Skript threshold qarorlari aniqligini, hit@1 / hit@k koʻrsatkichlarini va score taqsimotini chiqaradi — `QAROR_SCORE_THRESHOLD` ni qayta sozlash uchun ayni kerakli maʼlumot.

## Maʼlum cheklovlar

- Hujjat matni faqat lotin alifbosida indekslangan; kirill versiyasi (`data/234_11.05.2026_уз.doc`) alohida indekslanmaydi. Kirill savollar qoidaviy transliteratsiya orqali qidiriladi — noaniq yoki kam uchraydigan soʻzlar uchun retrieval sifati biroz pasayishi mumkin.
- Similarity score chegarasi (0.55) kichik eval toʻplam asosida tanlangan; kengroq toʻplam bilan `scripts/evaluate_retrieval.py` orqali qayta baholash tavsiya etiladi.
#   t e s t _ q a r o r _ 2 3 4  
 #   t e s t _ q a r o r _ 2 3 4  
 