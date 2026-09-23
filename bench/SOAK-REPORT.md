# Codna vs Cursor vs Codex — 97-repo go-live soak

_Generated from 97 public-repo runs · Codna engine-behind on gpt-5.4 · one repo at a time, full 3-way fix._

## Headline

- **Codna success: 95/96** real runs `ok` (clone-failed/bad-URL excluded).
- **Codna median fix: 9s** (range 5–58s) vs Cursor (median ~210s) / Codex (median ~142s) — **~18–26× faster**.
- Codna context fed to the agent: ~5–6k tokens (25–190× repo reduction); Codex consumed 0.8M–11M tokens/repo.
- 2 flags, both benign: `schedule` = bad repo URL (clone failed, not Codna); `websockets` = a transient embedding-endpoint blip (not reproducible) — hardened by recall-retry (#50) + graceful degradation.

## Per-repo results

| Repo | Codna | Cursor | Codex | Codna status |
|---|--:|--:|--:|---|
| anyio | 7s | 346s | 172s | ok |
| arrow | 18s | 170s | 143s | ok |
| asgiref | 9s | 230s | 150s | ok |
| attrs | 14s | 241s | 187s | ok |
| bleach | 7s | 153s | 166s | ok |
| blinker | 10s | 122s | 101s | ok |
| boltons | 7s | 161s | 109s | ok |
| cachetools | 9s | 187s | 163s | ok |
| cattrs | 9s | 105s | 85s | ok |
| click | 13s | 81s | 82s | ok |
| colorama | 7s | 123s | 68s | ok |
| coveragepy | 13s | 121s | 68s | ok |
| croniter | 14s | 268s | 152s | ok |
| deepdiff | 11s | 314s | 228s | ok |
| dirty-equals | 5s | 241s | 143s | ok |
| emoji | 7s | 290s | 161s | ok |
| environs | 7s | 183s | 110s | ok |
| faker | 19s | 130s | 94s | ok |
| feedparser | 10s | 110s | 87s | ok |
| flask | 12s | 900s | 150s | ok |
| freezegun | 7s | 70s | 64s | ok |
| furl | 6s | 360s | 119s | ok |
| gunicorn | 11s | 248s | 207s | ok |
| h11 | 8s | 120s | 92s | ok |
| httpx | 13s | 114s | 284s | ok |
| hug | 7s | 523s | 122s | ok |
| humanize | 7s | 224s | 162s | ok |
| idna | 7s | 454s | 177s | ok |
| inflection | 7s | 92s | 40s | ok |
| itsdangerous | 6s | 148s | 102s | ok |
| jinja | 12s | 175s | 156s | ok |
| jmespath.py | 7s | 205s | 135s | ok |
| jsonpickle | 6s | 220s | 141s | ok |
| jsonschema | 13s | 147s | 98s | ok |
| loguru | 9s | 630s | 293s | ok |
| markdown | 10s | 396s | 161s | ok |
| marshmallow | 12s | 508s | 144s | ok |
| mistune | 7s | 296s | 185s | ok |
| more-itertools | 8s | 263s | 107s | ok |
| multidict | 9s | 293s | 212s | ok |
| oauthlib | 10s | 189s | 223s | ok |
| packaging | 58s | 542s | 425s | ok |
| paramiko | 10s | 215s | 112s | ok |
| parse | 8s | 474s | 127s | ok |
| pendulum | 9s | 818s | 196s | ok |
| pexpect | 7s | 900s | 178s | ok |
| platformdirs | 8s | 126s | 133s | ok |
| pluggy | 9s | 176s | 221s | ok |
| ply | 7s | 166s | 76s | ok |
| prettytable | 6s | 202s | 202s | ok |
| pygments | 23s | 212s | 136s | ok |
| pyjwt | 20s | 82s | 76s | ok |
| pyparsing | 7s | 141s | 118s | ok |
| pyrsistent | 21s | 177s | 99s | ok |
| pytest | 13s | 370s | 161s | ok |
| python-pathspec | 10s | 192s | 155s | ok |
| python-prompt-toolkit | 10s | 205s | 443s | ok |
| python-semver | 13s | 266s | 163s | ok |
| python-sortedcontainers | 10s | 525s | 414s | ok |
| python-tabulate | 19s | 445s | 207s | ok |
| pytz | 11s | 126s | 117s | ok |
| pyyaml | 6s | 131s | 130s | ok |
| requests-oauthlib | 8s | 104s | 190s | ok |
| requests | 7s | 175s | 168s | ok |
| rich | 9s | 503s | 346s | ok |
| schedule | — | — | — | clone failed (bad URL) |
| schema | 6s | 106s | 155s | ok |
| shellingham | 7s | 210s | 129s | ok |
| simplejson | 6s | 233s | 252s | ok |
| six | 5s | 166s | 147s | ok |
| sniffio | 7s | 197s | 100s | ok |
| soupsieve | 6s | 550s | 76s | ok |
| sqlparse | 14s | 248s | 195s | ok |
| starlette | 7s | 218s | 129s | ok |
| tenacity | 14s | 100s | 99s | ok |
| texttable | 5s | 191s | 79s | ok |
| tinydb | 6s | 283s | 175s | ok |
| toml | 6s | 116s | 102s | ok |
| tomlkit | 9s | 289s | 102s | ok |
| toolz | 6s | 120s | 111s | ok |
| tqdm | 7s | 95s | 91s | ok |
| traitlets | 15s | 358s | 235s | ok |
| trio | 18s | 258s | 121s | ok |
| typer | 19s | 776s | 401s | ok |
| uritemplate | 6s | 244s | 87s | ok |
| urllib3 | 12s | 140s | 107s | ok |
| uvicorn | 7s | 412s | 142s | ok |
| validators | 9s | 355s | 98s | ok |
| voluptuous | 6s | 201s | 87s | ok |
| watchdog | 6s | 900s | 74s | ok |
| wcwidth | 9s | 87s | 91s | ok |
| websockets | 50s | 722s | 209s | exit 1 |
| werkzeug | 11s | 193s | 208s | ok |
| wrapt | 12s | 354s | 386s | ok |
| xmltodict | 6s | 83s | 82s | ok |
| yarl | 6s | 249s | 161s | ok |
| zipp | 6s | 328s | 215s | ok |
