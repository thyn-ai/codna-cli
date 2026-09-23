# Codna memory: OUR bigram embedding vs LLM embedding

_Codna-only (gpt-5.4 fix, engine-behind) across the same public repos + issues as the go-live soak. The ONLY variable is the recall **embedder**: in-process Telys **bigram** (`CODNA_MEMORY_EMBED=local`, offline) vs **OpenAI `text-embedding-3-large`** via the key-gated engine `/v1/embeddings` (remote). Cursor/Codex omitted — already in SOAK-REPORT.md. Query-aware soft-rerank is live for both._

## Verdict — default to the bigram

| metric | OUR bigram | LLM embedding |
|---|--:|--:|
| Codna fix success | **95/97** | 95/97 |
| median fix time | **8.0s** | 9.0s |
| median fix cost | $0.05 | $0.05 |
| total fix cost | **$5.69** | $5.86 |
| **median index time** | **0.3s** (in-process) | 6.3s (API) |
| index worst-case | 60.2s | 113.5s |
| external dependency | **none (offline)** | `/v1/embeddings` API |

**The bigram matches the LLM embedder on fix success, time, and cost — while indexing ~20× faster, fully offline, and free.** The LLM embedding adds latency + an external dependency with no improvement in the actual fix outcome.

### Notes
- `traitlets` bigram `exit 1` was a **transient** fix hiccup (re-ran → `ok`, patch produced) — not embedding-related; the engine localizes independently of recall.
- `schedule` = a bad repo URL in the list (clone failed), not a Codna failure.
- Index time is the bigram's structural win: in-process Mojo kernel vs a network embedding call.

## Per-repo (fix time · cost · status · index)

| Repo | bigram | LLM | bigram idx | LLM idx |
|---|---|---|--:|--:|
| anyio | 9s $0.041 ok | 7s $0.066 ok | 0.9s | 24.2s |
| arrow | 13s $0.111 ok | 18s $0.104 ok | 1.2s | 13.7s |
| asgiref | 11s $0.059 ok | 9s $0.045 ok | 0.2s | 2.4s |
| attrs | 16s $0.108 ok | 14s $0.104 ok | 0.6s | 9.2s |
| bleach | 7s $0.043 ok | 7s $0.043 ok | 0.3s | 5.8s |
| blinker | 10s $0.05 ok | 10s $0.054 ok | 0.2s | 1.2s |
| boltons | 15s $0.015 ok | 7s $0.036 ok | 0.5s | 11.3s |
| cachetools | 6s $0.039 ok | 9s $0.038 ok | 0.2s | 3.8s |
| cattrs | 8s $0.04 ok | 9s $0.041 ok | 0.3s | 8.0s |
| click | 13s $0.052 ok | 13s $0.056 ok | 0.6s | 11.4s |
| colorama | 6s $0.043 ok | 7s $0.043 ok | 0.2s | 1.7s |
| coveragepy | 11s $0.159 ok | 13s $0.153 ok | 0.9s | 22.0s |
| croniter | 6s $0.029 ok | 14s $0.119 ok | 0.3s | 3.9s |
| deepdiff | 8s $0.069 ok | 11s $0.135 ok | 0.7s | 13.0s |
| dirty-equals | 6s $0.042 ok | 5s $0.043 ok | 0.2s | — |
| emoji | 7s $0.052 ok | 7s $0.083 ok | 0.2s | 2.1s |
| environs | 4s $0.026 ok | 7s $0.029 ok | 0.2s | 2.5s |
| faker | 11s $0.048 ok | 19s $0.091 ok | 2.3s | 43.6s |
| feedparser | 8s $0.093 ok | 10s $0.047 ok | 0.2s | 4.3s |
| flask | 7s $0.022 ok | 12s $0.111 ok | 0.3s | 8.1s |
| freezegun | 6s $0.041 ok | 7s $0.066 ok | 0.2s | 2.7s |
| furl | 5s $0.02 ok | 6s $0.022 ok | 0.3s | 2.3s |
| gunicorn | 8s $0.063 ok | 11s $0.039 ok | 1.2s | 46.3s |
| h11 | 6s $0.106 ok | 8s $0.042 ok | 0.2s | 2.6s |
| httpx | 11s $0.089 ok | 13s $0.089 ok | 0.4s | 11.1s |
| hug | 6s $0.038 ok | 7s $0.039 ok | 0.4s | 12.0s |
| humanize | 7s $0.023 ok | 7s $0.018 ok | 0.2s | 1.3s |
| idna | 6s $0.044 ok | 7s $0.044 ok | 60.2s | 113.5s |
| inflection | 5s $0.031 ok | 7s $0.02 ok | 0.2s | 1.6s |
| itsdangerous | 7s $0.065 ok | 6s $0.041 ok | 0.2s | 2.2s |
| jinja | 18s $0.11 ok | 12s $0.052 ok | 0.6s | 14.8s |
| jmespath.py | 6s $0.039 ok | 7s $0.04 ok | 0.2s | 3.1s |
| jsonpickle | 8s $0.086 ok | 6s $0.043 ok | 0.4s | 10.4s |
| jsonschema | 9s $0.049 ok | 13s $0.058 ok | 0.4s | 6.1s |
| loguru | 22s $0.035 ok | 9s $0.056 ok | 0.3s | 11.3s |
| markdown | 8s $0.078 ok | 10s $0.11 ok | 0.5s | 12.0s |
| marshmallow | 27s $0.098 ok | 12s $0.115 ok | 0.6s | 9.4s |
| mistune | 6s $0.055 ok | 7s $0.055 ok | 0.2s | 5.0s |
| more-itertools | 6s $0.03 ok | 8s $0.03 ok | 1.8s | 12.6s |
| multidict | 7s $0.036 ok | 9s $0.038 ok | 0.3s | 6.3s |
| oauthlib | 8s $0.05 ok | 10s $0.122 ok | 0.3s | 12.2s |
| packaging | 12s $0.087 ok | 58s $0.072 ok | 0.8s | 16.8s |
| paramiko | 12s $0.155 ok | 10s $0.052 ok | — | 19.6s |
| parse | 9s $0.056 ok | 8s $0.058 ok | 0.2s | 2.1s |
| pendulum | 10s $0.052 ok | 9s $0.088 ok | 0.4s | 14.7s |
| pexpect | 19s $0.118 ok | 7s $0.052 ok | 0.3s | 6.6s |
| platformdirs | 6s $0.047 ok | 8s $0.074 ok | 0.2s | 3.8s |
| pluggy | 6s $0.036 ok | 9s $0.065 ok | 0.2s | 2.8s |
| ply | 6s $0.036 ok | 7s $0.037 ok | 0.3s | 13.8s |
| prettytable | 6s $0.049 ok | 6s $0.051 ok | 0.4s | 4.5s |
| pygments | 14s $0.056 ok | 23s $0.094 ok | 1.0s | 19.0s |
| pyjwt | 39s $0.077 ok | 20s $0.019 ok | 0.3s | 4.3s |
| pyparsing | 7s $0.049 ok | 7s $0.049 ok | 2.2s | 16.0s |
| pyrsistent | 7s $0.039 ok | 21s $0.051 ok | 0.3s | 9.6s |
| pytest | 16s $0.315 ok | 13s $0.159 ok | 2.6s | 51.8s |
| python-pathspec | 6s $0.051 ok | 10s $0.128 ok | 0.3s | 8.9s |
| python-prompt-toolkit | 9s $0.049 ok | 10s $0.049 ok | 0.7s | 22.2s |
| python-semver | 9s $0.055 ok | 13s $0.066 ok | 0.2s | 2.6s |
| python-sortedcontainers | 7s $0.052 ok | 10s $0.057 ok | 0.3s | 6.3s |
| python-tabulate | 14s $0.072 ok | 19s $0.107 ok | 0.4s | 4.5s |
| pytz | 14s $0.056 ok | 11s $0.05 ok | 0.2s | 2.1s |
| pyyaml | 7s $0.06 ok | 6s $0.042 ok | 0.3s | 6.6s |
| requests | 7s $0.049 ok | 7s $0.05 ok | 0.4s | 6.0s |
| requests-oauthlib | 6s $0.035 ok | 8s $0.035 ok | 0.2s | 2.1s |
| rich | 9s $0.077 ok | 9s $0.052 ok | 0.6s | 14.5s |
| schedule | clone-failed | clone-failed | — | — |
| schema | 8s $0.037 ok | 6s $0.024 ok | 0.2s | 2.2s |
| shellingham | 7s $0.046 ok | 7s $0.046 ok | 0.2s | 0.8s |
| simplejson | 8s $0.072 ok | 6s $0.042 ok | 0.2s | 4.5s |
| six | 5s $0.019 ok | 5s $0.02 ok | 0.2s | 1.2s |
| sniffio | 8s $0.067 ok | 7s $0.042 ok | 0.1s | 1.1s |
| soupsieve | 11s $0.056 ok | 6s $0.047 ok | 0.3s | 8.0s |
| sqlparse | 9s $0.097 ok | 14s $0.11 ok | 0.2s | 5.0s |
| starlette | 7s $0.036 ok | 7s $0.036 ok | 0.4s | 13.3s |
| tenacity | 8s $0.045 ok | 14s $0.01 ok | 0.3s | 5.5s |
| texttable | 5s $0.029 ok | 5s $0.015 ok | 0.2s | 3.0s |
| tinydb | 8s $0.068 ok | 6s $0.066 ok | 0.2s | 3.5s |
| toml | 7s $0.042 ok | 6s $0.025 ok | 0.2s | 1.5s |
| tomlkit | 6s $0.041 ok | 9s $0.096 ok | 0.4s | 6.1s |
| toolz | 6s $0.042 ok | 6s $0.042 ok | 0.2s | 4.0s |
| tqdm | 6s $0.046 ok | 7s $0.047 ok | 0.3s | 5.4s |
| traitlets | 47s $None exit 1 | 15s $0.112 ok | 0.8s | 11.3s |
| trio | 8s $0.103 ok | 18s $0.249 ok | 0.7s | 19.7s |
| typer | 24s $0.095 ok | 19s $0.092 ok | 0.5s | 18.2s |
| uritemplate | 6s $0.051 ok | 6s $0.051 ok | 0.2s | 1.1s |
| urllib3 | 9s $0.052 ok | 12s $0.089 ok | 0.7s | 14.7s |
| uvicorn | 9s $0.042 ok | 7s $0.04 ok | 0.3s | 7.2s |
| validators | 11s $0.08 ok | 9s $0.047 ok | 0.2s | 2.7s |
| voluptuous | 7s $0.042 ok | 6s $0.026 ok | 0.3s | 4.2s |
| watchdog | 7s $0.064 ok | 6s $0.042 ok | 0.2s | 6.2s |
| wcwidth | 8s $0.047 ok | 9s $0.073 ok | 0.4s | 8.6s |
| websockets | 9s $0.081 ok | 50s $None exit 1 | 1.0s | 33.5s |
| werkzeug | 12s $0.054 ok | 11s $0.055 ok | 0.7s | 17.4s |
| wrapt | 8s $0.066 ok | 12s $0.086 ok | 0.5s | 18.9s |
| xmltodict | 8s $0.033 ok | 6s $0.022 ok | 0.2s | 2.3s |
| yarl | 8s $0.046 ok | 6s $0.038 ok | 0.5s | 9.5s |
| zipp | 9s $0.037 ok | 6s $0.036 ok | 0.2s | 1.7s |
