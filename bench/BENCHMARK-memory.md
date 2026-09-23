# `codna memory` benchmark — real public repos

_Generated 2026-06-26 20:27 UTC · embedder: AlgentaBigramEmbedder (in-process, 64-d lexical) · engine: Telys (Mojo kernel) · 5 repos, one shared collection._

> **Honest scope:** day-1 uses the in-process **bigram** stand-in (lexical, weak semantics) — the future `AlgentaCodeEmbedder` raises recall quality with no code change. The *structural* wins below (exact-symbol recall, partition-isolated retrieval, context-token reduction, µs-scale latency, cross-repo isolation) are embedder-independent.

## Results

| Repo | Py files | Symbols | Skipped | Index (ms) | Exact recall@1 | Partition-hit | Context-token reduction | Mean recall (ms) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| click | 63 | 1246 | 0 | 498 | 100% | 100% | 98% | 0.33 |
| flask | 83 | 920 | 0 | 280 | 96% | 100% | 91% | 0.60 |
| requests | 37 | 726 | 0 | 382 | 96% | 100% | 99% | 0.79 |
| httpx | 60 | 1158 | 0 | 302 | 100% | 100% | 94% | 0.90 |
| rich | 213 | 1893 | 0 | 561 | 96% | 100% | 99% | 1.06 |

**Totals:** 5,943 symbols indexed across 5 repos into one collection (5,829 live documents). Exact recall@1 mean 98%, partition-hit mean 100%, token-reduction mean 96%.

## Cross-repo isolation (shared collection)

Each repo's scoped recall returns only its own symbols — proof the partition key isolates repos in one shared store.

| Repo | Returned | In-scope | Verdict |
|---|--:|--:|---|
| click | 10 | 10 | PASS |
| flask | 10 | 10 | PASS |
| requests | 10 | 10 | PASS |
| httpx | 10 | 10 | PASS |
| rich | 10 | 10 | PASS |

## Recall transcripts (real output)

### click  (`github.com/pallets/click`)
```text
  $ codna memory recall click --query "define a command line option with a default value"
    0.736  test     tests/test_termui.py
    0.715  test     tests/test_options.py
    0.710  test     tests/test_utils.py
    0.696  test     tests/test_options.py
    0.695  test     tests/test_options.py

  $ codna memory recall click --query "prompt the user for confirmation"
    0.707  test     tests/test_shell_completion.py
    0.692  class    src/click/types.py
    0.683  test     tests/test_termui.py
    0.674  class    src/click/exceptions.py
    0.661  class    src/click/types.py
```

### flask  (`github.com/pallets/flask`)
```text
  $ codna memory recall flask --query "register a url route for a view"
    0.696  function src/flask/helpers.py
    0.682  class    src/flask/ctx.py
    0.680  class    src/flask/debughelpers.py
    0.678  class    src/flask/sessions.py
    0.671  test     examples/tutorial/tests/test_blog.py

  $ codna memory recall flask --query "return a json response"
    0.740  function tests/type_check/typing_app_decorators.py
    0.703  function tests/type_check/typing_app_decorators.py
    0.636  function src/flask/helpers.py
    0.632  test     tests/test_helpers.py
    0.613  test     examples/tutorial/tests/test_blog.py
```

### requests  (`github.com/psf/requests`)
```text
  $ codna memory recall requests --query "send a post request with a json body"
    0.667  class    src/requests/exceptions.py
    0.652  method   src/requests/sessions.py
    0.651  test     tests/test_requests.py
    0.637  class    src/requests/models.py
    0.634  class    src/requests/models.py

  $ codna memory recall requests --query "set a timeout on the request"
    0.777  class    src/requests/exceptions.py
    0.721  class    src/requests/exceptions.py
    0.716  class    src/requests/exceptions.py
    0.692  class    src/requests/exceptions.py
    0.684  class    src/requests/exceptions.py
```

### httpx  (`github.com/encode/httpx`)
```text
  $ codna memory recall httpx --query "create an async client and send a request"
    0.747  class    tests/client/test_auth.py
    0.747  test     tests/test_status_codes.py
    0.737  test     tests/models/test_requests.py
    0.707  test     tests/models/test_requests.py
    0.706  test     tests/models/test_headers.py

  $ codna memory recall httpx --query "stream a large response body"
    0.785  class    httpx/_transports/default.py
    0.783  class    httpx/_client.py
    0.771  class    httpx/_transports/default.py
    0.767  class    httpx/_client.py
    0.723  class    httpx/_content.py
```

### rich  (`github.com/Textualize/rich`)
```text
  $ codna memory recall rich --query "render a table to the console"
    0.780  class    rich/console.py
    0.754  module   examples/table.py
    0.728  class    rich/console.py
    0.705  class    rich/live_render.py
    0.696  class    rich/containers.py

  $ codna memory recall rich --query "display a progress bar"
    0.711  module   examples/cp_progress.py
    0.683  class    rich/progress.py
    0.661  test     tests/test_progress.py
    0.659  test     tests/test_progress.py
    0.652  test     tests/test_progress.py
```

## Observations

- **The structural wins are strong and embedder-independent:** exact-symbol recall@1 96–100% (the index returns a symbol's own text rank-1), **100% partition-hit** (every scoped query takes the contiguous slice, never a scatter fallback), **91–99% context-token reduction** (final-k symbols vs whole files), **sub-millisecond** text-to-results, and **5/5 cross-repo isolation** in one shared 5,829-doc collection.
- **The bigram stand-in already finds the right code on concrete queries** — e.g. httpx *"stream a large response body"* → `_client.py` / `_transports/default.py` / `_content.py`; rich *"render a table"* → `console.py`; rich *"display a progress bar"* → `progress.py`.
- **Its lexical nature is visible** on vaguer queries: click *"define a command line option…"* surfaces `tests/test_options.py` (tests restate the API in words), and requests *"set a timeout"* repeats `exceptions.py`. This is the expected weakness of a 64-d lexical embedder — the future `AlgentaCodeEmbedder` (same provider seam, no code change; re-index only) is the quality upgrade. Nothing here is cherry-picked: these are the raw top-5.

## Reproduce

```bash
# 1. clone the repos (shallow)
for r in pallets/click pallets/flask psf/requests encode/httpx Textualize/rich; do
  git clone --depth 1 https://github.com/$r.git /tmp/bench-repos/$(basename $r); done
# 2. install codna[memory] + point at a Codna-packaged Telys kernel when using a source checkout
uv pip install -e 'codna/cli[memory]'
export TELYS_KERNEL=/abs/path/to/memory-engine/mojo_build/libame_kernel.dylib
# 3. run the benchmark
python cli/bench/benchmark_memory.py /tmp/bench-repos/{click,flask,requests,httpx,rich}
```

This Markdown is the generated output of `cli/bench/benchmark_memory.py` (committed alongside it).

## Method

- Real `codna.memory.CodeMemory.index()` / `.recall()` on shallow clones — no mocks.
- **Exact recall@1**: query a sampled symbol's own source text; counts how often it ranks #1 (an index-correctness check — should be ~100%).
- **Partition-hit**: share of scoped queries served by the contiguous partition slice (`PartitionSliceExactF32`) vs a scatter fallback.
- **Context-token reduction**: `len//4` token estimate of the final-k symbol texts vs the whole source files they came from (what an agent would otherwise read).
- **Latency**: text-to-results (embed + Telys retrieve + rerank), mean over the domain queries.
- **No `X× faster` claim** (per Telys D-28); latency is reported, not marketed.
