"""Live smoke test: the browser tool actually browses. Gated by ``CODNA_BROWSER_TOOL_E2E=1``.

**Why this test exists, and why it is gated.** Everything else about the browser tool — the switch, the
pinned package, the flags, the server entry, the six tool bodies' contracts — is asserted without a
browser in ``test_browser_tool.py`` and ``test_browser_tool_wiring.py``. What those cannot prove is the
thing the tool is FOR: that a real Playwright MCP server starts, opens a real page, and that clicking,
typing, reading and screenshotting through it work. That needs Node, a downloaded browser and a few
seconds, so it is opt-in:

    CODNA_BROWSER_TOOL_E2E=1 python -m pytest tests/test_browser_tool_e2e.py

**What it asserts, in order**, because each step is only meaningful after the one before:

1. the server starts and advertises the six Playwright tools the integration depends on;
2. a loopback HTTP server (a real page, not a ``data:`` URL) navigates and its accessible text is read;
3. a click runs and the page's text CHANGES — the assertion a snapshot alone cannot make;
4. typing into an input lands, read back from the page;
5. a screenshot is written and the bytes are a real PNG;
6. waiting for text works, and waiting for something absent is a TIMEOUT rather than a hang;
7. a ``data:`` URL works with no server at all — the fastest way to check markup.

**The page is served over loopback on purpose.** ``file://`` is blocked by Playwright's own policy unless
the tool is widened with ``CODNA_BROWSER_FILE_ACCESS=unrestricted``, and a test that requires the widening
switch would stop proving the DEFAULT configuration works. A loopback HTTP server is also what a
just-built app is served from, so this exercises the real shape of the goal's "browser check".

**Nothing here touches the network.** Every URL is ``127.0.0.1`` on an ephemeral port, or a ``data:`` URL.
``npx`` may fetch the pinned package once on a cold cache — that is the package manager, not the test.
"""
from __future__ import annotations

import http.server
import os
import socket
import threading
from pathlib import Path
from typing import Final

import pytest

from codna import browser_bridge, browser_tool
from codna.browser_bridge import BrowserToolError, PlaywrightClient

ENABLE_ENV: Final = "CODNA_BROWSER_TOOL_E2E"

#: The page the loopback server serves. Small, self-contained, and every interaction observable in the
#: accessibility tree: a heading to find, a button that changes text, an input that accepts a value.
PAGE: Final = """<!doctype html>
<html>
  <head><title>Codna browser smoke</title></head>
  <body>
    <h1>Codna browser smoke</h1>
    <p id="status">idle</p>
    <button id="go" onclick="document.getElementById('status').textContent = 'clicked-ok'">Go</button>
    <input id="name" aria-label="name" />
    <script>
      // A value the typing test reads back from the DOM rather than from the input's default.
      document.getElementById('name').addEventListener('input', (event) => {
        document.getElementById('status').textContent = 'typed:' + event.target.value;
      });
    </script>
  </body>
</html>
"""


def _e2e_enabled() -> bool:
    return (os.environ.get(ENABLE_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


pytestmark = pytest.mark.skipif(
    not _e2e_enabled(),
    reason=f"live browser smoke test; set {ENABLE_ENV}=1 to run it (needs Node.js and a browser)",
)


@pytest.fixture(scope="module")
def page_url() -> str:
    """A loopback HTTP server for the page above, on an ephemeral port, torn down after the module."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _PageHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``PAGE`` at ``/``. A real HTTP response, so this is indistinguishable from an app."""

    def do_GET(self) -> None:  # noqa: N802 - the stdlib's spelling
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence the per-request log; a passing test should print nothing."""


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> PlaywrightClient:
    """One real session for the module: starting Chromium per test would dominate the run time."""
    if not browser_tool.browser_tool_state({"CODNA_BROWSER_TOOL": "1"})["launchable"]:
        pytest.skip("npx is not on PATH; the browser tool cannot start on this machine")
    config = browser_tool.browser_tool_config({
        "CODNA_RUNTIME_ROOT": str(tmp_path_factory.mktemp("browser-e2e")),
    })
    session = PlaywrightClient(config)
    try:
        session.start()
        yield session
    finally:
        session.close()


# ── 1. the server starts and advertises the tools the integration depends on ───────────────────


def test_the_running_server_advertises_every_required_tool(client: PlaywrightClient) -> None:
    """The pinned package must still ship the six tools the contract names.

    This is the test a version bump has to satisfy: if ``browser_take_screenshot`` is renamed upstream, the
    Codna tool that drives it would fail at the first call — here, instead, one test goes red.
    """
    listed = client.list_tools()
    missing = [name for name in browser_tool.REQUIRED_PLAYWRIGHT_TOOLS if name not in listed]
    assert missing == [], f"{browser_tool.PLAYWRIGHT_MCP_PACKAGE} is missing {missing}; listed {listed}"


# ── 2-4. navigate, read, click, type — the assertion a screenshot cannot make ──────────────────


def test_navigate_reads_the_page_as_accessible_text(client: PlaywrightClient, page_url: str) -> None:
    result = client.navigate(page_url)
    assert result.is_error is False, result.text
    # The heading is in the page, and Playwright's snapshot is where a model reads it.
    assert "Codna browser smoke" in result.text


def test_a_click_CHANGES_the_page(client: PlaywrightClient, page_url: str) -> None:
    """The point of a browser check: act, then observe a DIFFERENT page.

    A snapshot before and after is the whole contract — if this passed without the text changing, the tool
    would be reporting a page it never touched.
    """
    client.navigate(page_url)
    before = client.snapshot()
    assert "clicked-ok" not in before.text

    clicked = client.click("#go", element="Go")
    assert clicked.is_error is False, clicked.text

    after = client.snapshot()
    assert "clicked-ok" in after.text


def test_typing_lands_and_is_read_back_from_the_page(
    client: PlaywrightClient, page_url: str,
) -> None:
    client.navigate(page_url)
    typed = client.type_text("#name", "codna-browser", element="name")
    assert typed.is_error is False, typed.text

    # The page's own listener echoes the value into #status, so this asserts the DOM received the input,
    # not merely that a fill call returned successfully.
    assert "typed:codna-browser" in client.snapshot().text


def test_wait_for_text_returns_once_it_appears(client: PlaywrightClient, page_url: str) -> None:
    client.navigate(page_url)
    client.click("#go", element="Go")
    waited = client.wait_for(text="clicked-ok", time_s=10)
    assert waited.is_error is False, waited.text


def test_waiting_for_absent_text_TIMES_OUT_rather_than_hanging(
    client: PlaywrightClient, page_url: str,
) -> None:
    """A wait that never resolves must report an error, not block the tool forever."""
    client.navigate(page_url)
    result = client.wait_for(text="this-text-is-not-on-the-page", time_s=1)
    assert result.is_error is True
    assert "Timeout" in result.text or "timeout" in result.text


# ── 5. the screenshot is a real image, written where the config says ──────────────────────────


def test_a_screenshot_is_written_and_is_a_real_png(
    client: PlaywrightClient, page_url: str,
) -> None:
    client.navigate(page_url)
    result = client.screenshot("smoke.png")
    assert result.is_error is False, result.text
    assert result.path == "smoke.png"
    assert len(result.images) == 1
    image = result.images[0]
    # A PNG signature, not merely non-empty bytes: a 0-byte file would also satisfy "a file exists".
    assert image["mime_type"] == "image/png"
    assert image["data"][:8].encode("ascii") == b"iVBORw0K"
    written = Path(client.config.output_dir) / "smoke.png"
    assert written.is_file()
    assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert image["bytes"] == written.stat().st_size


def test_a_screenshot_outside_the_output_directory_is_refused(
    client: PlaywrightClient, page_url: str,
) -> None:
    client.navigate(page_url)
    with pytest.raises(BrowserToolError) as caught:
        client.screenshot("../escape.png")
    assert caught.value.code == "browser_screenshot_path_invalid"


# ── 6-7. a data: URL needs no server at all, and the tool bodies answer as text ───────────────


def test_a_data_url_works_with_no_server_and_the_body_answers_as_json(
    client: PlaywrightClient,
) -> None:
    """The quickest markup check: no server, no file, no widening of the file policy."""
    url = "data:text/html,<h1>inline-smoke</h1><p>no-server-needed</p>"
    navigate, snapshot = client.navigate(url), client.snapshot()
    assert navigate.is_error is False, navigate.text
    assert "inline-smoke" in snapshot.text


def test_the_registered_bodies_drive_the_real_session(
    client: PlaywrightClient, page_url: str, monkeypatch,
) -> None:
    """End to end through the public tool bodies — the functions `mcp_server` registers.

    ``browser_client`` is pointed at THIS session so the module-level bodies (which a real MCP client reaches
    through the registered tool) are exercised against the same live browser.
    """
    monkeypatch.setattr(browser_bridge, "browser_client", lambda config=None: client)

    import json

    navigated = json.loads(browser_bridge.browser_navigate_json(page_url))
    assert navigated["is_error"] is False
    assert "Codna browser smoke" in navigated["text"]

    clicked = json.loads(browser_bridge.browser_click_json("#go", "Go"))
    assert clicked["is_error"] is False
    assert "clicked-ok" in json.loads(browser_bridge.browser_snapshot_json())["text"]

    shot = json.loads(browser_bridge.browser_screenshot_json("body.png"))
    assert shot["is_error"] is False
    assert shot["images"][0]["mime_type"] == "image/png"
    assert shot["images"][0]["bytes"] > 0


def test_an_unreachable_host_is_a_structured_error_not_a_hang(client: PlaywrightClient) -> None:
    """A dead port must answer with an error the agent can act on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    result = client.call("browser_navigate", {"url": f"http://127.0.0.1:{dead_port}/"},
                         timeout_s=30)
    assert result.is_error is True
