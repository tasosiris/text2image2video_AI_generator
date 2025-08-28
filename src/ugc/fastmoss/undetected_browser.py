from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright


@dataclass
class BrowserConfig:
    headless: bool = True
    user_agent: Optional[str] = None
    locale: str = "en-US"
    timezone_id: str = "UTC"
    viewport: Optional[dict] = None
    proxy: Optional[str] = None
    slow_mo_ms: int = 0
    devtools: bool = False
    record_video_dir: Optional[str] = None


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


async def launch_stealth_browser(config: Optional[BrowserConfig] = None) -> tuple[Playwright, Browser, BrowserContext]:
    """Launch a Chromium browser with a set of stealthy defaults.

    Notes:
        - While this reduces automation signals, no solution guarantees undetectability.
        - Prefer using a real profile and sane timings; avoid exotic flags.
    """

    config = config or BrowserConfig()
    user_agent = config.user_agent or DEFAULT_USER_AGENT

    pw: Playwright = await async_playwright().start()
    args = [
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--lang=en-US,en",
    ]
    if config.proxy:
        args.append(f"--proxy-server={config.proxy}")

    if not config.headless and config.devtools:
        args.append("--auto-open-devtools-for-tabs")

    browser: Browser = await pw.chromium.launch(
        headless=config.headless,
        args=args,
        slow_mo=config.slow_mo_ms if config.slow_mo_ms else 0,
    )

    context_kwargs = dict(
        user_agent=user_agent,
        locale=config.locale,
        timezone_id=config.timezone_id,
        viewport=config.viewport or {"width": 1366, "height": 768},
    )

    if config.record_video_dir:
        context_kwargs["record_video_dir"] = config.record_video_dir

    context: BrowserContext = await browser.new_context(**context_kwargs)

    # Stealth tweaks applied in the context via init script
    stealth_js = r"""
// Pass the Chrome Test.
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

// Mimic plugins
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });

// Mimic languages
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

// Permissions query override
const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
if (originalQuery) {
  window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
      Promise.resolve({ state: Notification.permission }) :
      originalQuery(parameters)
  );
}

// WebGL vendor/renderer spoof
const getParameterProxy = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function (parameter) {
  if (parameter === 37445) return 'Intel Inc.'; // UNMASKED_VENDOR_WEBGL
  if (parameter === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
  return getParameterProxy.apply(this, [parameter]);
};

// Remove playwright traces
delete window.__playwright;
delete window.__driver_evaluate;
    """
    await context.add_init_script(stealth_js)

    return pw, browser, context


async def open_new_page(context: BrowserContext) -> Page:
    page: Page = await context.new_page()
    # Reasonable navigation timeout
    page.set_default_navigation_timeout(109000000)
    page.set_default_timeout(100000000)
    return page


async def graceful_close(playwright: Playwright, browser: Browser) -> None:
    try:
        await browser.close()
    except Exception:
        pass
    try:
        await playwright.stop()
    except Exception:
        pass


async def main_demo() -> None:
    pw, browser, context = await launch_stealth_browser()
    page = await open_new_page(context)
    await page.goto("https://example.com", wait_until="domcontentloaded")
    await graceful_close(pw, browser)


if __name__ == "__main__":
    asyncio.run(main_demo())
