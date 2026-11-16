// Drives a website browser through CDP addresses scoped to leases and prints, as JSON, what each client saw.
// Usage: node scoped.mjs <buyer ws_url> <seller ws_url> <default ws_url> <frame page url>:
//   Playwright on the buyer's address: its default context's cookies and pages, then a new page at the frame page, a
//   click inside its cross-origin iframe and a download from it; Puppeteer on the seller's: its pages, its contexts
//   and a new page's cookies; Playwright on the default address: its pages; then Playwright on the buyer's and the
//   seller's addresses at once, each clicking inside the iframe ten times.
import { chromium } from "playwright-core";
import puppeteer from "puppeteer-core";

const [buyerWs, sellerWs, defaultWs, framePage] = process.argv.slice(2);
const TIMEOUT_MS = 60_000;
const CLICKS = 10;

async function settled(work) {
  try {
    return { ok: await work };
  } catch (error) {
    return { error: String(error.message || error).split("\n")[0] };
  }
}

async function framed(context) {
  const page = await context.newPage();
  await page.goto(framePage, { timeout: TIMEOUT_MS });
  return page;
}

async function onBuyer() {
  const browser = await chromium.connectOverCDP(buyerWs, { timeout: TIMEOUT_MS });
  const context = browser.contexts()[0];
  const seen = {
    contexts: browser.contexts().length,
    cookies: (await context.cookies()).map((cookie) => `${cookie.name}=${cookie.value}`),
    pages: context.pages().map((page) => page.url()),
  };
  const page = await framed(context);
  const frame = page.frameLocator("iframe");
  await frame.locator("#press").click({ timeout: TIMEOUT_MS });
  seen.clicked = await frame.locator("body").getAttribute("data-clicked");
  seen.frame_url = page.frames()[1]?.url();
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: TIMEOUT_MS }),
                                        frame.locator("#get").click({ timeout: TIMEOUT_MS })]);
  seen.download = download.suggestedFilename();
  await page.close();
  return seen;
}

async function onSeller() {
  const browser = await puppeteer.connect({ browserWSEndpoint: sellerWs, protocolTimeout: TIMEOUT_MS });
  const seen = {
    contexts: browser.browserContexts().length,
    pages: (await browser.pages()).map((page) => page.url()),
  };
  const page = await browser.newPage();
  await page.goto(framePage, { timeout: TIMEOUT_MS });
  seen.cookies = (await page.cookies(framePage)).map((cookie) => `${cookie.name}=${cookie.value}`);
  await page.close();
  await browser.disconnect();
  return seen;
}

async function onDefault() {
  const browser = await chromium.connectOverCDP(defaultWs, { timeout: TIMEOUT_MS });
  return { contexts: browser.contexts().length, pages: browser.contexts()[0].pages().map((page) => page.url()) };
}

async function clicking(endpoint) {
  const browser = await chromium.connectOverCDP(endpoint, { timeout: TIMEOUT_MS });
  const page = await framed(browser.contexts()[0]);
  const frame = page.frameLocator("iframe");
  const started = Date.now();
  for (let i = 0; i < CLICKS; i++) {
    await frame.locator("#press").click({ timeout: TIMEOUT_MS });
  }
  const seen = { clicked: await frame.locator("body").getAttribute("data-clicked"), ms: Date.now() - started };
  await page.close();
  return seen;
}

const result = {
  buyer: await settled(onBuyer()),
  seller: await settled(onSeller()),
  default: await settled(onDefault()),
};
const [buyerClicks, sellerClicks] = await Promise.all([settled(clicking(buyerWs)), settled(clicking(sellerWs))]);
result.at_once = { buyer: buyerClicks, seller: sellerClicks };
console.log(JSON.stringify(result));
process.exit(0);
