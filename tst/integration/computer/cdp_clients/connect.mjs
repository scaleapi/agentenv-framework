// Connects Playwright and Puppeteer to a website browser's CDP address and prints, as JSON, what each saw.
// Usage: node connect.mjs <check> <http_url> <ws_url> <site url> [exact], <check> being:
//   together: Playwright by http_url and Puppeteer by ws_url at once, then Playwright by ws_url, each opening the site;
//   refused: both clients at the address with its token changed, or as given with "exact", each expected to fail.
import { chromium } from "playwright-core";
import puppeteer from "puppeteer-core";

const [check, httpUrl, wsUrl, site, exact] = process.argv.slice(2);
const TIMEOUT_MS = 60_000;

async function playwright(endpoint) {
  const browser = await chromium.connectOverCDP(endpoint, { timeout: TIMEOUT_MS });
  const context = browser.contexts()[0];
  const page = await context.newPage();
  await page.goto(site, { timeout: TIMEOUT_MS });
  const seen = { title: await page.title(), contexts: browser.contexts().length };
  await page.close();
  return seen;
}

async function pptr(endpoint) {
  const browser = await puppeteer.connect({ browserWSEndpoint: endpoint, protocolTimeout: TIMEOUT_MS });
  const page = await browser.newPage();
  await page.goto(site, { timeout: TIMEOUT_MS });
  const seen = { title: await page.title(), version: await browser.version() };
  await page.close();
  await browser.disconnect();
  return seen;
}

async function settled(work) {
  try {
    return { ok: await work };
  } catch (error) {
    return { error: String(error.message || error).split("\n")[0] };
  }
}

const wrong = (url) => (exact ? url : url.replace(/\/cdp\/[^/]+\//, "/cdp/not-a-token/"));
let result;
if (check === "together") {
  const [byHttp, byWs] = await Promise.all([settled(playwright(httpUrl)), settled(pptr(wsUrl))]);
  result = { playwright_http: byHttp, puppeteer_ws: byWs, playwright_ws: await settled(playwright(wsUrl)) };
} else {
  result = { playwright: await settled(playwright(wrong(httpUrl))), puppeteer: await settled(pptr(wrong(wsUrl))) };
}
console.log(JSON.stringify(result));
process.exit(0);
