// Agents acting at once on a website browser through CDP addresses scoped to leases, while each context is recorded;
// prints, as JSON, what each did. Usage: node recorded.mjs <buyer ws_url> <seller ws_url> <default ws_url> <site>:
//   Playwright on the buyer's address: on its red tab, clicks the link that opens an orange popup, closes the popup
//   after a while, then clicks on the red tab again; Puppeteer on the seller's: makes a browser context of its own and
//   opens a yellow page in it; Playwright on the default address: clicks on its green tab now and then.
import { chromium } from "playwright-core";
import puppeteer from "puppeteer-core";

const [buyerWs, sellerWs, defaultWs, site] = process.argv.slice(2);
const TIMEOUT_MS = 60_000;
const HOLD_MS = 4_000;

async function settled(work) {
  try {
    return { ok: await work };
  } catch (error) {
    return { error: String(error.message || error).split("\n")[0] };
  }
}

function hold(ms = HOLD_MS) {
  return new Promise((done) => setTimeout(done, ms));
}

async function onBuyer() {
  const browser = await chromium.connectOverCDP(buyerWs, { timeout: TIMEOUT_MS });
  const page = browser.contexts()[0].pages().find((each) => each.url().endsWith("/red"));
  await page.click("#here", { timeout: TIMEOUT_MS });
  await hold();
  const [popup] = await Promise.all([page.waitForEvent("popup", { timeout: TIMEOUT_MS }),
                                     page.click("#pop", { timeout: TIMEOUT_MS })]);
  await popup.waitForLoadState("load", { timeout: TIMEOUT_MS });
  await popup.click("#here", { timeout: TIMEOUT_MS });
  await hold();
  const popped = popup.url();
  await popup.close();
  await hold(1_000);
  await page.click("#here", { timeout: TIMEOUT_MS });
  await hold();
  return { popup: popped };
}

async function onSeller() {
  const browser = await puppeteer.connect({ browserWSEndpoint: sellerWs, protocolTimeout: TIMEOUT_MS });
  const own = await browser.createBrowserContext();
  const page = await own.newPage();
  await page.goto(`${site}yellow`, { timeout: TIMEOUT_MS });
  await page.click("#here");
  await hold(2 * HOLD_MS);
  const seen = { own: page.url() };
  await browser.disconnect();
  return seen;
}

async function onDefault() {
  const browser = await chromium.connectOverCDP(defaultWs, { timeout: TIMEOUT_MS });
  const page = browser.contexts()[0].pages().find((each) => each.url().endsWith("/green"));
  for (let i = 0; i < 4; i++) {
    await page.click("#here", { timeout: TIMEOUT_MS });
    await hold(HOLD_MS / 2);
  }
  return { clicks: 4 };
}

const [buyer, seller, onTheDefault] = await Promise.all([settled(onBuyer()), settled(onSeller()),
                                                         settled(onDefault())]);
console.log(JSON.stringify({ buyer, seller, default: onTheDefault }));
process.exit(0);
