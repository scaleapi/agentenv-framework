// One wait of SECONDS of universe time in Node: node uwait.js API SECONDS. Prints READY, then DONE <mono> <wall>.
const [api, s] = [process.argv[2], parseFloat(process.argv[3])];
const n0 = process.hrtime.bigint(), w0 = Date.now();
console.log("READY");
const done = () => console.log(`DONE ${Number(process.hrtime.bigint() - n0) / 1e9} ${(Date.now() - w0) / 1e3}`);
if (api === "node_settimeout") setTimeout(done, s * 1000);
else if (api === "node_setinterval") { const t = setInterval(() => { clearInterval(t); done(); }, s * 1000); }
else if (api === "node_promises_settimeout") require("timers/promises").setTimeout(s * 1000).then(done);
