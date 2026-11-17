"""A browser's network log as its proxies write it: the shared site proxy keeps each exchange whole, the request as
the browser sent it, a site's sign-in headers removed, and the answer as the browser got it, its body up to a limit
by its type, a WebSocket's messages counted, and each tunnel, failure and refusal; the base's stand-in keeps each
connection. Both number their lines after the last the file holds, and the protocol's HAR of what they wrote has the
fields a grader reads. The proxies are network programs, so these tests open sockets, on loopback alone."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import socket
import ssl
import struct

import pytest
from agentenv_protocol.website_browser.network import NetworkLog as NetworkReader
from agentenv_protocol.website_browser.network import har

from agent_env.env.envs.computer.utils import site_proxy
from agent_env.env.envs.computer.utils.site_proxy import NetworkLog
from tst.unit.env.envs.test_computer_site_proxy import (Upstream, _config, _exchange, _forward_port, _live_hosts_at,
                                                        _proxy)
from tst.unit.env.envs.test_computer_stand_in_proxy import GITHUB, _ask, _serving, _stand_in

pytestmark = pytest.mark.usefixtures("socket_enabled")

PAGE = b"<p>caf\xe9</p>"
BIG = b"a" * (1 << 20) + b"b" * 1000
IMAGE = bytes(range(256)) * 8
API = json.dumps({"ok": True, "items": [1, 2, 3]}).encode()


def _chunked(data: bytes, size: int = 7) -> bytes:
    return b"".join(b"%x\r\n%s\r\n" % (len(data[at:at + size]), data[at:at + size])
                    for at in range(0, len(data), size)) + b"0\r\n\r\n"


def _masked(opcode: int, payload: bytes, final: bool = True) -> bytes:
    mask = b"\x01\x02\x03\x04"
    return bytes([(0x80 if final else 0) | opcode, 0x80 | len(payload)]) + mask + bytes(
        byte ^ mask[at % 4] for at, byte in enumerate(payload))


def _site(upstream_port):
    """A site's answers by path: a page in Latin-1 setting a cookie under the route, a sign-in redirecting onto the
    route, gzipped JSON sent chunked, a text answer past the limit, an image, and a WebSocket that answers twice."""
    async def respond(start, headers, body, reader, writer):
        path = start.split(" ")[1].removeprefix("/website/notes")
        if path.startswith("/page"):
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=iso-8859-1\r\nContent-Length: %d\r\n"
                         b"Set-Cookie: sid=s2; Path=/website/notes/; HttpOnly\r\n\r\n%s" % (len(PAGE), PAGE))
        elif path == "/login":
            writer.write(f"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{upstream_port()}/website/notes/home\r\n"
                         f"Content-Length: 0\r\n\r\n".encode())
        elif path == "/api":
            writer.write(b"HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n" + _chunked(gzip.compress(API)))
        elif path == "/big":
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s" % (
                len(BIG), BIG))
        elif path == "/logo.png":
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: %d\r\n\r\n%s" % (
                len(IMAGE), IMAGE))
        elif path == "/socket":
            writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
            await writer.drain()
            await reader.readexactly(len(_masked(1, b"he", final=False)) + len(_masked(0, b"y")) +
                                     len(_masked(9, b"")))
            writer.write(b"\x81\x02hi" + b"\x81\x7e" + struct.pack(">H", 200) + b"z" * 200)
        else:
            writer.write(b"HTTP/1.1 204 No Content\r\n\r\n")
    return respond


async def _over_tls(port, issued, request: bytes):
    answer = await _exchange(port, request, server_hostname="notes.test", cadata=issued.ca_pem)
    answer[3][1].close()
    return answer


async def _ended(path, timeout_s: float = 5.0) -> None:
    """Until the log at ``path`` has an entry and every entry in it has ended, as a connection's does once both its
    ends have closed."""
    for _ in range(int(timeout_s / 0.02)):
        if (entries := _entries(path)) and not any("_open" in entry for entry in entries):
            return
        await asyncio.sleep(0.02)


def _entries(path) -> list[dict]:
    return har(NetworkReader(path).every_line())["log"]["entries"]


def _headers(entry_part) -> dict:
    return {header["name"].lower(): header["value"] for header in entry_part["headers"]}


@pytest.mark.asyncio
async def test_the_site_proxy_keeps_each_exchange_whole_as_the_browser_sent_and_got_it(tmp_path):
    path = tmp_path / "network.jsonl"
    log = NetworkLog.open(str(path))
    form = b"user=amy&password=pw"
    multipart = (b"--XyZ\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nTrip\r\n--XyZ\r\n"
                 b"Content-Disposition: form-data; name=\"file\"; filename=\"a.txt\"\r\nContent-Type: text/plain"
                 b"\r\n\r\nnotes\r\n--XyZ--\r\n")
    async with Upstream() as upstream:
        upstream.respond = _site(lambda: upstream.port)
        config, issued = _config(tmp_path, upstream.port)
        config.sites[0].sign_in_headers = ["X-Notes-Key"]
        async with _proxy(config, network=log) as port:
            page = await _over_tls(port, issued, b"GET /page?q=caf%C3%A9&n= HTTP/1.1\r\nHost: notes.test\r\n"
                                                 b"Cookie: sid=s1; theme=dark\r\nX-Notes-Key: secret\r\n"
                                                 b"Referer: https://notes.test/\r\nSec-Fetch-Mode: navigate\r\n\r\n")
            login = await _over_tls(port, issued, b"POST /login HTTP/1.1\r\nHost: notes.test\r\nContent-Type: "
                                                  b"application/x-www-form-urlencoded\r\nContent-Length: %d\r\n\r\n%s"
                                                  % (len(form), form))
            api = await _over_tls(port, issued, b"POST /api HTTP/1.1\r\nHost: notes.test\r\nContent-Type: "
                                                b"application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
                                                + _chunked(b'{"title":"Trip"}'))
            await _over_tls(port, issued, b"POST /upload HTTP/1.1\r\nHost: notes.test\r\nContent-Type: "
                                          b"multipart/form-data; boundary=XyZ\r\nContent-Length: %d\r\n\r\n%s"
                                          % (len(multipart), multipart))
            await _over_tls(port, issued, b"GET /big HTTP/1.1\r\nHost: notes.test\r\n\r\n")
            await _over_tls(port, issued, b"GET /logo.png HTTP/1.1\r\nHost: notes.test\r\n\r\n")
        async with _proxy(config, scheme="http", network=log) as port:
            plain = await _exchange(port, b"GET /page HTTP/1.1\r\nHost: notes.test\r\n\r\n")
            plain[3][1].close()

    entries = _entries(path)
    assert [(entry["request"]["method"], entry["request"]["url"], entry["response"]["status"]) for entry in entries] \
        == [("GET", "https://notes.test/page?q=caf%C3%A9&n=", 200), ("POST", "https://notes.test/login", 302),
            ("POST", "https://notes.test/api", 201), ("POST", "https://notes.test/upload", 204),
            ("GET", "https://notes.test/big", 200), ("GET", "https://notes.test/logo.png", 200),
            ("GET", "http://notes.test/page", 308)]
    first = entries[0]
    assert first["request"]["queryString"] == [{"name": "q", "value": "café"}, {"name": "n", "value": ""}]
    assert first["request"]["cookies"] == [{"name": "sid", "value": "s1"}, {"name": "theme", "value": "dark"}]
    assert _headers(first["request"])["x-notes-key"] == "<removed>"
    assert _headers(first["request"])["referer"] == "https://notes.test/"
    assert _headers(first["request"])["sec-fetch-mode"] == "navigate"
    assert _headers(first["response"]) == page[1]
    assert first["response"]["cookies"] == [{"name": "sid", "value": "s2", "httpOnly": True, "secure": False,
                                             "path": "/"}]
    assert first["response"]["content"] == {"size": len(PAGE), "mimeType": "text/html; charset=iso-8859-1",
                                            "text": "<p>café</p>"}
    assert entries[1]["request"]["postData"] == {"mimeType": "application/x-www-form-urlencoded",
                                                 "text": "user=amy&password=pw"}
    assert entries[1]["response"]["redirectURL"] == login[1]["location"] == "https://notes.test/home"
    assert entries[2]["request"]["postData"]["text"] == '{"title":"Trip"}'
    assert json.loads(entries[2]["response"]["content"]["text"]) == json.loads(API) == json.loads(gzip.decompress(
        api[2]))
    assert entries[3]["request"]["postData"] == {"mimeType": "multipart/form-data; boundary=XyZ",
                                                 "text": multipart.decode()}
    big = entries[4]["response"]["content"]
    assert (big["size"], len(big["text"]), big["_truncated"]) == (len(BIG), 1 << 20, True)
    assert entries[5]["response"]["content"] == {"size": len(IMAGE), "mimeType": "image/png",
                                                 "_sha256": hashlib.sha256(IMAGE).hexdigest()}
    assert entries[6]["response"]["redirectURL"] == plain[1]["location"] == "https://notes.test/page"
    assert all(entry["timings"]["wait"] <= entry["time"] for entry in entries)
    assert not any(key in entry for entry in entries for key in ("_open", "_failed", "_refused"))


@pytest.mark.asyncio
async def test_the_site_proxy_counts_a_websockets_messages_each_way(tmp_path):
    path = tmp_path / "network.jsonl"
    async with Upstream() as upstream:
        upstream.respond = _site(lambda: upstream.port)
        config, issued = _config(tmp_path, upstream.port)
        async with _proxy(config, network=NetworkLog.open(str(path))) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port, server_hostname="notes.test",
                                                           ssl=ssl.create_default_context(cadata=issued.ca_pem))
            writer.write(b"GET /socket HTTP/1.1\r\nHost: notes.test\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n"
                         b"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n")
            assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 101 ")
            writer.write(_masked(1, b"he", final=False) + _masked(0, b"y") + _masked(9, b""))
            back = await reader.read()
            writer.close()
            await _ended(path)

    [entry] = _entries(path)
    assert back == b"\x81\x02hi" + b"\x81\x7e" + struct.pack(">H", 200) + b"z" * 200
    assert (entry["request"]["url"], entry["response"]["status"], entry["_resourceType"]) == (
        "wss://notes.test/socket", 101, "websocket")
    assert entry["_websocket"] == {"messages_sent": 1, "messages_received": 2, "bytes_sent": 8 + 7 + 6,
                                   "bytes_received": len(back)}


@pytest.mark.parametrize("cut", [1, 2, 3, 5, 9])
def test_websocket_frames_are_counted_across_any_cut_and_control_frames_arent_messages(cut):
    stream = (_masked(1, b"x" * 100, final=False) + _masked(10, b"") + _masked(0, b"y" * 20) +
              b"\x82\x7f" + struct.pack(">Q", 70000) + b"\0" * 70000 + b"\x88\x00")
    frames = site_proxy.Frames()
    for at in range(0, len(stream), cut):
        frames.add(stream[at:at + cut])

    assert (frames.messages, frames.bytes) == (2, len(stream))


@pytest.mark.asyncio
async def test_the_forward_port_keeps_a_sites_tunnel_by_its_requests_a_live_hosts_whole_and_refusals_and_failures(
        tmp_path, monkeypatch):
    path = tmp_path / "network.jsonl"
    async with Upstream() as upstream, Upstream() as live:
        config, issued = _config(tmp_path, upstream.port)
        config.direct = ["github.com", "gone.github.com"]
        with _live_hosts_at(monkeypatch, live.port):
            real = site_proxy.asyncio.open_connection

            async def gone_or_live(host, port, **kwargs):
                if host == "gone.github.com":
                    raise socket.gaierror(-2, "Name or service not known")
                return await real(host, port, **kwargs)

            monkeypatch.setattr(site_proxy.asyncio, "open_connection", gone_or_live)
            async with _forward_port(config, network=NetworkLog.open(str(path))) as port:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"CONNECT notes.test:443 HTTP/1.1\r\nHost: notes.test:443\r\n\r\n")
                await reader.readuntil(b"\r\n\r\n")
                await writer.start_tls(ssl.create_default_context(cadata=issued.ca_pem), server_hostname="notes.test")
                await _exchange(port, b"GET /a?b=1 HTTP/1.1\r\nHost: notes.test\r\n\r\n", reader_writer=(reader, writer))
                writer.close()
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com:443\r\n\r\n")
                await reader.readuntil(b"\r\n\r\n")
                await _exchange(port, b"GET /x HTTP/1.1\r\nHost: github.com\r\n\r\n", reader_writer=(reader, writer))
                writer.close()
                for request in (b"POST http://github.com/a?b=1 HTTP/1.1\r\nHost: github.com\r\nContent-Length: 2\r\n"
                                b"Cookie: gh=1\r\n\r\nhi",
                                b"CONNECT gone.github.com:443 HTTP/1.1\r\nHost: gone.github.com:443\r\n\r\n",
                                b"CONNECT evil.test:443 HTTP/1.1\r\nHost: evil.test:443\r\n\r\n"):
                    (await _exchange(port, request))[3][1].close()
                await asyncio.sleep(0.3)

    site, tunnel, plain, gone, refused = _entries(path)
    assert (site["request"]["url"], site["response"]["content"]["size"]) == ("https://notes.test/a?b=1", 5)
    assert (tunnel["request"]["method"], tunnel["request"]["url"], tunnel["serverIPAddress"]) == (
        "CONNECT", "https://github.com:443/", "127.0.0.1")
    assert tunnel["_tunnel"]["bytes_sent"] > 0 and tunnel["_tunnel"]["bytes_received"] > 0
    assert (plain["request"]["method"], plain["request"]["url"], plain["request"]["postData"]["text"]) == (
        "POST", "http://github.com/a?b=1", "hi")
    assert plain["request"]["cookies"] == [{"name": "gh", "value": "1"}]
    assert (plain["response"]["status"], plain["response"]["content"].get("text")) == (200, None)
    assert plain["response"]["content"]["size"] == 5 and plain["serverIPAddress"] == "127.0.0.1"
    assert gone["_failed"] == {"reason": "doesn't resolve", "error": "gaierror: [Errno -2] Name or service not known"}
    assert (refused["_refused"], refused["request"]["url"]) == ({"reason": "no site of this computer"},
                                                                "https://evil.test:443/")
    assert not any("_open" in entry for entry in (site, tunnel, plain, gone, refused))


@pytest.mark.asyncio
async def test_a_request_still_going_is_an_open_entry_until_it_ends(tmp_path):
    path = tmp_path / "network.jsonl"
    answered = asyncio.Event()

    async def slow(start, headers, body, reader, writer):
        await answered.wait()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

    async with Upstream(slow) as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _proxy(config, network=NetworkLog.open(str(path))) as port:
            asked = asyncio.create_task(_over_tls(port, issued, b"GET /slow HTTP/1.1\r\nHost: notes.test\r\n\r\n"))
            for _ in range(100):
                if path.stat().st_size:
                    break
                await asyncio.sleep(0.02)
            [going] = _entries(path)
            answered.set()
            await asyncio.wait_for(asked, 5)

    [done] = _entries(path)
    assert (going["_open"], going["response"]["status"]) == (True, 0)
    assert "_open" not in done and done["response"]["status"] == 200


def test_the_site_proxy_numbers_after_the_last_line_the_file_holds_and_ends_a_cut_one(tmp_path):
    path = tmp_path / "network.jsonl"
    path.write_bytes(b'{"seq":1,"at":"t","end":{"id":"a"}}\n{"seq":7,"at":"t","end":{"id":"b"}}\n{"seq":8,"at":')
    log = NetworkLog.open(str(path))
    log.write("end", {"id": "c", "ended": "t"})

    assert [line["seq"] for line in NetworkReader(path).every_line()] == [1, 7, 8]
    assert path.read_bytes().splitlines()[2] == b'{"seq":8,"at":'


def test_the_site_proxy_opens_its_log_readable_by_all_and_never_through_a_link(tmp_path):
    path, elsewhere = tmp_path / "network.jsonl", tmp_path / "elsewhere"
    previous = os.umask(0o077)
    try:
        NetworkLog.open(str(path)).write("end", {"id": "a"})
    finally:
        os.umask(previous)
    elsewhere.write_text("")
    (tmp_path / "linked.jsonl").symlink_to(elsewhere)

    assert path.stat().st_mode & 0o777 == 0o644
    with pytest.raises(OSError):
        NetworkLog.open(str(tmp_path / "linked.jsonl"))
    assert elsewhere.read_text() == ""


def test_the_site_proxy_logs_a_write_it_couldnt_make_and_carries_on(tmp_path, capsys):
    log = NetworkLog.open(str(tmp_path / "network.jsonl"))
    log.file.close()
    log.write("end", {"id": "a"})

    [logged] = [json.loads(line) for line in capsys.readouterr().err.splitlines() if "network log" in line]
    assert logged["error"]["connection"] == "network log"


def test_the_stand_in_keeps_each_connection_and_each_one_that_failed(monkeypatch, tmp_path):
    path = tmp_path / "network.jsonl"
    with _serving(monkeypatch, names={"nowhere.test": []}) as proxy:
        module = proxy.module
        monkeypatch.setattr(module, "NETWORK", module.NetworkLog(str(path)))
        _ask(proxy.port, b"CONNECT site.test:443 HTTP/1.1\r\nHost: site.test:443\r\n\r\n",
             then=b"GET / HTTP/1.1\r\nHost: site.test\r\n\r\n")
        _ask(proxy.port, b"GET http://site.test/a?b=1 HTTP/1.1\r\nHost: site.test\r\n\r\n")
        _ask(proxy.port, b"CONNECT nowhere.test:443 HTTP/1.1\r\nHost: nowhere.test:443\r\n\r\n")

    lines = NetworkReader(path).every_line()
    assert [line["seq"] for line in lines] == [1, 2, 3, 4, 5]
    assert [next(kind for kind in ("open", "end") if kind in line) for line in lines] == [
        "open", "end", "open", "end", "end"]
    tunnel, forwarded, failed = har(lines)["log"]["entries"]
    assert (tunnel["request"]["method"], tunnel["request"]["url"], tunnel["serverIPAddress"]) == (
        "CONNECT", "https://site.test:443/", GITHUB)
    assert tunnel["_tunnel"]["bytes_sent"] > 0 and tunnel["_tunnel"]["bytes_received"] > 0
    assert (forwarded["request"]["method"], forwarded["request"]["url"]) == ("GET", "http://site.test/a?b=1")
    assert failed["_failed"] == {"reason": "doesn't resolve", "error": "gaierror: [Errno -2] Name or service not known"}
    assert failed["request"]["url"] == "https://nowhere.test:443/"


def test_the_stand_in_numbers_after_the_last_line_the_file_holds_and_ends_a_cut_one(tmp_path):
    path = tmp_path / "network.jsonl"
    path.write_bytes(b'{"seq":4,"at":"t","end":{"id":"a"}}\n{"seq":5,"at":')
    module = _stand_in()
    module.NetworkLog(str(path)).write("end", {"id": "b"})

    assert [line["seq"] for line in NetworkReader(path).every_line()] == [4, 5]



async def _live(handle):
    """A live host on loopback, each connection given to ``handle(reader, writer)``."""
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _head(reader) -> bytes:
    return await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)


@pytest.mark.asyncio
async def test_a_live_host_s_100_continue_reaches_the_browser_before_it_sends_its_body(tmp_path, monkeypatch):
    path = tmp_path / "network.jsonl"

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
        body = await reader.readexactly(5)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\n" + body)
        await writer.drain()
        writer.close()

    server, port = await _live(handle)
    async with Upstream() as upstream:
        config, _ = _config(tmp_path, upstream.port)
        config.direct = ["live.test"]
        with _live_hosts_at(monkeypatch, port):
            async with _forward_port(config, network=NetworkLog.open(str(path))) as proxy:
                reader, writer = await asyncio.open_connection("127.0.0.1", proxy)
                writer.write(b"POST http://live.test/up HTTP/1.1\r\nHost: live.test\r\nContent-Length: 5\r\n"
                             b"Expect: 100-continue\r\n\r\n")
                await writer.drain()
                go_on = await _head(reader)
                writer.write(b"hello")
                answered = await _head(reader)
                echoed = await reader.readexactly(5)
                writer.close()
                await _ended(path)
    server.close()

    [entry] = _entries(path)
    assert go_on.startswith(b"HTTP/1.1 100 ") and answered.startswith(b"HTTP/1.1 200 ") and echoed == b"hello"
    assert (entry["response"]["status"], entry["request"]["postData"]["text"],
            entry["response"]["content"]["text"]) == (200, "hello", "hello")


@pytest.mark.asyncio
async def test_a_live_host_that_turns_an_upload_down_early_is_answered_at_once(tmp_path, monkeypatch):
    path = tmp_path / "network.jsonl"
    done = asyncio.Event()

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        await done.wait()
        writer.close()

    server, port = await _live(handle)
    async with Upstream() as upstream:
        config, _ = _config(tmp_path, upstream.port)
        config.direct = ["live.test"]
        with _live_hosts_at(monkeypatch, port):
            async with _forward_port(config, network=NetworkLog.open(str(path))) as proxy:
                reader, writer = await asyncio.open_connection("127.0.0.1", proxy)
                writer.write(b"POST http://live.test/up HTTP/1.1\r\nHost: live.test\r\nContent-Length: %d\r\n\r\n"
                             % (64 << 20))
                sending = asyncio.ensure_future(_send_slowly(writer, 64 << 20))
                answered = await _head(reader)
                sending.cancel()
                writer.close()
                await _ended(path)
    done.set()
    server.close()

    [entry] = _entries(path)
    assert answered.startswith(b"HTTP/1.1 413 ")
    assert entry["response"]["status"] == 413 and entry["request"]["bodySize"] < 64 << 20


async def _send_slowly(writer, size: int) -> None:
    for _ in range(size // 65536):
        writer.write(b"x" * 65536)
        await writer.drain()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [b"HTTP/1.1 200 OK\r\nNoColonHere\r\n\r\nhello",
                                    b"HTTP/1.1 200 OK\nContent-Length: 5\n\nhello"], ids=["no-colon", "bare-lf"])
async def test_a_live_host_s_answer_the_proxy_cant_read_is_relayed_as_it_came(tmp_path, monkeypatch, answer):
    path = tmp_path / "network.jsonl"

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(answer)
        await writer.drain()
        writer.close()

    server, port = await _live(handle)
    async with Upstream() as upstream:
        config, _ = _config(tmp_path, upstream.port)
        config.direct = ["live.test"]
        with _live_hosts_at(monkeypatch, port):
            async with _forward_port(config, network=NetworkLog.open(str(path))) as proxy:
                reader, writer = await asyncio.open_connection("127.0.0.1", proxy)
                writer.write(b"GET http://live.test/ HTTP/1.1\r\nHost: live.test\r\n\r\n")
                got = await asyncio.wait_for(reader.read(), 5)
                writer.close()
                await _ended(path)
    server.close()

    [entry] = _entries(path)
    assert got == answer
    assert entry["_unparsed"]["reason"] == "an answer the proxy can't read as HTTP"


def test_with_no_network_log_a_body_is_counted_but_neither_kept_nor_hashed():
    capture = site_proxy.SiteProxy(site_proxy.Config(user=None, sites=[]))._capture(1 << 20)
    capture.add(b"x" * 4096)

    assert capture.record() == {"size": 4096} and not capture.data and capture.digest is None


def test_the_stand_in_turns_down_a_bad_content_length_before_it_connects(monkeypatch, tmp_path):
    path = tmp_path / "network.jsonl"
    with _serving(monkeypatch) as proxy:
        monkeypatch.setattr(proxy.module, "NETWORK", proxy.module.NetworkLog(str(path)))
        answer = _ask(proxy.port, b"POST http://site.test/a HTTP/1.1\r\nHost: site.test\r\nContent-Length: abc\r\n\r\n")

    assert answer.startswith(b"HTTP/1.1 400 ") and proxy.opened == []
    assert NetworkReader(path).every_line() == []


LEFT_OPEN = (b'{"seq":1,"at":"2026-11-12T10:00:00.000Z","open":{"id":"9-0-1","kind":"exchange",'
             b'"started":"2026-11-12T10:00:00.000Z","request":{"method":"GET","url":"https://notes.test/a"}}}\n'
             b'{"seq":2,"at":"2026-11-12T10:00:00.100Z","open":{"id":"9-0-2","kind":"tunnel","host":"cdn.test",'
             b'"port":443,"started":"2026-11-12T10:00:00.100Z"}}\n'
             b'{"seq":3,"at":"2026-11-12T10:00:00.200Z","end":{"id":"9-0-2","ended":"2026-11-12T10:00:00.200Z"}}\n')


def test_the_site_proxy_started_again_ends_what_the_one_before_left_open_and_never_reuses_an_id(tmp_path):
    path = tmp_path / "network.jsonl"
    path.write_bytes(LEFT_OPEN)
    log = NetworkLog.open(str(path))
    again = NetworkLog.open(str(path))

    first, second = _entries(path)
    assert first["_failed"] == site_proxy.RESTARTED and first["request"]["url"] == "https://notes.test/a"
    assert "_failed" not in second and not any("_open" in entry for entry in (first, second))
    assert [line["seq"] for line in NetworkReader(path).every_line()] == [1, 2, 3, 4]
    assert log.key() != again.key() and log.key().startswith(f"{os.getpid()}-3-")


def test_the_stand_in_started_again_ends_what_the_one_before_left_open(tmp_path):
    path = tmp_path / "network.jsonl"
    path.write_bytes(LEFT_OPEN)
    module = _stand_in()
    module.NetworkLog(str(path))

    first, second = _entries(path)
    assert first["_failed"] == module.RESTARTED and "_failed" not in second
