"""A website browser's network log as browser/v1 reads it and data/get exports it: lines read in order after a number,
a page at a time, a line still being written left for the next read; and HAR 1.2 made of them, each request and tunnel
one entry, its open and end lines joined by id, in the order they started, with the fields a grader reads, and bodies
decoded from gzip, deflate, br and zstd, cut at a limit, or left base64 when they can't be."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import zlib

import brotli
import pytest
import zstandard

from agentenv_protocol.website_browser import network
from agentenv_protocol.website_browser.network import NetworkLog, har

T0, T1, T2, T3 = "2026-11-11T10:00:00.000Z", "2026-11-11T10:00:00.100Z", "2026-11-11T10:00:00.300Z", \
    "2026-11-11T10:00:01.000Z"


def _line(seq: int, kind: str, record: dict) -> dict:
    return {"seq": seq, "at": T0, kind: record}


def _write(path, lines, tail: str = "") -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines) + tail)


def _body(data: bytes, **extra) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "base64": base64.b64encode(data).decode(),
            **extra}


def _exchange(key: str, request: dict, response: dict | None = None, started: str = T0, **end) -> list[dict]:
    opened = _line(1, "open", {"id": key, "kind": "exchange", "started": started, "request": request})
    ended = {"id": key, "ended": T2, **end}
    if response is not None:
        ended.update(answered=T1, response=response)
    return [opened, _line(2, "end", ended)]


def _get(url="https://shop.test/", **fields) -> dict:
    return {"method": "GET", "url": url, "http_version": "HTTP/1.1", "headers": [["Host", "shop.test"]], **fields}


def _ok(headers, body: dict | None = None, status=200, reason="OK") -> dict:
    return {"status": status, "reason": reason, "http_version": "HTTP/1.1", "headers": headers,
            **({"body": body} if body else {})}


def test_a_read_is_the_lines_after_a_number_in_order_a_page_at_a_time(tmp_path):
    path = tmp_path / "network.jsonl"
    _write(path, [_line(n, "end", {"id": str(n), "ended": T0}) for n in range(1, 6)])
    log = NetworkLog(path)

    first = log.read(after=0, limit=2)
    second = log.read(after=first["last"], limit=2)
    rest = log.read(after=second["last"])

    assert ([line["seq"] for line in first["lines"]], first["last"], first["more"]) == ([1, 2], 2, True)
    assert ([line["seq"] for line in second["lines"]], second["more"]) == ([3, 4], True)
    assert ([line["seq"] for line in rest["lines"]], rest["last"], rest["more"]) == ([5], 5, False)
    assert log.read(after=5) == {"lines": [], "last": 5, "more": False}
    assert [line["seq"] for line in log.read(after=3)["lines"]] == [4, 5]


def test_a_line_still_being_written_is_left_for_the_next_read_and_a_bad_line_passed_over(tmp_path):
    path = tmp_path / "network.jsonl"
    _write(path, [_line(1, "end", {"id": "1", "ended": T0})], tail='not json\n{"seq":2,"at":"')
    log = NetworkLog(path)

    assert [line["seq"] for line in log.read()["lines"]] == [1]
    with path.open("a") as file:
        file.write(f'{T0}","end":{{"id":"2","ended":"{T0}"}}}}\n')
    assert [line["seq"] for line in log.read(after=1)["lines"]] == [2]


def test_a_log_begun_again_shorter_is_read_from_its_start(tmp_path):
    path = tmp_path / "network.jsonl"
    _write(path, [_line(n, "end", {"id": str(n), "ended": T0, "pad": "x" * 200}) for n in range(1, 4)])
    log = NetworkLog(path)
    log.read()
    _write(path, [_line(4, "end", {"id": "4", "ended": T0})])

    assert [line["seq"] for line in log.read(after=3)["lines"]] == [4]


def test_a_line_that_isnt_numbered_is_passed_over_and_the_rest_read(tmp_path):
    path = tmp_path / "network.jsonl"
    _write(path, [_line(1, "end", {"id": "1", "ended": T0}), {"seq": None, "end": {"id": "2"}},
                  {"seq": "3", "end": {"id": "3"}}, {"seq": True}, [4], _line(5, "end", {"id": "5", "ended": T0})])

    assert [line["seq"] for line in NetworkLog(path).read()["lines"]] == [1, 5]
    assert [line["seq"] for line in NetworkLog(path).read(after=1)["lines"]] == [5]


def test_a_missing_log_reads_as_empty_and_exports_as_har_with_no_entries(tmp_path):
    log = NetworkLog(tmp_path / "none.jsonl")

    assert log.read(after=4) == {"lines": [], "last": 4, "more": False}
    assert har(log.every_line())["log"] == {"version": "1.2", "creator": {"name": "agentenv",
                                                                          "version": network._version()},
                                            "entries": []}


def test_a_read_stops_at_its_byte_limit_but_always_returns_one_line(tmp_path, monkeypatch):
    path = tmp_path / "network.jsonl"
    _write(path, [_line(n, "end", {"id": str(n), "ended": T0, "pad": "x" * 500}) for n in range(1, 4)])
    monkeypatch.setattr(network, "READ_BYTES", 100)

    page = NetworkLog(path).read()

    assert ([line["seq"] for line in page["lines"]], page["more"]) == ([1], True)
    assert [line["seq"] for line in NetworkLog(path).every_line()] == [1, 2, 3]


def test_an_exchange_is_one_entry_with_the_fields_a_grader_reads():
    form = b"user=amy&next=%2Fcart"
    request = _get("https://shop.test/login?from=home&q=", method="POST", headers=[
        ["Host", "shop.test"], ["Cookie", "sid=s1; theme=dark"], ["Referer", "https://shop.test/"],
        ["Content-Type", "application/x-www-form-urlencoded"], ["X-Shop-Key", "<removed>"]], body=_body(form))
    response = _ok([["Location", "https://shop.test/cart"], ["Content-Type", "text/html; charset=utf-8"],
                    ["Set-Cookie", "sid=s2; Path=/; HttpOnly; Secure; SameSite=Lax"]],
                   _body(b"<a>moved</a>"), status=303, reason="See Other")

    [entry] = har(_exchange("1-1", request, response))["log"]["entries"]

    assert (entry["startedDateTime"], entry["time"], entry["timings"]) == (T0, 300, {"send": 0, "wait": 100,
                                                                                      "receive": 200})
    sent = entry["request"]
    assert (sent["method"], sent["url"], sent["httpVersion"]) == ("POST", "https://shop.test/login?from=home&q=",
                                                                  "HTTP/1.1")
    assert sent["queryString"] == [{"name": "from", "value": "home"}, {"name": "q", "value": ""}]
    assert sent["cookies"] == [{"name": "sid", "value": "s1"}, {"name": "theme", "value": "dark"}]
    assert {"name": "X-Shop-Key", "value": "<removed>"} in sent["headers"]
    assert sent["postData"] == {"mimeType": "application/x-www-form-urlencoded", "text": "user=amy&next=%2Fcart"}
    assert sent["bodySize"] == len(form)
    answer = entry["response"]
    assert (answer["status"], answer["statusText"], answer["redirectURL"]) == (303, "See Other",
                                                                               "https://shop.test/cart")
    assert answer["cookies"] == [{"name": "sid", "value": "s2", "httpOnly": True, "secure": True, "path": "/",
                                  "sameSite": "Lax"}]
    assert answer["content"] == {"size": 12, "mimeType": "text/html; charset=utf-8", "text": "<a>moved</a>"}


def test_a_location_on_an_answer_not_a_redirect_is_no_redirect_url():
    [entry] = har(_exchange("1-1", _get(), _ok([["Location", "https://shop.test/x"]])))["log"]["entries"]

    assert entry["response"]["redirectURL"] == ""


@pytest.mark.parametrize("coding, encode", [
    ("gzip", gzip.compress),
    ("deflate", zlib.compress),
    ("deflate", lambda data: (lambda c: c.compress(data) + c.flush())(zlib.compressobj(wbits=-zlib.MAX_WBITS))),
    ("br", brotli.compress),
    ("zstd", lambda data: zstandard.ZstdCompressor().compress(data)),
    ("gzip, br", lambda data: brotli.compress(gzip.compress(data))),
], ids=["gzip", "deflate", "raw-deflate", "br", "zstd", "gzip-then-br"])
def test_an_answer_body_is_decoded_from_its_coding(coding, encode):
    text = json.dumps({"items": list(range(200))}).encode()
    wire = encode(text)
    response = _ok([["Content-Type", "application/json"], ["Content-Encoding", coding]], _body(wire))

    [entry] = har(_exchange("1-1", _get(), response))["log"]["entries"]

    content = entry["response"]["content"]
    assert json.loads(content["text"]) == {"items": list(range(200))}
    assert (content["size"], entry["response"]["bodySize"]) == (len(text), len(wire))
    assert content["compression"] == len(text) - len(wire)


@pytest.mark.parametrize("coding, wire", [("compress", b"\x1f\x9d..."), ("gzip", b"not gzip at all")],
                         ids=["unknown-coding", "broken-gzip"])
def test_an_answer_body_that_cant_be_decoded_stays_base64_with_its_coding_named(coding, wire):
    response = _ok([["Content-Type", "text/plain"], ["Content-Encoding", coding]], _body(wire))

    [entry] = har(_exchange("1-1", _get(), response))["log"]["entries"]

    content = entry["response"]["content"]
    assert (content["encoding"], content["_undecoded"], base64.b64decode(content["text"])) == ("base64", coding, wire)


def test_an_answer_body_is_cut_at_the_limit_once_decoded_and_says_so(monkeypatch):
    monkeypatch.setattr(network, "DECODED_LIMIT", 1000)
    wire = gzip.compress(b"a" * 5000)
    response = _ok([["Content-Type", "text/plain"], ["Content-Encoding", "gzip"]], _body(wire))

    [entry] = har(_exchange("1-1", _get(), response))["log"]["entries"]

    content = entry["response"]["content"]
    assert (content["text"], content["_truncated"]) == ("a" * 1000, True)


def test_a_body_the_proxy_cut_says_so_and_decodes_as_far_as_it_goes():
    text = b"x" * 4000
    wire = gzip.compress(text)
    response = _ok([["Content-Type", "text/plain"], ["Content-Encoding", "gzip"]],
                   {**_body(wire[:len(wire) // 2]), "size": len(wire), "truncated": True})

    [entry] = har(_exchange("1-1", _get(), response))["log"]["entries"]

    content = entry["response"]["content"]
    assert content["_truncated"] is True and entry["response"]["bodySize"] == len(wire)
    assert "compression" not in content


def test_a_body_is_decoded_in_its_charset():
    response = _ok([["Content-Type", "text/plain; charset=iso-8859-1"]], _body("café".encode("latin-1")))

    [entry] = har(_exchange("1-1", _get(), response))["log"]["entries"]

    assert entry["response"]["content"]["text"] == "café"


def test_an_answer_the_proxy_kept_no_bytes_of_has_its_size_and_sha256():
    image = bytes(range(256)) * 4
    body = {"size": len(image), "sha256": hashlib.sha256(image).hexdigest()}

    [entry] = har(_exchange("1-1", _get(), _ok([["Content-Type", "image/png"]], body)))["log"]["entries"]

    assert entry["response"]["content"] == {"size": 1024, "mimeType": "image/png",
                                            "_sha256": hashlib.sha256(image).hexdigest()}


def test_a_websocket_is_its_ws_url_its_resource_type_and_its_message_counts():
    request = _get("https://chat.test/socket", headers=[["Upgrade", "websocket"], ["Connection", "Upgrade"]])
    counts = {"messages_sent": 1, "messages_received": 2, "bytes_sent": 14, "bytes_received": 9}

    [entry] = har(_exchange("1-1", request, _ok([["Upgrade", "websocket"]], status=101,
                                                reason="Switching Protocols"), websocket=counts))["log"]["entries"]

    assert (entry["request"]["url"], entry["_resourceType"], entry["_websocket"]) == (
        "wss://chat.test/socket", "websocket", counts)


def test_a_tunnel_is_a_connect_answered_as_established_with_its_address_and_bytes():
    lines = [_line(1, "open", {"id": "1-1", "kind": "tunnel", "host": "cdn.test", "port": 443, "started": T0,
                               "address": "198.51.100.7"}),
             _line(2, "end", {"id": "1-1", "ended": T3, "bytes_sent": 812, "bytes_received": 40210})]

    [entry] = har(lines)["log"]["entries"]

    assert (entry["request"]["method"], entry["request"]["url"]) == ("CONNECT", "https://cdn.test:443/")
    assert (entry["response"]["status"], entry["response"]["statusText"]) == (200, "Connection established")
    assert entry["serverIPAddress"] == "198.51.100.7" and entry["time"] == 1000
    assert entry["_tunnel"] == {"host": "cdn.test", "port": 443, "address": "198.51.100.7", "bytes_sent": 812,
                                "bytes_received": 40210}


def test_a_plain_request_the_stand_in_forwarded_is_its_method_and_url_its_answer_unread():
    lines = [_line(1, "open", {"id": "1-1", "kind": "tunnel", "host": "shop.test", "port": 80, "method": "GET",
                               "url": "http://shop.test/a?b=1", "started": T0, "address": "198.51.100.7"}),
             _line(2, "end", {"id": "1-1", "ended": T2, "bytes_sent": 80, "bytes_received": 500})]

    [entry] = har(lines)["log"]["entries"]

    assert (entry["request"]["method"], entry["request"]["url"]) == ("GET", "http://shop.test/a?b=1")
    assert entry["request"]["queryString"] == [{"name": "b", "value": "1"}]
    assert entry["response"]["status"] == 0 and entry["_tunnel"]["bytes_received"] == 500


def test_a_failed_a_refused_and_an_open_connection_are_entries_that_say_so():
    lines = [_line(1, "end", {"id": "1-1", "kind": "tunnel", "host": "gone.test", "port": 443, "started": T0,
                              "ended": T1, "failed": {"reason": "doesn't resolve", "error": "gaierror"}}),
             _line(2, "end", {"id": "1-2", "kind": "refused", "method": "GET", "url": "evil.test", "started": T1,
                              "ended": T1, "refused": {"reason": "no site of this computer"}}),
             _line(3, "open", {"id": "1-3", "kind": "exchange", "started": T2, "request": _get()})]

    failed, refused, still = har(lines)["log"]["entries"]

    assert failed["_failed"] == {"reason": "doesn't resolve", "error": "gaierror"}
    assert (failed["response"]["status"], failed["request"]["url"]) == (0, "https://gone.test:443/")
    assert refused["_refused"] == {"reason": "no site of this computer"}
    assert (refused["request"]["method"], refused["request"]["url"]) == ("GET", "http://evil.test/")
    assert still["_open"] is True and still["response"]["status"] == 0 and still["time"] == 0


def test_entries_are_in_the_order_they_started_not_the_order_they_ended():
    slow = _exchange("1-1", _get("https://shop.test/slow"), _ok([]), started=T0)
    fast = _exchange("1-2", _get("https://shop.test/fast"), _ok([]), started=T1)
    lines = [slow[0], fast[0], fast[1], slow[1]]

    assert [entry["request"]["url"] for entry in har(lines)["log"]["entries"]] == [
        "https://shop.test/slow", "https://shop.test/fast"]


@pytest.mark.asyncio
async def test_the_har_text_is_every_line_as_har(tmp_path):
    path = tmp_path / "network.jsonl"
    _write(path, _exchange("1-1", _get(), _ok([["Content-Type", "text/plain"]], _body(b"hi"))))

    exported = json.loads(await NetworkLog(path).har_text())

    assert exported["log"]["entries"][0]["response"]["content"]["text"] == "hi"


@pytest.mark.parametrize("coding, encode", [
    ("gzip", gzip.compress),
    ("br", brotli.compress),
    ("zstd", lambda data: zstandard.ZstdCompressor().compress(data)),
], ids=["gzip", "br", "zstd"])
def test_a_small_body_that_decodes_huge_stops_at_the_limit_answer_and_request_alike(coding, encode):
    wire = encode(b"a" * (64 << 20))
    response = _ok([["Content-Type", "text/plain"], ["Content-Encoding", coding]], _body(wire))
    request = _get(method="POST", headers=[["Content-Type", "text/plain"], ["Content-Encoding", coding]],
                   body=_body(wire))

    [entry] = har(_exchange("1-1", request, response))["log"]["entries"]

    answer, sent = entry["response"]["content"], entry["request"]["postData"]
    assert (len(answer["text"]), answer["_truncated"]) == (network.DECODED_LIMIT, True)
    assert (len(sent["text"]), sent["_truncated"]) == (network.REQUEST_DECODED_LIMIT, True)


def test_a_line_that_cant_be_an_entry_is_left_out_and_counted_and_the_rest_export():
    good = _exchange("1-1", _get(), _ok([["Content-Type", "text/plain"]], _body(b"hi")))
    lines = [*good, {"seq": 3, "at": T0, "open": {"kind": "exchange"}},
             {"seq": 4, "at": T0, "end": {"id": 7}}, {"seq": 5, "at": T0, "open": {"id": "1-2", "started": 5}},
             {"seq": 6, "at": T0, "end": {"id": "1-3", "started": T0, "response": {"status": "x", "headers": 3}}},
             {"seq": 7, "at": T0}]

    exported = har(lines)["log"]

    assert [entry["response"]["content"]["text"] for entry in exported["entries"]] == ["hi"]
    assert exported["_unreadable"] == 5


def test_a_cookies_expiry_is_iso_8601_and_one_that_doesnt_parse_is_left_out():
    response = _ok([["Set-Cookie", "a=1; Expires=Wed, 09 Jun 2027 10:18:14 GMT"],
                    ["Set-Cookie", "b=2; Expires=someday"], ["Set-Cookie", "c=3; Expires=Fri, 31 Dec 9999 23:59:59 -0100"]])

    [entry] = har(_exchange("1-1", _get(), response))["log"]["entries"]

    first, second, third = entry["response"]["cookies"]
    assert first["expires"] == "2027-06-09T10:18:14.000Z" and "expires" not in second and "expires" not in third


def test_an_answer_the_proxy_couldnt_read_says_so():
    [entry] = har(_exchange("1-1", _get("http://live.test/"), None,
                            unparsed={"reason": "an answer the proxy can't read as HTTP", "error": "bad"}))["log"][
        "entries"]

    assert entry["_unparsed"]["error"] == "bad" and entry["response"]["status"] == 0
