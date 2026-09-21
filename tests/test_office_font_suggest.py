"""Office 翻译的字体建议：内置语言查表，自定义语言问 LLM，失败回退拉丁轨。

运行: conda run -n mineru python -m pytest tests/test_office_font_suggest.py -v
"""
import asyncio

import httpx
import pytest

import app


def test_parse_suggestion_valid():
    got = app._office_parse_font_suggestion(
        '{"fonts": ["Sylfaen", "Noto Sans Georgian"], "track": "latin"}')
    assert got == {"fonts": ["Sylfaen", "Noto Sans Georgian"], "track": "latin"}


def test_parse_suggestion_with_fences_and_broken_json():
    got = app._office_parse_font_suggestion(
        '```json\n{"fonts": ["Angsana New"], "track": "cs",}\n```')
    assert got["fonts"] == ["Angsana New"]
    assert got["track"] == "cs"


def test_parse_suggestion_bad_track_normalized():
    got = app._office_parse_font_suggestion('{"fonts": ["X"], "track": "whatever"}')
    assert got["track"] == "latin"


def test_parse_suggestion_garbage_raises():
    with pytest.raises(ValueError):
        app._office_parse_font_suggestion("我不知道")


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app),
                             base_url="http://t")


def test_endpoint_builtin_lang_needs_no_llm(monkeypatch):
    async def _boom(lang):
        raise AssertionError("内置语言不该触发 LLM")
    monkeypatch.setattr(app, "_office_font_suggest_llm", _boom)

    async def run():
        async with _client() as c:
            r = await c.post("/office/font_suggest", data={"lang": "泰语"})
            return r.json()
    got = asyncio.run(run())
    assert got == {"fonts": ["TH Sarabun New", "Angsana New"], "track": "cs"}


def test_endpoint_custom_lang_uses_llm_and_caches(monkeypatch):
    calls = []

    async def _fake(lang):
        calls.append(lang)
        return {"fonts": ["Sylfaen"], "track": "latin"}
    monkeypatch.setattr(app, "_office_font_suggest_llm", _fake)
    app._office_font_cache.pop("格鲁吉亚语", None)

    async def run():
        async with _client() as c:
            a = (await c.post("/office/font_suggest", data={"lang": "格鲁吉亚语"})).json()
            b = (await c.post("/office/font_suggest", data={"lang": "格鲁吉亚语"})).json()
            return a, b
    a, b = asyncio.run(run())
    assert a == b == {"fonts": ["Sylfaen"], "track": "latin"}
    assert calls == ["格鲁吉亚语"], "第二次必须命中缓存，不再打 LLM"


def test_endpoint_llm_failure_falls_back(monkeypatch):
    async def _fail(lang):
        raise RuntimeError("gateway down")
    monkeypatch.setattr(app, "_office_font_suggest_llm", _fail)
    app._office_font_cache.pop("克林贡语", None)

    async def run():
        async with _client() as c:
            return (await c.post("/office/font_suggest", data={"lang": "克林贡语"})).json()
    got = asyncio.run(run())
    assert got["track"] == "latin"
    assert got["fonts"], "失败也要给可用的兜底字体"
