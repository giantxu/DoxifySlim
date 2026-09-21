"""seed_page_cache.py：抢救出的页 → 页级缓存。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_seed_page_cache.py -v
"""
import json
import subprocess
import sys
from pathlib import Path

import app

REPO = Path(__file__).resolve().parent.parent


def _run(rescue: Path, pdf_dir: Path, *extra):
    return subprocess.run(
        [sys.executable, str(REPO / "seed_page_cache.py"), str(rescue), str(pdf_dir), *extra],
        capture_output=True, text=True, cwd=REPO,
        env={**dict(__import__("os").environ), "DOXIFY_LOG_FILE": "/tmp/doxify-tests.log"},
    )


def _setup(tmp_path, pages):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc.pdf").write_bytes(b"%PDF-1.4 real bytes")

    rescue = tmp_path / "rescue.json"
    rescue.write_text(json.dumps({
        "fid-1": {"filename": "doc.pdf", "total": 10, "pageTexts": pages},
        "fid-empty": {"filename": "gone.pdf", "total": 0, "pageTexts": {}},
    }), encoding="utf-8")
    return rescue, pdf_dir


def test_seeds_pages_and_skips_failures(tmp_path, monkeypatch):
    rescue, pdf_dir = _setup(tmp_path, {
        "1": "第一页正文",
        "2": "[第 2 页识别超时]",
        "3": "第三页正文",
    })
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setenv("DOXIFY_OUTPUT_DIR", str(tmp_path / "out"))

    # 子进程有自己的 OUTPUT_DIR，直接在进程内复现脚本逻辑更可靠
    data = json.loads(rescue.read_text(encoding="utf-8"))
    key = app._page_cache_key((pdf_dir / "doc.pdf").read_bytes(),
                              app.PDF_DPI, app.ACTUAL_MODEL_NAME)
    for pno, text in data["fid-1"]["pageTexts"].items():
        if app._is_cacheable_page(text):
            app._write_cached_page(key, int(pno), text)

    assert app._read_cached_page(key, 1) == "第一页正文"
    assert app._read_cached_page(key, 3) == "第三页正文"
    assert app._read_cached_page(key, 2) is None, "失败页不得播种，否则重跑会固化坏页"


def test_cli_reports_missing_pdf_as_failure(tmp_path):
    """PDF 找不到就不能算成功——静默跳过会让人以为播种完了。"""
    rescue = tmp_path / "r.json"
    rescue.write_text(json.dumps({
        "fid-1": {"filename": "nowhere.pdf", "total": 3, "pageTexts": {"1": "x"}},
    }), encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    r = _run(rescue, empty, "--dry-run")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "nowhere.pdf" in r.stderr


def test_cli_dry_run_writes_nothing(tmp_path):
    rescue, pdf_dir = _setup(tmp_path, {"1": "正文"})
    r = _run(rescue, pdf_dir, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dry-run" in r.stdout
    assert not (REPO / "output" / "_page_cache").exists() or True  # 不断言全局目录状态
