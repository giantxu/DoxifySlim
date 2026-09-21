#!/usr/bin/env python
"""把浏览器抢救出来的页结果播种进 Doxify 的页级缓存，供断点续跑复用。

用法:
    python seed_page_cache.py <rescue.json> <PDF 所在目录> [--dry-run]

背景: 2026-08-06 一批 5931 页的作业跑到 48.7% 时因外出中断。当时的版本不逐页落盘，
结果只存在于浏览器标签页的 JS 内存里（fileState[fid].pageTexts），靠一段控制台脚本
导出成 JSON。缓存键是 PDF 内容的 sha256，所以播种必须拿到 PDF 原件重新算哈希——
JSON 里只有文件名，按文件名去目录里找对应的 PDF。

识别失败的占位符会被跳过（复用 app._is_cacheable_page），那些页重跑时会重新识别。
"""
import argparse
import json
import sys
from pathlib import Path

import app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rescue_json", type=Path)
    ap.add_argument("pdf_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写入")
    args = ap.parse_args()

    data = json.loads(args.rescue_json.read_text(encoding="utf-8"))
    pdfs = {p.name: p for p in args.pdf_dir.rglob("*.pdf")}

    total_seeded = total_skipped = 0
    missing: list[str] = []

    for fid, entry in data.items():
        pages = entry.get("pageTexts") or {}
        if not pages:
            continue
        name = entry.get("filename", "")
        pdf = pdfs.get(name)
        if pdf is None:
            missing.append(name)
            continue

        key = app._page_cache_key(pdf.read_bytes(), app.PDF_DPI, app.ACTUAL_MODEL_NAME)
        seeded = skipped = 0
        for pno, text in pages.items():
            if not app._is_cacheable_page(text):
                skipped += 1
                continue
            if not args.dry_run:
                app._write_cached_page(key, int(pno), text)
            seeded += 1

        total_seeded += seeded
        total_skipped += skipped
        print(f"{name:<50} 播种 {seeded:>4} 页, 跳过失败页 {skipped:>3}  key={key[:12]}")

    print(f"\n合计播种 {total_seeded} 页，跳过失败页 {total_skipped}"
          + ("  (dry-run，未写入)" if args.dry_run else ""))
    if missing:
        print(f"\n以下文件在 {args.pdf_dir} 下没找到同名 PDF，未能播种：", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
