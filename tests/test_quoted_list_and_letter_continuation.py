"""被误标成引用块的列表，以及跨页断开的字母序号列表。

运行: conda run -n mineru python -m pytest tests/test_quoted_list_and_letter_continuation.py -v

背景（2026-08-21）：VLM 把相对正文缩进的列表当成了引用块，输出 "> a. steel;"。
实测该文档 3 个引用块里 2 个其实是列表（每行都是条目），只有 1 个是真引用。
更糟的是它逐页不一致：第 60 页整张列表带 >，第 61 页续写的 "h. other" 不带，
于是列表在页边界断成两块。

两条修正：
  1. 每一行都是列表条目的引用块 → 去掉 > 前缀还原为列表；真引用不满足条件，不受影响。
  2. 字母序号连续的相邻块（g. → h.）接成一块。

第 2 条**只认字母序号**。曾用「编号连续」做判据，结果把全文所有编号段落（1. 2. …
314.）全都算成了断裂列表——那些本来就该各自成段。限定字母后全文只剩 1 处命中，
正是真正断裂的那个。
"""
import app


# ------------------------------------------------ 引用块还原为列表


def test_quoted_list_is_unquoted():
    md = "> a. steel;\n> b. hydraulics;\n> c. electrical parts;"
    assert app._unquote_list_blocks(md) == "a. steel;\nb. hydraulics;\nc. electrical parts;"


def test_real_blockquote_is_preserved():
    md = "> *Machines designed for the lifting of people, equipment and/or materials.*"
    assert app._unquote_list_blocks(md) == md


def test_mixed_blockquote_is_preserved():
    """只要有一行不是条目，整块就当真引用处理。"""
    md = "> The regulation states:\n> a. first;\n> b. second;"
    assert app._unquote_list_blocks(md) == md


def test_quoted_bullet_and_numbered_lists():
    assert app._unquote_list_blocks("> - one\n> - two") == "- one\n- two"
    assert app._unquote_list_blocks("> 1. one\n> 2. two") == "1. one\n2. two"


def test_other_blocks_untouched():
    md = "普通段落。\n\n| A | B |\n|:---|:---|\n| 1 | 2 |"
    assert app._unquote_list_blocks(md) == md


# ------------------------------------------------ 字母序号跨页接合


def test_letter_sequence_across_blocks_is_joined():
    md = "a. steel;\nf. plastics;\ng. traction; and\n\nh. other (includes anything else).\n"
    out = app._merge_broken_paragraphs(md)
    assert "g. traction; and\nh. other (includes anything else)." in out, out


def test_non_consecutive_letters_are_not_joined():
    md = "a. steel;\nb. hydraulics;\n\nd. engines;\n"
    out = app._merge_broken_paragraphs(md)
    assert "b. hydraulics;\n\nd. engines;" in out, out


def test_numbered_paragraphs_are_never_joined():
    """全文编号段落 1. 2. 3. 各自成段，绝不能因「编号连续」被并起来。"""
    md = ("313. The TRA identified the following categories.\n\n"
          "314. The TRA consulted on these categories.\n\n"
          "315. The TRA asked for data accordingly.\n")
    out = app._merge_broken_paragraphs(md)
    assert out.count("\n\n") == 2, out


def test_new_list_starting_at_a_is_not_joined():
    md = "x. last item of previous list\n\na. first item of a new list\n"
    out = app._merge_broken_paragraphs(md)
    assert "previous list\n\na. first item" in out, out
