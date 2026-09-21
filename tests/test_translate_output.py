"""译文输出的前导空白抑制。

运行: cd /Users/xuzheng/CodingProjects/Doxify && conda run -n mineru python -m pytest tests/test_translate_output.py -v

背景（2026-08-12）：译文第一行前总是多一个空格。实测取证——用真实网关翻译
"Hi Sheng Tao —"，模型吐出的第一个 token 就是 ' 嗨'（带前导空格）。空格来自模型
的分词器（SentencePiece 类，词首空格属于 token 本身），不是我们加的，前端拼接
首块时也没有任何前缀。

但不能无条件 lstrip：Markdown 的缩进有语义（4 空格代码块、嵌套列表项），而
split_markdown_chunks 用 "\\n".join(lines) 拼块，首行若带缩进，块本身就以空白开头。
所以抑制只在**源块自身不以空白开头**时启用。
"""
import app


def test_strips_leading_space_from_first_token():
    out, trimming = app._trim_leading_ws(" 嗨", True)
    assert out == "嗨"
    assert trimming is False, "出现非空白后就该停止抑制"


def test_passes_later_tokens_through_untouched():
    """抑制结束后必须原样透传，包括 token 内部和末尾的空格。"""
    out, trimming = app._trim_leading_ws(" 世界 ", False)
    assert out == " 世界 "
    assert trimming is False


def test_swallows_all_whitespace_token_and_keeps_trimming():
    """整块空白的 token 全吞掉，且抑制状态保持——下一个 token 可能还带前导空白。"""
    out, trimming = app._trim_leading_ws("   ", True)
    assert out == ""
    assert trimming is True


def test_strips_leading_newline_too():
    """换行与空格同属一类产物，一并抑制。"""
    out, trimming = app._trim_leading_ws("\n\n盛涛", True)
    assert out == "盛涛"
    assert trimming is False


def test_disabled_trimming_preserves_indentation():
    """源块本身带缩进时不得改动——4 空格是代码块，动了就破坏 Markdown 语义。"""
    out, trimming = app._trim_leading_ws("    code line", False)
    assert out == "    code line"
    assert trimming is False


def test_only_leading_whitespace_is_removed_not_inner():
    out, trimming = app._trim_leading_ws("  a  b", True)
    assert out == "a  b", "只去前导，内部空格必须保留"
    assert trimming is False


def test_empty_token_does_not_end_trimming():
    out, trimming = app._trim_leading_ws("", True)
    assert out == ""
    assert trimming is True


# ---------------------------------------------------------------- 启用条件


def test_trimming_enabled_only_when_source_has_no_leading_space():
    """决定是否启用抑制的判据：源块自身是否以空白开头。"""
    assert app._should_trim_leading_ws("Hi Sheng Tao —") is True
    assert app._should_trim_leading_ws("# 标题") is True
    assert app._should_trim_leading_ws("    code line") is False, "4 空格缩进代码块"
    assert app._should_trim_leading_ws("  - 嵌套列表项") is False
    assert app._should_trim_leading_ws("\n开头就是换行") is False
    assert app._should_trim_leading_ws("") is False, "空块无所谓，别启用"


# ---------------------------------------------------------------- 串起来


def test_stream_sequence_removes_exactly_one_leading_space():
    """模拟实测到的真实 token 序列：' 嗨' '，' '盛' '涛' ' —'。

    只有最前面那个空格该消失；' —' 里的空格是句中空格，必须原样保留。
    """
    tokens = [" 嗨", "，", "盛", "涛", " —"]
    trimming = app._should_trim_leading_ws("Hi Sheng Tao —")
    out = []
    for t in tokens:
        piece, trimming = app._trim_leading_ws(t, trimming)
        out.append(piece)
    assert "".join(out) == "嗨，盛涛 —"


def test_indented_source_keeps_its_leading_space():
    tokens = [" 代码", "行"]
    trimming = app._should_trim_leading_ws("    code line")
    out = []
    for t in tokens:
        piece, trimming = app._trim_leading_ws(t, trimming)
        out.append(piece)
    assert "".join(out) == " 代码行", "源块有缩进时不得改动译文前导空白"
