"""Typed table cells and escaped Markdown -- the F1 output contract.

These tests run without a model. `blocks_from_regions` takes PPStructure-shaped
region dicts, so the serialization contract can be exercised directly.

What they pin down:
  - table HTML from the model becomes typed cells with row, column and spans
  - cell text is text: markup that arrives inside a cell never leaves as markup
  - `markdown` carries a pipe table, never an HTML fragment
  - `html` is regenerated from the cells with every text escaped
  - upstream cell-box evidence survives, labeled with its unverified frame
  - recognized text is passed through verbatim, because it is document data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import structure  # noqa: E402

SPAN_TABLE = (
    "<table><tbody>"
    '<tr><td rowspan="2">Alpha</td><td>Beta</td></tr>'
    "<tr><td>Gamma</td></tr>"
    "</tbody></table>"
)


def _table_region(html: str, cell_bbox=None, bbox=(0.0, 0.0, 100.0, 40.0)) -> dict:
    res: dict = {"html": html}
    if cell_bbox is not None:
        res["cell_bbox"] = cell_bbox
    return {"type": "table", "bbox": list(bbox), "res": res}


def _text_region(text: str, rtype: str = "text", bbox=(0.0, 0.0, 100.0, 20.0)) -> dict:
    return {
        "type": rtype,
        "bbox": list(bbox),
        "res": [{"text": text, "confidence": 0.99, "text_region": []}],
    }


def _table_block(html: str, **kwargs) -> dict:
    page = structure.blocks_from_regions([_table_region(html, **kwargs)])
    return page["blocks"][0]


# ---------------------------------------------------------------- typed cells

def test_simple_table_yields_typed_cells_with_positions():
    cells = _table_block(
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
    )["cells"]
    assert [(c["row"], c["col"], c["text"]) for c in cells] == [
        (0, 0, "A"), (0, 1, "B"), (1, 0, "C"), (1, 1, "D"),
    ]
    assert all(c["row_span"] == 1 and c["col_span"] == 1 for c in cells)


def test_row_span_pushes_the_next_row_cell_into_the_free_column():
    """The shipped fixture's shape: a rowspan must not collapse row two."""
    cells = _table_block(SPAN_TABLE)["cells"]
    by_text = {c["text"]: c for c in cells}
    assert by_text["Alpha"]["row"] == 0 and by_text["Alpha"]["col"] == 0
    assert by_text["Alpha"]["row_span"] == 2
    assert by_text["Beta"]["row"] == 0 and by_text["Beta"]["col"] == 1
    # Gamma is on row 1. Column 0 is covered by Alpha's span, so it lands in 1.
    assert by_text["Gamma"]["row"] == 1 and by_text["Gamma"]["col"] == 1


def test_column_span_advances_the_next_cell():
    cells = _table_block(
        '<table><tr><td colspan="2">Wide</td><td>Edge</td></tr></table>'
    )["cells"]
    assert cells[0]["col"] == 0 and cells[0]["col_span"] == 2
    assert cells[1]["col"] == 2


def test_header_cells_are_marked():
    cells = _table_block(
        "<table><thead><tr><th>Item</th></tr></thead>"
        "<tbody><tr><td>Widget</td></tr></tbody></table>"
    )["cells"]
    assert cells[0]["header"] is True
    assert cells[1]["header"] is False


def test_absurd_span_values_are_clamped_and_never_crash():
    cells = _table_block(
        '<table><tr><td rowspan="999999" colspan="-4">X</td></tr></table>'
    )["cells"]
    assert cells[0]["row_span"] == structure._MAX_SPAN
    assert cells[0]["col_span"] == 1


def test_a_span_pair_that_would_size_a_huge_grid_is_refused():
    """Two large spans multiply. Report the table as unreadable instead."""
    block = _table_block(
        '<table><tr><td rowspan="1000" colspan="1000">X</td></tr></table>'
    )
    assert block["cells"] == []
    assert block["html"] == ""
    assert block["warnings"]


# ------------------------------------------------------- text stays text

def test_escaped_cell_text_is_decoded_then_escaped_again_on_export():
    """The upstream matcher escapes cell text; we recover it and re-escape."""
    block = _table_block(
        "<table><tr><td>&lt;img src=x onerror=alert(1)&gt;</td></tr></table>"
    )
    assert block["cells"][0]["text"] == "<img src=x onerror=alert(1)>"
    assert "&lt;img src=x onerror=alert(1)&gt;" in block["html"]
    assert "<img" not in block["html"]


def test_markup_inside_a_cell_never_leaves_as_markup():
    block = _table_block(
        "<table><tr><td><img src=x onerror=alert(1)>Beta</td></tr></table>"
    )
    assert block["cells"][0]["text"] == "Beta"
    assert "<img" not in block["html"]
    assert "onerror" not in block["html"]


def test_script_content_inside_a_cell_becomes_plain_text():
    block = _table_block(
        "<table><tr><td><script>alert(1)</script></td></tr></table>"
    )
    assert block["cells"][0]["text"] == "alert(1)"
    assert "<script" not in block["html"]


def test_attribute_injection_in_cell_text_is_escaped_on_export():
    block = _table_block(
        '<table><tr><td>&quot;&gt;&lt;svg onload=alert(1)&gt;</td></tr></table>'
    )
    assert block["cells"][0]["text"] == '"><svg onload=alert(1)>'
    assert "<svg" not in block["html"]
    assert "&quot;&gt;&lt;svg onload=alert(1)&gt;" in block["html"]


# ----------------------------------------------------------------- markdown
#
# The Markdown export is the one place where recognized text is not verbatim.
# Markdown is handed to renderers that pass raw HTML straight through, so
# recognized text is backslash-escaped there and only there. The structure
# XLiteOCR adds itself (heading markers, table pipes) stays live.

def _unescaped_angle_brackets(md: str) -> list[int]:
    """Positions of < or > that a Markdown renderer would read as markup."""
    found = []
    i = 0
    while i < len(md):
        if md[i] == "\\" and i + 1 < len(md):
            i += 2
            continue
        if md[i] in "<>":
            found.append(i)
        i += 1
    return found


def test_markdown_renders_a_pipe_table():
    page = structure.blocks_from_regions([_table_region(SPAN_TABLE)])
    assert page["markdown"] == (
        "| Alpha | Beta |\n"
        "| --- | --- |\n"
        "|  | Gamma |"
    )


def test_markdown_escape_covers_html_backslashes_and_punctuation():
    assert structure.markdown_escape("<p>") == "\\<p\\>"
    # Every ASCII punctuation character is escaped, ":" included, so there is
    # no per-character judgement about which ones matter in which dialect.
    assert structure.markdown_escape("C:\\path") == "C\\:\\\\path"
    assert structure.markdown_escape("a|b") == "a\\|b"
    assert structure.markdown_escape("plain text 42") == "plain text 42"


def test_markdown_unescape_is_the_exact_inverse():
    """The demo renderers run the same inverse, so this is their contract."""
    for original in [
        "<p onclick=alert(1)>CLICK</p>",
        "C:\\Users\\file.txt",
        "**not bold** and `not code`",
        "| a | b |",
        "## not a heading",
        "Total $1,234.56 due 09/30/2026",
        "backslash at end \\",
        "",
    ]:
        escaped = structure.markdown_escape(original)
        assert structure.markdown_unescape(escaped) == original


def test_markdown_never_carries_raw_html():
    payload = "<img src=x onerror=alert(1)>"
    page = structure.blocks_from_regions([
        _table_region(
            "<table><tr><td>&lt;img src=x onerror=alert(1)&gt;</td></tr></table>"
        ),
        _text_region(payload, bbox=(0, 100, 100, 120)),
    ])
    md = page["markdown"]
    assert "<table" not in md and "<td" not in md
    # Nothing an HTML-enabled Markdown renderer would treat as a tag.
    assert _unescaped_angle_brackets(md) == []
    # The recognized characters are still recoverable, exactly.
    assert payload in structure.markdown_unescape(md)


def test_pipes_in_cell_text_do_not_break_the_row():
    page = structure.blocks_from_regions([
        _table_region("<table><tr><td>a|b</td><td>c</td></tr></table>")
    ])
    assert page["markdown"].splitlines()[0] == "| a\\|b | c |"


def test_table_cell_text_is_escaped_inside_our_own_row_structure():
    page = structure.blocks_from_regions([
        _table_region(
            "<table><tr><td>&lt;img src=x&gt;</td><td>a|b</td></tr></table>"
        )
    ])
    assert page["markdown"].splitlines()[0] == "| \\<img src\\=x\\> | a\\|b |"


def test_newlines_in_cell_text_are_collapsed():
    cells = _table_block("<table><tr><td>one\n  two</td></tr></table>")["cells"]
    assert cells[0]["text"] == "one two"


def test_html_export_is_unaffected_by_markdown_escaping():
    block = _table_block("<table><tr><td>a|b\\c</td></tr></table>")
    assert block["cells"][0]["text"] == "a|b\\c"
    assert "<td>a|b\\c</td>" in block["html"]


def test_recognized_text_is_verbatim_in_fields_and_escaped_in_markdown():
    """A page that says "<p onclick=..." says that, in the text fields."""
    payload = "<p onclick=alert(1)>CLICK</p>"
    page = structure.blocks_from_regions([_text_region(payload)])
    assert page["blocks"][0]["text"] == payload
    assert payload not in page["markdown"]
    assert page["markdown"] == structure.markdown_escape(payload)
    assert structure.markdown_unescape(page["markdown"]) == payload


def test_recognized_text_cannot_forge_markdown_structure():
    page = structure.blocks_from_regions([
        _text_region("## Fake heading", bbox=(0, 0, 10, 10)),
        _text_region("- fake list item", bbox=(0, 20, 10, 30)),
        _text_region("| fake | row |", bbox=(0, 40, 10, 50)),
    ])
    lines = page["markdown"].splitlines()
    assert lines[0].startswith("\\#\\#")
    assert lines[2].startswith("\\-")
    assert lines[4].startswith("\\|")


def _unescaped_count(text, char):
    """Occurrences of `char` that are not preceded by a backslash escape."""
    n = i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            i += 2
            continue
        if text[i] == char:
            n += 1
        i += 1
    return n


def test_recognized_text_cannot_forge_inline_markdown():
    """Inline delimiters are escaped as well as block markers.

    This is the contract the demo renderers depend on. They match `code` and
    **bold** while walking the string, so they have to read the escape pair
    first; a renderer that runs a plain pattern over the raw string turns a
    document's own backticks into an inline-code element with stray
    backslashes around it. Escaping only the block markers would leave that
    door open, so the inline delimiters are asserted here too.
    """
    page = structure.blocks_from_regions([
        _text_region("Run `ls -la` in the shell", bbox=(0, 0, 10, 10)),
        _text_region("**not bold** and _not italic_", bbox=(0, 20, 10, 30)),
        _text_region("C:\\path\\`cmd`", bbox=(0, 40, 10, 50)),
    ])
    md = page["markdown"]
    assert "\\`" in md
    assert "\\*\\*" in md
    # Nothing is left for an inline matcher to find.
    assert _unescaped_count(md, "`") == 0
    assert _unescaped_count(md, "*") == 0
    assert _unescaped_count(md, "_") == 0
    lines = [ln for ln in md.splitlines() if ln]
    assert structure.markdown_unescape(lines[0]) == "Run `ls -la` in the shell"
    assert structure.markdown_unescape(lines[1]) == "**not bold** and _not italic_"
    assert structure.markdown_unescape(lines[2]) == "C:\\path\\`cmd`"


def test_a_backslash_before_a_delimiter_survives_the_round_trip():
    """The document's own backslash and the escape backslash must not merge.

    `C:\\` followed by a backtick is the case that separates a correct inverse
    from one that drops a character: the backslash is escaped to two, and the
    backtick to backslash-backtick, so the reader gets both back.
    """
    for original in [
        "\\`",
        "`",
        "\\\\`code`",
        "escaped already: \\* \\` \\\\",
        "``",
    ]:
        escaped = structure.markdown_escape(original)
        assert _unescaped_count(escaped, "`") == 0
        assert structure.markdown_unescape(escaped) == original


def test_title_marker_stays_live_while_its_text_is_escaped():
    page = structure.blocks_from_regions([
        _text_region("# <b>Title</b>", rtype="title", bbox=(0, 0, 10, 10)),
    ])
    assert page["markdown"] == "# \\# \\<b\\>Title\\<\\/b\\>"
    assert structure.markdown_unescape(page["markdown"][2:]) == "# <b>Title</b>"


def test_empty_title_emits_no_stray_heading():
    page = structure.blocks_from_regions([
        {"type": "title", "bbox": [0, 0, 10, 10], "res": []},
    ])
    assert page["markdown"] == ""


def test_title_and_figure_markdown_unchanged():
    page = structure.blocks_from_regions([
        _text_region("Second paragraph", rtype="title", bbox=(0, 0, 10, 10)),
        _text_region("body", bbox=(0, 20, 10, 30)),
    ])
    assert page["markdown"] == "# Second paragraph\n\nbody"


# ------------------------------------------------------------ evidence, order

def test_upstream_cell_boxes_are_preserved_with_a_labeled_frame():
    block = _table_block(
        "<table><tr><td>A</td></tr></table>",
        cell_bbox=[[1, 2, 3, 4], [5, 6, 7, 8]],
    )
    assert block["cell_boxes"] == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert block["cell_boxes_frame"] == structure.CELL_BOX_FRAME_UNVERIFIED


def test_missing_cell_boxes_add_no_fields():
    block = _table_block("<table><tr><td>A</td></tr></table>")
    assert "cell_boxes" not in block
    assert "cell_boxes_frame" not in block


def test_unreadable_table_warns_instead_of_guessing():
    block = _table_block("<table>not a table at all")
    assert block["cells"] == []
    assert block["html"] == ""
    assert block["warnings"] == [
        "table structure could not be read from the model output"
    ]


def test_block_shape_and_reading_order_are_unchanged():
    """Ordering stays top-to-bottom then left-to-right in this cluster."""
    page = structure.blocks_from_regions([
        _text_region("LEFT BOTTOM", bbox=(0, 40, 100, 60)),
        _text_region("RIGHT TOP", bbox=(200, 0, 300, 20)),
        _text_region("LEFT TOP", bbox=(0, 0, 100, 20)),
    ])
    assert [b["text"] for b in page["blocks"]] == [
        "LEFT TOP", "RIGHT TOP", "LEFT BOTTOM",
    ]
    assert page["blocks"][0]["bbox"] == [0.0, 0.0, 100.0, 20.0]
    assert page["blocks"][0]["type"] == "text"


def test_parse_delegates_to_the_serializer_without_loading_a_model(monkeypatch):
    """parse() is a thin wrapper; cover its wiring with a stub engine."""
    from PIL import Image

    captured = {}

    def fake_engine(arr):
        captured["shape"] = arr.shape
        return [_text_region("hello")]

    monkeypatch.setattr(structure, "_get_structure", lambda lang="en": fake_engine)
    page = structure.parse(Image.new("RGB", (4, 3), "white"))
    assert captured["shape"] == (3, 4, 3)
    assert page["markdown"] == "hello"
    assert page["blocks"][0]["text"] == "hello"
    assert page["figures"] == []


def test_figures_are_skipped_without_an_image():
    page = structure.blocks_from_regions([
        {"type": "figure", "bbox": [0, 0, 10, 10], "res": None}
    ])
    assert page["figures"] == []
    assert page["blocks"][0]["type"] == "figure"
    assert "svg" not in page["blocks"][0]
