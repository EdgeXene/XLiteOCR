"""Structured-document layer for XLiteOCR (PP-Structure, Apache-2.0).

Wraps PaddleOCR's PPStructure to produce, per page:
  - markdown   : a reading-order markdown rendering (text, never HTML)
  - blocks     : typed layout blocks [{type, bbox, ...}]
  - figures    : figure regions vectorized to SVG (engine/figures.py)

PPStructure region shape (confirmed on this build):
  {"type": "text|title|table|figure|list|...", "bbox": [x0,y0,x1,y1], "res": ...}
    - table -> res = {"html": "<table>...", "cell_bbox": [...]}
    - text/title/list -> res = [ {"text":..., "confidence":..., "text_region":...}, ... ]

Table output contract (changed 2026-09-07)
------------------------------------------
The model's table HTML is parsed here into typed `cells` (row, column, spans,
text, header flag) and is never forwarded as markup:

  - `markdown` carries a GFM pipe table built from those cells. It no longer
    embeds an HTML fragment, so a consumer never has to decide whether a line
    of Markdown is model markup or recognized text.
  - `html` is still emitted for compatibility, but it is regenerated from the
    typed cells with every cell text escaped. It is export data, not a
    fragment lifted from the model, and it is not meant to be inserted into a
    page: build a table from `cells` instead.
  - `cell_boxes` carries the upstream `cell_bbox` evidence unchanged, labeled
    with `cell_boxes_frame` because the coordinate frame of those boxes has
    not been verified against page pixels yet.

Recognized text and the Markdown export (changed 2026-09-07)
------------------------------------------------------------
Recognized text is document data. `lines[].text`, `full_text`, `blocks[].text`
and `cells[].text` therefore carry it verbatim: if a page contains the
characters "<p onclick=...>", that is what the document says and callers need
to see it.

`markdown` is different, because Markdown is a format that consumers hand to a
renderer, and most renderers pass raw HTML straight through. Recognized text
placed in `markdown` is therefore backslash-escaped over CommonMark's ASCII
punctuation set, so a consumer renders the characters the document contained
instead of interpreting them as markup or as structure. Only recognized text
is escaped; the structure XLiteOCR itself adds (heading markers, table pipes,
the figure placeholder) stays live. `markdown_unescape` reverses it exactly.

Every consumer must treat all recognized text as untrusted input.

Formula recognition is DISABLED by default: its LaTeX model is ~100 MB and slow,
at odds with "lightweight". Enable via XLITE_FORMULA=1 if needed.
"""

from __future__ import annotations

import html as html_lib
import os
import threading
import types
from functools import lru_cache
from html.parser import HTMLParser

import numpy as np
from PIL import Image

from . import figures as figures_mod

_LOCK = threading.Lock()
_FORMULA = os.environ.get("XLITE_FORMULA", "0") == "1"

# Upper bound on a single rowspan/colspan value. A malformed or hostile span
# cannot be allowed to size a grid.
_MAX_SPAN = 1000

# Upper bounds on one table. A pair of large spans would otherwise make the
# occupancy grid grow as their product, so a table past these limits is
# reported as unreadable rather than expanded. A 100 by 500 table, far larger
# than anything the layout model produces, stays well inside them.
_MAX_TABLE_CELLS = 20_000
_MAX_GRID_SLOTS = 200_000


class _TableTooLarge(Exception):
    """Raised internally when a table fragment exceeds the parser's bounds."""

# Label for the upstream cell-box evidence. The frame those boxes live in
# (page pixels or region-relative) is not verified yet; see cluster C3.
CELL_BOX_FRAME_UNVERIFIED = "upstream-unverified"


def _log_adapter(outcome: str) -> None:
    """Record what the geometry adapter did, so a silent skip is visible."""
    import logging

    logging.getLogger("xlite-ocr").info("table bbox adapter: %s", outcome)


@lru_cache(maxsize=2)
def _get_structure(lang: str = "en"):
    from paddleocr import PPStructure

    if os.name == "nt":
        # Same guard as ocr_engine._get_ocr: paddle 3.3.1 defaults oneDNN
        # ON for CPU inference and its win_amd64 oneDNN fused_conv2d
        # kernel is broken on the PP models; paddleocr only ever ENABLES
        # onednn, so force it off on every Config before predictors build.
        from paddle import inference as _pi
        if not getattr(_pi.Config, "_vx_onednn_off", False):
            class _Config(_pi.Config):
                _vx_onednn_off = True
                def __init__(self, *args):
                    super().__init__(*args)
                    (self.disable_onednn
                     if hasattr(self, "disable_onednn")
                     else self.disable_mkldnn)()
            _pi.Config = _Config

    with _LOCK:
        engine = PPStructure(
            lang=lang,
            show_log=False,
            use_gpu=False,
            # broken win_amd64 oneDNN kernels; Linux is fine both ways
            enable_mkldnn=(os.name != "nt"),
            cpu_threads=16,
            recovery=False,
            formula=_FORMULA,
        )
        # Correct the table cell geometry on this instance. See
        # corrected_table_bbox_decode: the shipped decoder scales both axes by
        # the crop's longer dimension, which puts every cell of a non-square
        # table in the wrong place.
        _log_adapter(install_table_bbox_adapter(engine))
        return engine


# ------------------------------------------------- table geometry adapter

# Where the table decoder sits on a constructed PPStructure, confirmed by
# walking a live engine: engine.table_system.table_structurer.postprocess_op.
_TABLE_DECODER_PATH = ("table_system", "table_structurer", "postprocess_op")


def corrected_table_bbox_decode(bbox, shape):
    """Scale a normalized table cell box by the crop's OWN width and height.

    The installed paddleocr TableLabelDecode._bbox_decode does this:

        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        h, w = pad_h, pad_w        # original dimensions replaced by the pad
        bbox[0::2] *= w
        bbox[1::2] *= h
        bbox[0::2] /= ratio_w
        bbox[1::2] /= ratio_h

    `shape` is built by ResizeTableImage as [height, width, ratio, ratio] and
    extended by PaddingTableImage with [pad_h, pad_w]. The resize ratio is
    UNIFORM (`max_len / max(height, width)`) and the pad target is a square of
    side `max_len`, so `pad / ratio` reduces to `max(height, width)` on BOTH
    axes. Every coordinate is therefore scaled by the crop's longer dimension,
    and on a non-square table the short axis is stretched past the edge of the
    crop it came from. Measured on the shipped fixture: a 451 x 136 crop
    produced cell_bbox y values reaching 425.

    The correct scaling, which is what upstream release/2.7 does for this
    decoder, is the original width for x and the original height for y. The
    ratio division is not needed because the model's coordinates are already
    normalized against the original crop.

    Verified on the shipped 3 x 3 fixture: all nine texts land in the right
    cells with this, and none of them do without it.
    """
    h, w = float(shape[0]), float(shape[1])
    bbox[0::2] *= w
    bbox[1::2] *= h
    return bbox


def install_table_bbox_adapter(engine) -> str:
    """Correct the table decoder on ONE engine instance. Returns what it did.

    Scoped three ways on purpose:
      * the instance, not the class, so nothing else in the process changes;
      * an EXACT type match, so TableMasterLabelDecode (a subclass whose
        geometry differs) and any decoder added later are left alone;
      * no dependency file is edited, so a reinstall cannot silently drop it.
    """
    node = engine
    for attr in _TABLE_DECODER_PATH:
        node = getattr(node, attr, None)
        if node is None:
            return f"skipped: no {'.'.join(_TABLE_DECODER_PATH)} on this engine"

    if type(node).__name__ != "TableLabelDecode":
        return f"skipped: decoder is {type(node).__name__}, not TableLabelDecode"
    if getattr(node, "_xliteocr_bbox_adapter", False):
        return "already installed"

    node._bbox_decode = types.MethodType(
        lambda self, bbox, shape: corrected_table_bbox_decode(bbox, shape), node
    )
    node._xliteocr_bbox_adapter = True
    return "installed"


def _region_text(res) -> str:
    """Join the OCR lines of a text/title/list region into one string."""
    if not isinstance(res, list):
        return ""
    parts = []
    for line in res:
        t = line.get("text") if isinstance(line, dict) else None
        if t:
            parts.append(t)
    return " ".join(parts)


# ---------------------------------------------------------------- table cells

def _span_value(raw) -> int:
    """Parse a rowspan/colspan attribute into a sane positive integer."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 1
    if value < 1:
        return 1
    return min(value, _MAX_SPAN)


class _TableCellExtractor(HTMLParser):
    """Turn a table HTML fragment into typed cells.

    Only tr/td/th structure and text are kept. Any other tag inside a cell is
    dropped and contributes nothing but the text it wraps, so markup that
    arrives inside the model's output cannot leave this parser as markup.
    `convert_charrefs` decodes the entities the upstream matcher writes, so
    `text` is the recognized string and the escaping happens again on output.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[dict] = []
        self._row = -1
        self._col = 0
        self._occupied: set[tuple[int, int]] = set()
        self._open: dict | None = None
        self._parts: list[str] = []

    # -- helpers

    def _close_cell(self) -> None:
        if self._open is None:
            return
        text = " ".join("".join(self._parts).split())
        self._open["text"] = text
        self.cells.append(self._open)
        self._open = None
        self._parts = []
        if len(self.cells) > _MAX_TABLE_CELLS:
            raise _TableTooLarge("too many cells")

    def _start_row(self) -> None:
        self._close_cell()
        self._row += 1
        self._col = 0

    def _start_cell(self, tag: str, attrs) -> None:
        self._close_cell()
        if self._row < 0:  # cells before any <tr>
            self._row = 0
        attr = {k.lower(): v for k, v in attrs}
        row_span = _span_value(attr.get("rowspan"))
        col_span = _span_value(attr.get("colspan"))
        while (self._row, self._col) in self._occupied:
            self._col += 1
        if len(self._occupied) + row_span * col_span > _MAX_GRID_SLOTS:
            raise _TableTooLarge("table grid too large")
        for r in range(self._row, self._row + row_span):
            for c in range(self._col, self._col + col_span):
                self._occupied.add((r, c))
        self._open = {
            "row": self._row,
            "col": self._col,
            "row_span": row_span,
            "col_span": col_span,
            "header": tag == "th",
            "text": "",
        }
        self._col += col_span

    # -- HTMLParser interface

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._start_row()
        elif tag in ("td", "th"):
            self._start_cell(tag, attrs)
        elif self._open is not None and tag in ("br", "p", "div", "li"):
            self._parts.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th", "tr"):
            self._close_cell()

    def handle_data(self, data):
        if self._open is not None:
            self._parts.append(data)

    def close(self):
        super().close()
        self._close_cell()


def cells_from_table_html(raw_html: str) -> tuple[list[dict], bool]:
    """Parse model table HTML into typed cells.

    Returns (cells, parsed). `parsed` is False when nothing usable came back,
    which the caller reports as a quality warning instead of guessing.
    """
    if not raw_html or not isinstance(raw_html, str):
        return [], False
    parser = _TableCellExtractor()
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception:
        return [], False
    cells = sorted(parser.cells, key=lambda c: (c["row"], c["col"]))
    return cells, bool(cells)


def cells_to_html(cells) -> str:
    """Rebuild a table fragment from typed cells, escaping every cell text.

    Emitted for compatibility with callers that stored the old `html` field.
    Everything here is generated locally from `cells`; nothing is copied from
    the model's markup.
    """
    if not cells:
        return ""
    rows: dict[int, list[dict]] = {}
    for cell in cells:
        rows.setdefault(cell["row"], []).append(cell)
    out = ["<table>"]
    for row_index in sorted(rows):
        out.append("<tr>")
        for cell in sorted(rows[row_index], key=lambda c: c["col"]):
            tag = "th" if cell.get("header") else "td"
            row_span = _span_value(cell.get("row_span", 1))
            col_span = _span_value(cell.get("col_span", 1))
            attrs = ""
            if row_span > 1:
                attrs += ' rowspan="%d"' % row_span
            if col_span > 1:
                attrs += ' colspan="%d"' % col_span
            text = html_lib.escape(str(cell.get("text", "")), quote=True)
            out.append(f"<{tag}{attrs}>{text}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


# CommonMark's escapable ASCII punctuation. Using the spec's own set rather
# than a hand-picked shortlist means no per-character reasoning about which
# characters can start a construct in which dialect, and it makes the inverse
# a single unambiguous rule.
_MARKDOWN_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def markdown_escape(text) -> str:
    """Backslash-escape recognized text for the Markdown export.

    A CommonMark renderer turns the result back into exactly the characters
    that were recognized, and an HTML-enabled renderer does not see markup:
    `<p>` becomes `\\<p\\>`. The backslash itself is in the set, so the
    encoding is unambiguous and `markdown_unescape` inverts it exactly.
    """
    out: list[str] = []
    for ch in str(text if text is not None else ""):
        if ch in _MARKDOWN_PUNCTUATION:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def markdown_unescape(text) -> str:
    """Inverse of markdown_escape. The demo renderers do the same thing."""
    out: list[str] = []
    source = str(text if text is not None else "")
    i = 0
    while i < len(source):
        ch = source[i]
        if ch == "\\" and i + 1 < len(source) and source[i + 1] in _MARKDOWN_PUNCTUATION:
            out.append(source[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _markdown_text(text) -> str:
    """Recognized text as it goes into the Markdown export.

    Whitespace is collapsed first so a newline inside a region cannot open a
    new Markdown block, then every escapable character is escaped. Pipes are
    covered by the punctuation set, so a cell cannot break its row either.
    """
    return markdown_escape(" ".join(str(text if text is not None else "").split()))


def cells_to_markdown(cells) -> str:
    """Render typed cells as a GFM pipe table.

    GFM tables need a header row, so row 0 is used as the header. The `header`
    flag on each cell keeps the model's own answer. Spanned cells put their
    text in the origin cell and leave the covered cells empty, because pipe
    tables cannot express a span.
    """
    if not cells:
        return ""
    n_rows = max(c["row"] + _span_value(c.get("row_span", 1)) for c in cells)
    n_cols = max(c["col"] + _span_value(c.get("col_span", 1)) for c in cells)
    if n_rows < 1 or n_cols < 1:
        return ""
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for cell in cells:
        row, col = cell["row"], cell["col"]
        if 0 <= row < n_rows and 0 <= col < n_cols:
            grid[row][col] = _markdown_text(cell.get("text", ""))
    lines = [
        "| " + " | ".join(grid[0]) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _coerce_boxes(raw) -> list[list[float]]:
    """Copy upstream cell boxes into plain JSON-serializable floats."""
    boxes: list[list[float]] = []
    for box in raw or []:
        try:
            boxes.append([float(v) for v in box])
        except (TypeError, ValueError):
            continue
    return boxes


# ------------------------------------------------------------------- assembly

def _block_to_markdown(block: dict) -> str:
    """Render one block. The markers are ours; the text is escaped data."""
    t = block.get("type", "text")
    if t == "title":
        heading = _markdown_text(block.get("text", ""))
        return f"# {heading}" if heading else ""
    if t == "table":
        return cells_to_markdown(block.get("cells") or [])
    if t == "figure":
        return "![figure](figure)"
    if t in ("header", "footer"):
        return ""
    return _markdown_text(block.get("text", ""))


def _bbox_of(region: dict) -> list[float]:
    try:
        return [float(v) for v in region.get("bbox", [0, 0, 0, 0])]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0, 0.0]


def blocks_from_regions(
    regions,
    image: Image.Image | None = None,
    with_figures: bool = True,
) -> dict:
    """Convert PPStructure regions into the XLiteOCR page payload.

    Split out of parse() so the serialization contract can be tested without
    loading a model. `image` is only needed for figure vectorization.
    """
    # Sort top-to-bottom, then left-to-right for stable reading order.
    ordered = sorted(regions, key=lambda r: (_bbox_of(r)[1], _bbox_of(r)[0]))

    blocks: list[dict] = []
    fig_out: list[dict] = []
    for r in ordered:
        rtype = r.get("type", "text")
        res = r.get("res")
        block: dict = {"type": rtype, "bbox": _bbox_of(r)}

        if rtype == "table" and isinstance(res, dict):
            raw_html = res.get("html") or ""
            cells, parsed = cells_from_table_html(raw_html)
            block["cells"] = cells
            block["html"] = cells_to_html(cells)
            boxes = _coerce_boxes(res.get("cell_bbox"))
            if boxes:
                block["cell_boxes"] = boxes
                block["cell_boxes_frame"] = CELL_BOX_FRAME_UNVERIFIED
            if raw_html and not parsed:
                block["warnings"] = [
                    "table structure could not be read from the model output"
                ]
        elif rtype == "figure":
            if with_figures and image is not None:
                try:
                    svg = figures_mod.figure_to_svg(image, block["bbox"])
                except Exception as e:  # vectorization is best-effort
                    svg = ""
                    block["svg_error"] = str(e)
                block["svg"] = svg
                fig_out.append({"box": block["bbox"], "type": "figure", "svg": svg})
        else:
            block["text"] = _region_text(res)

        blocks.append(block)

    markdown = "\n\n".join(
        md for md in (_block_to_markdown(b) for b in blocks) if md
    )
    return {"markdown": markdown, "blocks": blocks, "figures": fig_out}


def parse(image: Image.Image, lang: str = "en", with_figures: bool = True) -> dict:
    """Return {"markdown": str, "blocks": [...], "figures": [...]} for one page."""
    eng = _get_structure(lang)
    arr = np.asarray(image.convert("RGB"))
    with _LOCK:
        regions = eng(arr)
    return blocks_from_regions(regions, image=image, with_figures=with_figures)
