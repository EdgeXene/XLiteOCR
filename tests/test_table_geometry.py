"""Table cell geometry: the decoder scales each axis by its own dimension.

The shipped paddleocr TableLabelDecode scales BOTH axes by the crop's longer
dimension, so on any non-square table the short axis is stretched past the edge
of the crop the cells came from. These tests are written against the shape
contract rather than against the one fixture where it was noticed, so a wide,
tall or square table is each covered on its own terms.

Shape contract, read from the installed preprocessing:
  ResizeTableImage  -> shape = [height, width, ratio, ratio]
                       ratio = max_len / max(height, width)   (uniform)
  PaddingTableImage -> shape.extend([pad_h, pad_w])           (square, max_len)
"""

from __future__ import annotations

import numpy as np
import pytest

from engine import structure

MAX_LEN = 488


def shape_for(height: int, width: int, max_len: int = MAX_LEN):
    """The shape vector the real preprocessing produces for a crop."""
    ratio = max_len / (max(height, width) * 1.0)
    return np.array([height, width, ratio, ratio, max_len, max_len], dtype=float)


def shipped_decode(bbox, shape):
    """The installed implementation, copied verbatim, as the baseline."""
    bbox = np.array(bbox, dtype=float)
    h, w, ratio_h, ratio_w, pad_h, pad_w = shape
    h, w = pad_h, pad_w
    bbox[0::2] *= w
    bbox[1::2] *= h
    bbox[0::2] /= ratio_w
    bbox[1::2] /= ratio_h
    return bbox


def corrected(bbox, shape):
    return structure.corrected_table_bbox_decode(np.array(bbox, dtype=float), shape)


# A cell box in the model's normalized space: x0, y0, x1, y1 in [0, 1].
CELL = [0.10, 0.20, 0.90, 0.80]


# --------------------------------------------------------------- the invariant

@pytest.mark.parametrize("height,width", [
    (136, 451),    # the shipped fixture: wide
    (100, 400),    # wide
    (400, 100),    # tall
    (300, 300),    # square
    (17, 1000),    # extremely wide
    (1000, 17),    # extremely tall
    (1, 1),        # degenerate
])
def test_decoded_cells_stay_inside_the_crop(height, width):
    """A cell of a table cannot lie outside the table it came from."""
    shape = shape_for(height, width)
    x0, y0, x1, y1 = corrected(CELL, shape)
    assert 0 <= x0 <= width and 0 <= x1 <= width, f"x outside 0..{width}"
    assert 0 <= y0 <= height and 0 <= y1 <= height, f"y outside 0..{height}"


@pytest.mark.parametrize("height,width", [(136, 451), (100, 400), (400, 100)])
def test_the_shipped_decoder_puts_cells_outside_the_crop(height, width):
    """The defect, demonstrated rather than asserted from memory.

    This is what the measured 451 x 136 fixture showed: y values reaching 425.
    """
    shape = shape_for(height, width)
    _, _, _, y1_bad = shipped_decode(CELL, shape)
    _, _, x1_bad, _ = shipped_decode(CELL, shape)
    long_side = max(height, width)
    short_side = min(height, width)
    # Both axes were scaled by the long side.
    assert max(x1_bad, y1_bad) > short_side, (
        "expected the short axis to overflow its own dimension"
    )
    assert np.isclose(max(x1_bad, y1_bad), 0.90 * long_side, rtol=1e-6)


def test_a_square_table_is_unaffected_by_the_correction():
    """Where the two dimensions agree, the bug is invisible and so is the fix.

    This is why the defect survived: square-ish tables decode correctly either
    way, so a single fixture can easily fail to show it.
    """
    shape = shape_for(300, 300)
    assert np.allclose(corrected(CELL, shape), shipped_decode(CELL, shape))


# ------------------------------------------------------------- axis behavior

def test_x_scales_by_width_and_y_scales_by_height():
    shape = shape_for(height=136, width=451)
    x0, y0, x1, y1 = corrected([0.0, 0.0, 1.0, 1.0], shape)
    assert (x0, y0) == (0.0, 0.0)
    assert np.isclose(x1, 451.0)
    assert np.isclose(y1, 136.0)


def test_the_full_cell_grid_of_a_wide_table_lands_in_distinct_rows():
    """A 3 x 3 grid on a wide crop must resolve to three separated rows.

    Under the shipped decoder the rows are stretched over the long dimension
    and collapse into the wrong cells, which is what produced nine texts in the
    wrong places on the shipped fixture.
    """
    height, width = 136, 451
    shape = shape_for(height, width)
    rows = []
    for r in range(3):
        y0, y1 = r / 3, (r + 1) / 3
        decoded = corrected([0.0, y0, 1.0, y1], shape)
        rows.append((decoded[1], decoded[3]))

    for y0, y1 in rows:
        assert 0 <= y0 <= height and 0 <= y1 <= height
    # Rows are ordered, disjoint and together cover the crop's height.
    assert rows[0][1] <= rows[1][0] + 1e-9
    assert rows[1][1] <= rows[2][0] + 1e-9
    assert np.isclose(rows[2][1], height)


def test_spanning_cells_keep_their_proportions():
    """A cell spanning two of three columns must cover two thirds of the width."""
    shape = shape_for(height=200, width=600)
    span = corrected([0.0, 0.0, 2 / 3, 1.0], shape)
    assert np.isclose(span[2], 400.0)
    assert np.isclose(span[3], 200.0)


def test_decoding_is_independent_of_the_padding_size():
    """The answer must not depend on the model's internal padding target."""
    a = corrected(CELL, shape_for(136, 451, max_len=488))
    b = corrected(CELL, shape_for(136, 451, max_len=640))
    assert np.allclose(a, b)


# ------------------------------------------------------------ adapter scoping

class _FakeTableLabelDecode:
    pass


_FakeTableLabelDecode.__name__ = "TableLabelDecode"


class TableMasterLabelDecode:
    """A stand-in for the subclass whose geometry is NOT the same."""


class _Structurer:
    def __init__(self, decoder):
        self.postprocess_op = decoder


class _Engine:
    def __init__(self, decoder):
        self.table_system = type("TS", (), {})()
        self.table_system.table_structurer = _Structurer(decoder)


def test_the_adapter_installs_on_the_decoder_it_targets():
    decoder = _FakeTableLabelDecode()
    engine = _Engine(decoder)
    assert structure.install_table_bbox_adapter(engine) == "installed"
    assert decoder._xliteocr_bbox_adapter is True
    # Bound to the instance, so the class is untouched and any other engine in
    # the process keeps the stock behavior.
    assert "_bbox_decode" not in vars(_FakeTableLabelDecode)


def test_the_adapter_refuses_a_decoder_it_was_not_verified_against():
    """TableMaster has different geometry; silently patching it would be a guess."""
    engine = _Engine(TableMasterLabelDecode())
    outcome = structure.install_table_bbox_adapter(engine)
    assert outcome.startswith("skipped")
    assert "TableMasterLabelDecode" in outcome


def test_the_adapter_reports_rather_than_failing_when_the_path_is_gone():
    """An upstream refactor must be visible, not silently skipped."""
    outcome = structure.install_table_bbox_adapter(object())
    assert outcome.startswith("skipped")


def test_the_adapter_is_idempotent():
    engine = _Engine(_FakeTableLabelDecode())
    assert structure.install_table_bbox_adapter(engine) == "installed"
    assert structure.install_table_bbox_adapter(engine) == "already installed"


def test_the_installed_adapter_actually_decodes_correctly():
    decoder = _FakeTableLabelDecode()
    engine = _Engine(decoder)
    structure.install_table_bbox_adapter(engine)
    shape = shape_for(136, 451)
    out = decoder._bbox_decode(np.array(CELL, dtype=float), shape)
    assert 0 <= out[1] <= 136 and 0 <= out[3] <= 136


# ------------------------------------------------- end to end, real model

def test_the_shipped_table_fixture_resolves_into_the_right_cells():
    """The measurement that started this, run against the real engine.

    The diagnostic recorded nine texts assigned to the wrong cells of a 3 x 3
    table, with cell_bbox y values reaching 425 inside a 136-pixel-tall crop.
    With the adapter installed every text lands in its own cell, and the labels
    in the fixture say where each one belongs.

    Loads the layout model, so it is slower than the rest of this file.
    """
    from pathlib import Path

    from PIL import Image

    sample = Path(__file__).resolve().parent.parent / "samples" / "structdoc.png"
    if not sample.exists():
        pytest.skip("structdoc.png fixture not present")

    page = structure.parse(Image.open(sample).convert("RGB"))
    tables = [b for b in page["blocks"] if b.get("type") == "table"]
    assert tables, "no table region found in the fixture"

    grid = {(c["row"], c["col"]): c["text"] for c in tables[0]["cells"]}
    assert len(grid) == 9, f"expected a 3 x 3 grid, got {len(grid)} cells"

    # Each fixture cell is labeled with its own coordinates, so a misplaced
    # cell is visible without hardcoding the geometry. OCR reads the digit
    # zero as the letter O in this font, so compare on that basis.
    for row in range(3):
        for col in range(3):
            text = grid[(row, col)].upper().replace("O", "0")
            assert text == f"R{row}C{col}", (
                f"cell ({row},{col}) holds {grid[(row, col)]!r}"
            )
