# Structured output: typed table cells (API contract note)

Scope: the `?structured=true` response only. Drafted 2026-09-07 as part of the
F1 remediation. This note describes the change to the table representation and
nothing else; the ingest, completeness, orientation and reading-order work is
tracked separately and has not landed.

## Why it changed

The old response put the layout model's own HTML string in two places: in
`blocks[].html`, and again as a line inside the `markdown` string. A consumer
reading `markdown` therefore had to decide, line by line, whether a line was
model markup or recognized text. The demo made that decision with a tag-prefix
test and passed matching lines through to the page, which is how a document
that contains the characters `<p onclick=...>` ended up executing script in a
browser.

The fix removes the decision. Tables are typed data, Markdown is text, and no
part of the response is markup the client is expected to insert.

## What a table block looks like now

```json
{
  "type": "table",
  "bbox": [x0, y0, x1, y1],
  "cells": [
    {"row": 0, "col": 0, "row_span": 2, "col_span": 1, "header": false, "text": "Alpha"},
    {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "header": false, "text": "Beta"},
    {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "header": false, "text": "Gamma"}
  ],
  "html": "<table><tr><td rowspan=\"2\">Alpha</td><td>Beta</td></tr><tr><td>Gamma</td></tr></table>",
  "cell_boxes": [[x0, y0, x1, y1]],
  "cell_boxes_frame": "upstream-unverified"
}
```

- **`cells`** is the contract. `row` and `col` are zero-based grid positions
  with spans already applied, so a cell that follows a rowspan lands in the
  first free column rather than in column zero. `text` is the recognized text
  of the cell, with entity references decoded and whitespace collapsed.
- **`html`** is still emitted so existing callers keep a field they can store.
  It is now generated from `cells` with every cell text escaped, and the only
  attributes it ever carries are `rowspan` and `colspan`. It is export data.
  Do not insert it into a page: build the table from `cells`.
- **`cell_boxes`** is the layout model's own `cell_bbox` evidence, passed
  through unchanged. `cell_boxes_frame` is `upstream-unverified` because the
  coordinate frame of those boxes (page pixels or region-relative) has not
  been checked against page geometry yet. Do not mix them with line boxes
  until that is resolved.
- **`warnings`** appears when the table markup could not be read at all. In
  that case `cells` is empty and `html` is empty: the service reports that it
  could not resolve the structure instead of returning a guess.

## What changed in `markdown`

Two things: tables became pipe tables, and recognized text became escaped.

`markdown` carries a GFM pipe table and never an HTML fragment:

```
| Alpha | Beta |
| --- | --- |
|  | Gamma |
```

- Pipe tables have no way to express a span, so a spanned cell puts its text
  in its origin cell and leaves the covered cells empty. `cells` keeps the
  spans.
- GFM requires a header row, so row zero is rendered as the header. The
  `header` flag on each cell keeps the model's own answer, which may differ.

### Recognized text in `markdown` is escaped

Markdown is a format consumers hand to a renderer, and most renderers pass raw
HTML straight through. So recognized text placed in `markdown` is
backslash-escaped over CommonMark's escapable ASCII punctuation set:

| Recognized text                 | In `markdown`                           | What a renderer shows           |
| ------------------------------- | --------------------------------------- | ------------------------------- |
| `<p onclick=alert(1)>CLICK</p>` | `\<p onclick\=alert\(1\)\>CLICK\<\/p\>` | `<p onclick=alert(1)>CLICK</p>` |
| `C:\Users\file.txt`             | `C\:\\Users\\file\.txt`                 | `C:\Users\file.txt`             |
| `## Fake heading`               | `\#\# Fake heading`                     | `## Fake heading`               |

A pipe inside recognized cell text is escaped by the same rule, so it stays
inside its cell instead of ending it.

Only recognized text is escaped. The structure XLiteOCR adds itself stays
live: a title block is still `# ` plus its escaped text, table rows still use
real pipes, and a figure is still `![figure](figure)`. A recognized line that
looks like a heading, a list item or a table row therefore cannot become one.

The same holds inside a line. Backticks and asterisks are punctuation, so
`` `code` `` and `**bold**` in a document are escaped like anything else, and
recognized text cannot turn into an inline-code or bold element either. A
consumer that reads `markdown` has to unescape before it matches inline
syntax, not after: a pattern run over the raw string still matches the
backtick in `` \` `` and would show the reader a code box with stray
backslashes where the document had plain characters. Both demo renderers walk
the string once and read the escape pair first, which is what makes the
sentence above true for inline markup as well as block markup.

The spec's own punctuation set is used rather than a shortlist, so there is no
per-character judgement about which characters matter in which dialect, and
the inverse is a single unambiguous rule. `engine.structure.markdown_unescape`
implements it, and both demo renderers apply the same inverse at the point
they create a text node.

The trade-off is that the raw string is noisier to read: `Total $1,234.56`
appears as `Total \$1,234\.56`. It renders identically, and callers who want
clean strings should read the verbatim fields instead.

### The verbatim fields

`lines[].text`, `full_text`, `blocks[].text` and `cells[].text` carry exactly
what was recognized, with no escaping. If a page contains the characters
`<p onclick=alert(1)>`, those fields say so. Treat every recognized string as
untrusted input: escaped in `markdown` means safe to hand to a Markdown
renderer, not safe to insert into a page.

## Compatibility

- Existing fields keep their names and shapes: `pages`, `lines`, `full_text`,
  `blocks[].type`, `blocks[].bbox`, `blocks[].text`, `figures`, and
  `blocks[].html`.
- Breaking for one case: a client that scanned `markdown` for `<table>` will
  no longer find it and must read `cells` (or the pipe table).
- `schema_version` is not in the response yet. It arrives with the ingest and
  completeness work, together with page counts and completion status.
