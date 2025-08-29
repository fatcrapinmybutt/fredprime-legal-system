"""Utility functions for building DOCX files for the chief judge packet.

This module avoids duplicating raw WordprocessingML strings by storing
common fragments as templates.  The templates are formatted with the
desired text and zipped into a minimal DOCX package.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final
import zipfile
from xml.sax.saxutils import escape


# Module level constants used to assemble the DOCX structure.  These are
# defined once so edits to the underlying XML are straightforward.
DOCUMENT_TEMPLATE: Final[str] = (
    """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>{text}</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
)

CONTENT_TYPES_XML: Final[str] = (
    """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
            ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
)

RELS_XML: Final[str] = (
    """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
                Target="word/document.xml"/>
</Relationships>
"""
)


def create_docx_from_text(text: str, output_path: Path) -> Path:
    """Create a minimal DOCX containing *text* and write it to *output_path*.

    Parameters
    ----------
    text:
        Text to embed in the resulting document.
    output_path:
        Location where the DOCX will be written.

    Returns
    -------
    Path
        The path to the generated DOCX file.
    """

    document_xml = DOCUMENT_TEMPLATE.format(text=escape(text))

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        word_dir = tmpdir_path / "word"
        rels_dir = tmpdir_path / "_rels"
        word_dir.mkdir()
        rels_dir.mkdir()

        (word_dir / "document.xml").write_text(document_xml, encoding="utf-8")
        (tmpdir_path / "[Content_Types].xml").write_text(
            CONTENT_TYPES_XML, encoding="utf-8"
        )
        (rels_dir / ".rels").write_text(RELS_XML, encoding="utf-8")

        with zipfile.ZipFile(output_path, "w") as docx:
            docx.write(tmpdir_path / "[Content_Types].xml", "[Content_Types].xml")
            docx.write(rels_dir / ".rels", "_rels/.rels")
            docx.write(word_dir / "document.xml", "word/document.xml")

    return output_path


__all__ = [
    "create_docx_from_text",
    "DOCUMENT_TEMPLATE",
    "CONTENT_TYPES_XML",
    "RELS_XML",
]
