"""Text extraction for PDF, DOCX and TXT files."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class FileProcessingError(Exception):
    """Raised when a file cannot be read. The message is safe to show to the user."""


@dataclass
class ExtractionResult:
    text: str
    pages: int
    method: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(path: Path) -> ExtractionResult:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as exc:  # pragma: no cover
        raise FileProcessingError("PDF support requires the 'pypdf' package.") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise FileProcessingError("This PDF is password-protected.")
            except FileProcessingError:
                raise
            except Exception as exc:
                raise FileProcessingError("This PDF is encrypted and cannot be read.") from exc
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # one bad page should not kill the document
                logger.warning("Skipping unreadable PDF page in %s", path.name, exc_info=True)
        return ExtractionResult("\n\n".join(parts), len(reader.pages), "pypdf")
    except FileProcessingError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, AttributeError, TypeError) as exc:
        raise FileProcessingError(f"The PDF appears to be corrupted or invalid ({type(exc).__name__}).") from exc


def _read_docx(path: Path) -> ExtractionResult:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise FileProcessingError("DOCX support requires the 'python-docx' package.") from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise FileProcessingError(f"The DOCX file appears to be corrupted or invalid ({type(exc).__name__}).") from exc
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return ExtractionResult("\n".join(parts), 1, "python-docx")


def _read_txt(path: Path) -> ExtractionResult:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return ExtractionResult(raw.decode(enc), 1, f"text/{enc}")
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise FileProcessingError("The text file could not be decoded.")  # pragma: no cover


def extract_text(path: str | Path) -> ExtractionResult:
    path = Path(path)
    if not path.is_file():
        raise FileProcessingError(f"File not found: {path.name}")
    ext = path.suffix.lower()
    readers = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}
    if ext not in readers:
        raise FileProcessingError(f"Unsupported file type '{ext}'.")
    result = readers[ext](path)
    text = clean_text(result.text)
    if not text:
        hint = " It may be a scanned/image-only PDF (OCR is not supported)." if ext == ".pdf" else ""
        raise FileProcessingError(f"No extractable text found in '{path.name}'.{hint}")
    return ExtractionResult(text, result.pages, result.method)
