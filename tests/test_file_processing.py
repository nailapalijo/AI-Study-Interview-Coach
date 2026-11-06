import pytest
from docx import Document

from utils.file_processor import FileProcessingError, extract_text
from utils.validators import ValidationError, validate_text, validate_upload


def make_pdf(path, text):
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    path.write_bytes(out)


def test_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Hello   world\r\n\r\n\r\n\r\nSecond para", encoding="utf-8")
    r = extract_text(p)
    assert r.text == "Hello world\n\nSecond para"


def test_txt_latin1_fallback(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("café résumé".encode("latin-1"))
    assert "caf" in extract_text(p).text


def test_docx_paragraphs_and_tables(tmp_path):
    d = Document()
    d.add_paragraph("Python developer with 5 years experience")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Skill", "SQL"
    p = tmp_path / "cv.docx"
    d.save(p)
    r = extract_text(p)
    assert "Python developer" in r.text and "Skill | SQL" in r.text


def test_pdf_text(tmp_path):
    p = tmp_path / "a.pdf"
    make_pdf(p, "Kubernetes deployment notes")
    r = extract_text(p)
    assert "Kubernetes" in r.text and r.pages == 1


def test_corrupted_pdf_and_docx(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(FileProcessingError):
        extract_text(bad)
    bad2 = tmp_path / "bad.docx"
    bad2.write_bytes(b"not a zip")
    with pytest.raises(FileProcessingError):
        extract_text(bad2)


def test_empty_and_missing(tmp_path):
    e = tmp_path / "e.txt"
    e.write_text("   \n  ")
    with pytest.raises(FileProcessingError):
        extract_text(e)
    with pytest.raises(FileProcessingError):
        extract_text(tmp_path / "nope.txt")


def test_validate_upload():
    assert validate_upload("cv.PDF", 100) == ".pdf"
    for name, size in [("x.exe", 10), ("x.pdf", 0), ("x.pdf", 11 * 1024 * 1024), ("", 5), ("noext", 5)]:
        with pytest.raises(ValidationError):
            validate_upload(name, size)


def test_validate_text():
    assert validate_text("  hi  ", "X", 2) == "hi"
    with pytest.raises(ValidationError):
        validate_text("a", "X", 2)
    with pytest.raises(ValidationError):
        validate_text("a" * 11, "X", 1, 10)
