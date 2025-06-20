"""Script to create test PDF files for unit tests."""

from pathlib import Path

import pypdf


def create_simple_english_pdf():
    """Create a simple English PDF with text content."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = Path(__file__).parent / "simple_english.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Add metadata
    c.setTitle("Simple English Test PDF")
    c.setAuthor("Test Author")
    c.setSubject("Test Subject")

    # Add content
    c.drawString(100, 750, "This is a simple test PDF file.")
    c.drawString(100, 730, "It contains some English text for testing purposes.")
    c.drawString(100, 710, "The PDF extractor should be able to read this content.")

    # Save the PDF
    c.save()
    print(f"Created: {pdf_path}")


def create_multipage_pdf():
    """Create a multi-page PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = Path(__file__).parent / "multipage.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Add metadata
    c.setTitle("Multi-page Test PDF")
    c.setAuthor("Test System")

    # Page 1
    c.drawString(100, 750, "Page 1: Introduction")
    c.drawString(100, 730, "This is the first page of a multi-page PDF document.")
    c.showPage()

    # Page 2
    c.drawString(100, 750, "Page 2: Content")
    c.drawString(100, 730, "This is the second page with more content.")
    c.drawString(100, 710, "Each page should be extracted properly.")
    c.showPage()

    # Page 3
    c.drawString(100, 750, "Page 3: Conclusion")
    c.drawString(100, 730, "This is the final page of the document.")

    # Save the PDF
    c.save()
    print(f"Created: {pdf_path}")


def create_japanese_pdf():
    """Create a PDF with Japanese text."""
    # For simplicity, we'll create a text-based PDF using pypdf
    # In real tests, we'd use a proper Japanese font with reportlab
    pdf_path = Path(__file__).parent / "japanese.pdf"

    # Create a simple text file first
    text_content = """日本語のテストPDFファイル

このPDFファイルは日本語のテキストを含んでいます。
PDFエクストラクターは、この内容を正しく読み取ることができるはずです。

テスト内容：
• 漢字、ひらがな、カタカナ
• 句読点と記号
• 複数行のテキスト"""

    # Note: For actual Japanese PDF generation, we'd need proper font support
    # For testing purposes, we'll mark this as a TODO and create a placeholder
    with open(pdf_path, "w") as f:
        f.write("Placeholder for Japanese PDF test")
    print(f"Created placeholder: {pdf_path}")


def create_metadata_pdf():
    """Create a PDF with rich metadata."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = Path(__file__).parent / "metadata.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Add extensive metadata
    c.setTitle("PDF with Metadata")
    c.setAuthor("John Doe")
    c.setSubject("Testing PDF Metadata Extraction")
    c.setCreator("Test Creator Application")
    c.setProducer("Test PDF Producer")
    c.setKeywords(["test", "metadata", "extraction"])

    # Add content
    c.drawString(100, 750, "This PDF contains metadata that should be extracted.")

    # Save the PDF
    c.save()
    print(f"Created: {pdf_path}")


def create_empty_pdf():
    """Create an empty PDF file."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = Path(__file__).parent / "empty.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Don't add any content, just save
    c.save()
    print(f"Created: {pdf_path}")


if __name__ == "__main__":
    print("Creating test PDF files...")

    # Check if reportlab is available
    try:
        import reportlab

        create_simple_english_pdf()
        create_multipage_pdf()
        create_metadata_pdf()
        create_empty_pdf()
    except ImportError:
        print("reportlab not installed. Creating minimal test PDFs using pypdf...")

        # Create minimal PDFs using pypdf
        # Simple English PDF
        writer = pypdf.PdfWriter()
        writer.add_metadata(
            {
                "/Title": "Simple English Test PDF",
                "/Author": "Test Author",
            }
        )
        page = writer.add_blank_page(width=612, height=792)
        with open(Path(__file__).parent / "simple_english.pdf", "wb") as f:
            writer.write(f)
        print("Created minimal simple_english.pdf")

        # Multi-page PDF
        writer = pypdf.PdfWriter()
        writer.add_metadata(
            {
                "/Title": "Multi-page Test PDF",
                "/Author": "Test System",
            }
        )
        for i in range(3):
            writer.add_blank_page(width=612, height=792)
        with open(Path(__file__).parent / "multipage.pdf", "wb") as f:
            writer.write(f)
        print("Created minimal multipage.pdf")

        # Metadata PDF
        writer = pypdf.PdfWriter()
        writer.add_metadata(
            {
                "/Title": "PDF with Metadata",
                "/Author": "John Doe",
                "/Subject": "Testing PDF Metadata Extraction",
                "/Creator": "Test Creator Application",
                "/Producer": "Test PDF Producer",
                "/Keywords": "test, metadata, extraction",
            }
        )
        writer.add_blank_page(width=612, height=792)
        with open(Path(__file__).parent / "metadata.pdf", "wb") as f:
            writer.write(f)
        print("Created minimal metadata.pdf")

        # Empty PDF
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with open(Path(__file__).parent / "empty.pdf", "wb") as f:
            writer.write(f)
        print("Created minimal empty.pdf")

    create_japanese_pdf()
    print("\nAll test PDF files created!")
