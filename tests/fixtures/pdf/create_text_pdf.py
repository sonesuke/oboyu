"""Create a PDF with actual text content for testing."""

from fpdf import FPDF

# Create simple English PDF with text
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(0, 10, "Simple Test PDF", ln=True)
pdf.ln(5)
pdf.multi_cell(0, 10, "This is a simple test PDF file.\nIt contains some English text for testing purposes.\nThe PDF extractor should be able to read this content.")

pdf.output("simple_text.pdf")
print("Created simple_text.pdf with actual text content")

# Create multi-page PDF
pdf = FPDF()
pdf.set_font("Arial", size=12)

# Page 1
pdf.add_page()
pdf.cell(0, 10, "Page 1: Introduction", ln=True)
pdf.ln(5)
pdf.multi_cell(0, 10, "This is the first page of a multi-page PDF document.")

# Page 2  
pdf.add_page()
pdf.cell(0, 10, "Page 2: Content", ln=True)
pdf.ln(5)
pdf.multi_cell(0, 10, "This is the second page with more content.\nEach page should be extracted properly.")

# Page 3
pdf.add_page()
pdf.cell(0, 10, "Page 3: Conclusion", ln=True)
pdf.ln(5)
pdf.multi_cell(0, 10, "This is the final page of the document.")

pdf.output("multipage_text.pdf")
print("Created multipage_text.pdf with actual text content")