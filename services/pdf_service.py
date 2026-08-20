import os
import re
from pathlib import Path

class PDFService:
    @classmethod
    def generate_meeting_pdf(
        cls,
        meeting_title: str,
        created_at_str: str,
        duration_str: str,
        summary_markdown: str,
        transcript_text: str,
        output_pdf_path: Path
    ) -> Path:
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

            doc = SimpleDocTemplate(
                str(output_pdf_path),
                pagesize=letter,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40
            )

            styles = getSampleStyleSheet()

            # Custom Palette
            PRIMARY = colors.HexColor("#0f766e")    # Teal 700
            SECONDARY = colors.HexColor("#1e293b")  # Slate 800
            TEXT_MUTED = colors.HexColor("#64748b") # Slate 500
            BG_LIGHT = colors.HexColor("#f8fafc")   # Slate 50
            BORDER_COLOR = colors.HexColor("#e2e8f0")

            # Custom Typography Styles
            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontName='Helvetica-Bold',
                fontSize=20,
                leading=24,
                textColor=PRIMARY,
                alignment=TA_LEFT
            )

            subtitle_style = ParagraphStyle(
                'DocSubtitle',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                leading=14,
                textColor=TEXT_MUTED,
                alignment=TA_LEFT
            )

            h2_style = ParagraphStyle(
                'SectionHeading',
                parent=styles['Heading2'],
                fontName='Helvetica-Bold',
                fontSize=12,
                leading=16,
                textColor=PRIMARY,
                spaceBefore=12,
                spaceAfter=6
            )

            body_style = ParagraphStyle(
                'BodyContent',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=9.5,
                leading=14,
                textColor=SECONDARY,
                spaceAfter=5
            )

            bullet_style = ParagraphStyle(
                'BulletText',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=9.5,
                leading=14,
                textColor=SECONDARY,
                leftIndent=15,
                spaceAfter=4
            )

            meta_label_style = ParagraphStyle(
                'MetaLabel',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=9,
                leading=12,
                textColor=PRIMARY
            )

            meta_val_style = ParagraphStyle(
                'MetaVal',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=9,
                leading=12,
                textColor=SECONDARY
            )

            transcript_style = ParagraphStyle(
                'TranscriptBody',
                parent=styles['Normal'],
                fontName='Courier',
                fontSize=8,
                leading=11,
                textColor=SECONDARY
            )

            elements = []

            # 1. Header Banner
            elements.append(Paragraph("🦁 LionScribe &mdash; Meeting Summary & Notes", title_style))
            elements.append(Paragraph("Generated automatically with OpenAI Whisper and Llama 3", subtitle_style))
            elements.append(Spacer(1, 8))
            elements.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=4, spaceAfter=10))

            # 2. Meeting Metadata Box
            meta_data = [
                [Paragraph("Meeting Title:", meta_label_style), Paragraph(meeting_title, meta_val_style),
                 Paragraph("Date & Time:", meta_label_style), Paragraph(created_at_str, meta_val_style)],
                [Paragraph("Duration:", meta_label_style), Paragraph(duration_str, meta_val_style),
                 Paragraph("AI Models:", meta_label_style), Paragraph("OpenAI Whisper (Base) + Llama 3 (8B)", meta_val_style)]
            ]

            meta_table = Table(meta_data, colWidths=[80, 180, 75, 185])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
                ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(meta_table)
            elements.append(Spacer(1, 10))

            # 3. Parse Markdown Summary into structured flowables
            lines = summary_markdown.split('\n')
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith('#'):
                    clean_heading = re.sub(r'^#+\s*', '', line)
                    elements.append(Paragraph(clean_heading, h2_style))
                elif line.startswith('- [ ]') or line.startswith('- [x]'):
                    task_text = line.replace('- [ ]', '&#9633; ').replace('- [x]', '&#9632; ')
                    elements.append(Paragraph(task_text, bullet_style))
                elif line.startswith('- ') or line.startswith('* '):
                    bullet_text = "&bull; " + line[2:].strip()
                    elements.append(Paragraph(bullet_text, bullet_style))
                else:
                    elements.append(Paragraph(line, body_style))

            elements.append(Spacer(1, 12))

            # 4. Appendix: Full Raw Transcription snippet
            if transcript_text:
                elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR, spaceBefore=8, spaceAfter=8))
                elements.append(Paragraph("📄 Full Transcription Log", h2_style))
                
                for p in transcript_text.strip().split('\n\n'):
                    if p.strip():
                        elements.append(Paragraph(p.strip().replace('\n', '<br/>'), transcript_style))
                        elements.append(Spacer(1, 3))

            doc.build(elements)
            return output_pdf_path
        except Exception as e:
            print(f"[PDFService] Warning: Could not generate PDF via ReportLab ({e}). Creating plain text fallback.")
            with open(output_pdf_path, 'wb') as f:
                f.write(f"%PDF-1.4\n% LionScribe Meeting: {meeting_title}\n{summary_markdown}".encode('utf-8'))
            return output_pdf_path
