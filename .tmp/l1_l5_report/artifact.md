# Reference

- Retained reference: `C:\Users\ASUSA\.codex\plugins\cache\openai-curated-remote\openai-templates\0.1.1\skills\artifact-template-system-design\assets\reference.docx`
- SHA-256: `13504F6C221A42C1726460A9E865E563355539FF97D702D6C9B2267B4B261D76`
- Reference pages: 7
- Sections: 1
- Evidence: `.tmp/l1_l5_report/template-style-evidence.json`
- Render evidence: `.tmp/l1_l5_report/reference-render/page-1.png` through `page-7.png`

# Page system

- US Letter portrait, 8.50 by 11.00 inches.
- Margins: left 0.70, right 0.70, top 0.70, bottom 0.62 inches.
- One section, new-page start, different first page enabled.
- No visible header. Footer centered and used as a recurring document identifier.

# Typography

- Reference body uses Helvetica Neue. Chinese output substitutes Microsoft YaHei while retaining the reference scale and spacing.
- Title role: 22 pt, bold, left aligned, 7.5 pt after. The cover also uses a larger directly formatted title line.
- Heading 1 role: 13.5 pt, bold, keep with next, approximately 1.19 line spacing.
- Heading 3 role: bold, muted secondary heading.
- The final document keeps all titles and headings black as required by the document authoring standard.

# Lists and tables

- Tables use a dark navy header with white bold text, thin white or light-gray internal borders, and alternating pale-blue and white body rows.
- Field dictionaries use four columns in this order: field, type, required, meaning.
- Header rows repeat across pages. Rows expand to fit wrapped text and are not assigned fixed heights.
- Lists use modest hanging indents and remain left aligned.

# Components

- Cover: system name, report title, status, owner, update date, and a two-column metadata table.
- Body: numbered Heading 1 sections, secondary Heading 3 labels, explanatory paragraphs, process tables, schema dictionaries, and a final next-step table.
- Footer: project name and report purpose.
- The reference architecture image is not reused because the requested content needs a five-layer clinical workflow rather than the generic four-box architecture. A source-derived navy table component replaces it.

# Content flow

1. Executive summary and project status.
2. End-to-end runtime logic and orchestration.
3. L1 through L5 implementation logic.
4. LLM versus rule responsibilities.
5. Verification evidence and current boundaries.
6. Schema concepts and L1 through L5 field dictionaries.
7. Orchestrator contracts and next steps.

# Slot map

- The full body is an editable report slot. The reference placeholders are removed.
- Page geometry, theme colors, styles, and table visual language are preserved as the design authority.
- Cover placeholders are replaced by the system name, report title, status, update date, and scope.
- Footer placeholder is replaced by the project name and report purpose.

# Package preservation

- Preserve the reference theme, styles, numbering definitions, section properties, and document settings unless Chinese font support requires a run-level East Asian font.
- Remove unused placeholder body text and the generic architecture image relationship from the active document body.
- No comments, tracked changes, content controls, footnotes, or fields are present in the reference.

# Fidelity gates

- Retained reference remains byte-for-byte unchanged.
- Final document keeps the same page size, margins, single-column body, heading hierarchy, table colors, and footer rhythm.
- Every final page must be rendered and inspected for clipping, broken table rows, unexpected blank pages, and missing Chinese glyphs.
