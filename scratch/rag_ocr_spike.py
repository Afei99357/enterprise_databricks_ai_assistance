"""Spike: validate vision-model OCR quality on representative PDF pages.

Not part of the production pipeline -- a one-off validation script, run
manually with real Databricks credentials, to check whether a candidate
vision-capable model handles this document's trickiest layouts before
committing to the full RAG ingestion design:
  - page 5:  a dense table (row/column alignment)
  - page 3:  body text + a colored callout box (must stay separate)
  - page 25: a flyer template (non-linear layout, placeholder brackets)

Usage:
  export WNV_DATABRICKS_HOST=...
  export WNV_DATABRICKS_TOKEN=...
  export WNV_VISION_ENDPOINT=databricks-gemma-3-12b   # or your candidate model
  uv run python scratch/rag_ocr_spike.py
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from urllib import request

import pdfplumber

PDF_PATH = "data/documents/WNV-Outbreak-Communications-Toolkit-2025_508c.pdf"
TEST_PAGES = [3, 5, 25]  # callout box, table, flyer template
OUT_DIR = Path("scratch/ocr_spike_output")

EXTRACTION_PROMPT = """\
Extract all text from this document page as clean markdown.

Rules:
- Represent tables as markdown tables, preserving row/column alignment exactly.
- If a colored callout box or sidebar is visually separate from the main
  body text, describe it as a distinct block under a "Callout:" label,
  not interleaved into the paragraph it sits beside.
- Preserve bullet/sub-bullet nesting exactly as shown.
- If the page is a fill-in-the-blank template (contains bracketed
  placeholders like [INSERT ...]), extract it as-is and add a line at the
  top: "TEMPLATE PAGE - contains placeholder fields, not factual content."
- Do not summarize or omit content. Do not invent content not on the page.
"""


def render_page_to_png_bytes(pdf_path: str, page_number: int) -> bytes:
    """Render a 1-indexed PDF page to PNG bytes."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        image = page.to_image(resolution=150)
        buf = io.BytesIO()
        image.original.save(buf, format="PNG")
        return buf.getvalue()


def call_vision_model(image_bytes: bytes, host: str, token: str, endpoint: str) -> str:
    """Send one page image to a Databricks vision-capable chat endpoint."""
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode()
    url = f"https://{host}/serving-endpoints/{endpoint}/invocations"
    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


def main() -> None:
    host = os.environ["WNV_DATABRICKS_HOST"].removeprefix("https://").removesuffix("/")
    token = os.environ["WNV_DATABRICKS_TOKEN"]
    endpoint = os.environ.get("WNV_VISION_ENDPOINT", "databricks-gemma-3-12b")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for page_number in TEST_PAGES:
        print(f"--- Page {page_number} ---")
        image_bytes = render_page_to_png_bytes(PDF_PATH, page_number)
        markdown = call_vision_model(image_bytes, host, token, endpoint)
        out_path = OUT_DIR / f"page_{page_number}.md"
        out_path.write_text(markdown)
        print(markdown)
        print(f"(written to {out_path})\n")


if __name__ == "__main__":
    main()
