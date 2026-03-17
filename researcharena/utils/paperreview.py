"""Client for paperreview.ai (Stanford Agentic Reviewer).

paperreview.ai has no public API, so we automate the web interaction:
  1. Submit PDF via their upload form (multipart POST)
  2. Poll email inbox for the review access token
  3. Fetch the review page and parse the results

Requires: playwright (for form submission), imaplib (for email polling).

The review contains 7 dimension scores + qualitative sections:
  - Summary, Strengths, Weaknesses, Detailed Comments,
    Questions for Authors, Overall Assessment
Dimensions: originality, research_importance, claim_support,
  experimental_soundness, writing_clarity, community_value, prior_work_context
"""

from __future__ import annotations

import email
import imaplib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()

PAPERREVIEW_URL = "https://paperreview.ai"
REVIEW_VIEW_URL = f"{PAPERREVIEW_URL}/review"

# Venues supported by paperreview.ai
VENUE_MAP = {
    "neurips": "NeurIPS",
    "icml": "ICML",
    "iclr": "ICLR",
    "cvpr": "CVPR",
    "aaai": "AAAI",
    "ijcai": "IJCAI",
    "acl": "ACL",
    "emnlp": "EMNLP",
}

REVIEW_DIMENSIONS = [
    "originality",
    "research_importance",
    "claim_support",
    "experimental_soundness",
    "writing_clarity",
    "community_value",
    "prior_work_context",
]


@dataclass
class PaperReviewResult:
    """Parsed result from paperreview.ai."""
    dimensions: dict[str, float]  # 7 dimension scores
    overall_score: float | None   # 1-10 overall (ICLR-calibrated)
    summary: str
    strengths: str
    weaknesses: str
    detailed_comments: str
    questions: str
    overall_assessment: str
    raw_html: str                 # full review page HTML for debugging


def submit_paper(
    pdf_path: str | Path,
    email_address: str,
    venue: str = "neurips",
    timeout: int = 600,
) -> str:
    """Submit a PDF to paperreview.ai and return the page response.

    Uses Playwright to automate the web form since there's no public API.

    Args:
        pdf_path: Path to the PDF file to submit
        email_address: Email to receive the review access token
        venue: Target venue (neurips, iclr, icml, etc.)
        timeout: Max seconds to wait for submission

    Returns:
        Confirmation text from the page after submission
    """
    from playwright.sync_api import sync_playwright

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    venue_label = VENUE_MAP.get(venue.lower(), "NeurIPS")

    console.print(f"  Submitting {pdf_path.name} to paperreview.ai (venue: {venue_label})...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(PAPERREVIEW_URL, timeout=timeout * 1000)

        # Fill in the email
        page.fill('input[name="email"]', email_address)

        # Fill venue (text input, not dropdown)
        try:
            page.fill('input[name="customVenue"]', venue_label, timeout=5000)
        except Exception:
            pass

        # Upload PDF
        page.set_input_files('input[name="pdf"]', str(pdf_path))

        # Click submit
        page.click('text=Submit for Review')

        # Wait for confirmation
        page.wait_for_timeout(3000)
        confirmation = page.text_content("body")

        browser.close()

    console.print("  Submission complete. Waiting for review via email...")
    return confirmation


def poll_email_for_token(
    email_address: str,
    email_password: str,
    imap_server: str = "imap.gmail.com",
    poll_interval: int = 60,
    max_wait: int = 3600,
) -> str:
    """Poll email inbox for the paperreview.ai access token.

    Args:
        email_address: Email address to check
        email_password: Email password or app-specific password
        imap_server: IMAP server hostname
        poll_interval: Seconds between email checks
        max_wait: Max seconds to wait for the email

    Returns:
        The access token string from the email
    """
    console.print(f"  Polling {email_address} for review token (max {max_wait}s)...")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            mail = imaplib.IMAP4_SSL(imap_server)
            mail.login(email_address, email_password)
            mail.select("inbox")

            # Search for emails from paperreview.ai
            _, message_ids = mail.search(None, '(FROM "paperreview" UNSEEN)')

            for msg_id in message_ids[0].split():
                _, msg_data = mail.fetch(msg_id, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])

                body = _extract_email_body(msg)
                token = _extract_token_from_body(body)

                if token:
                    console.print(f"  Found review token: {token[:8]}...")
                    mail.logout()
                    return token

            mail.logout()

        except Exception as e:
            console.print(f"  [yellow]Email check failed: {e}[/]")

        elapsed = int(time.time() - start)
        console.print(f"  No token yet ({elapsed}s elapsed). Retrying in {poll_interval}s...")
        time.sleep(poll_interval)

    raise TimeoutError(
        f"No review token received after {max_wait}s. "
        f"Check {email_address} manually."
    )


def fetch_review(token: str, timeout: int = 60) -> PaperReviewResult:
    """Fetch and parse the review page using the access token.

    Args:
        token: Access token received via email

    Returns:
        Parsed PaperReviewResult
    """
    from playwright.sync_api import sync_playwright

    console.print(f"  Fetching review with token {token[:8]}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(REVIEW_VIEW_URL, timeout=timeout * 1000)

        # Enter the token and click "Load Review"
        page.fill('input[name="token"]', token)
        page.click('text=Load Review')

        # Wait for review to load
        page.wait_for_timeout(10000)

        html = page.content()
        text = page.text_content("body")

        browser.close()

    return _parse_review(text, html)


def submit_and_wait(
    pdf_path: str | Path,
    email_address: str,
    email_password: str,
    imap_server: str = "imap.gmail.com",
    venue: str = "neurips",
    poll_interval: int = 60,
    max_wait: int = 3600,
) -> PaperReviewResult:
    """Full flow: submit paper → poll for token → fetch review.

    This is the main entry point for the pipeline.
    """
    submit_paper(pdf_path, email_address, venue)
    token = poll_email_for_token(
        email_address, email_password, imap_server,
        poll_interval=poll_interval, max_wait=max_wait,
    )
    return fetch_review(token)


# ── Internal helpers ─────────────────────────────────────────────────────


def _extract_email_body(msg) -> str:
    """Extract plain text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
            elif part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="replace")
    return ""


def _extract_token_from_body(body: str) -> str | None:
    """Extract the review access token from email body.

    The token format is not documented — we try common patterns.
    """
    # Look for a direct token pattern (alphanumeric string after "token" or in a URL)
    patterns = [
        r"paperreview\.ai/review\?token=([a-zA-Z0-9_-]+)",
        r"paperreview\.ai/review/([a-zA-Z0-9_-]+)",
        r"access\s*token[:\s]+([a-zA-Z0-9_-]{8,})",
        r"token[:\s]+([a-zA-Z0-9_-]{8,})",
        # Fallback: any long alphanumeric string that looks like a token
        r"\b([a-zA-Z0-9_-]{20,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _estimate_score_from_text(sections: dict) -> float:
    """Estimate a numeric score from qualitative review text.

    paperreview.ai doesn't provide numeric scores, so we estimate
    based on keyword signals in the weaknesses and overall assessment.
    Returns a score on the ICLR 0-10 scale.
    """
    weaknesses = sections.get("weaknesses", "").lower()
    overall = sections.get("overall_assessment", "").lower()
    strengths = sections.get("strengths", "").lower()
    combined = weaknesses + " " + overall

    # Start at 6 (marginal accept) and adjust
    score = 6.0

    # Strong negative signals
    severe = ["fundamental flaw", "fatal", "fabricat", "not reproducible",
              "plagiari", "no contribution", "trivial contribution"]
    for s in severe:
        if s in combined:
            score -= 2.0

    # Moderate negative signals
    moderate = ["major weakness", "significant concern", "not convincing",
                "lacks novelty", "incremental", "insufficient experiment",
                "missing baseline", "unfair comparison", "overclaim",
                "not well-supported", "seriously lacking"]
    for s in moderate:
        if s in combined:
            score -= 0.5

    # Positive signals
    positive = ["strong contribution", "well-written", "convincing result",
                "novel approach", "significant improvement", "solid experiment",
                "well-motivated", "thorough evaluation", "impressive"]
    for s in positive:
        if s in combined or s in strengths:
            score += 0.3

    return max(0.0, min(10.0, round(score * 2) / 2))  # clamp and round to 0.5


def _parse_review(text: str, html: str) -> PaperReviewResult:
    """Parse review text into structured PaperReviewResult.

    The review page has sections: Summary, Strengths, Weaknesses,
    Detailed Comments, Questions for Authors, Overall Assessment.
    """
    sections = {
        "summary": "",
        "strengths": "",
        "weaknesses": "",
        "detailed_comments": "",
        "questions": "",
        "overall_assessment": "",
    }

    # Try to split by section headers
    section_patterns = {
        "summary": r"(?:^|\n)\s*(?:##?\s*)?Summary\s*\n(.*?)(?=\n\s*(?:##?\s*)?(?:Strengths|$))",
        "strengths": r"(?:^|\n)\s*(?:##?\s*)?Strengths\s*\n(.*?)(?=\n\s*(?:##?\s*)?(?:Weaknesses|$))",
        "weaknesses": r"(?:^|\n)\s*(?:##?\s*)?Weaknesses\s*\n(.*?)(?=\n\s*(?:##?\s*)?(?:Detailed|$))",
        "detailed_comments": r"(?:^|\n)\s*(?:##?\s*)?Detailed\s*Comments?\s*\n(.*?)(?=\n\s*(?:##?\s*)?(?:Questions|$))",
        "questions": r"(?:^|\n)\s*(?:##?\s*)?Questions?\s*(?:for\s*Authors?)?\s*\n(.*?)(?=\n\s*(?:##?\s*)?(?:Overall|$))",
        "overall_assessment": r"(?:^|\n)\s*(?:##?\s*)?Overall\s*Assessment\s*\n(.*?)$",
    }

    for key, pattern in section_patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()

    # Extract dimension scores (look for patterns like "Originality: 7/10" or "7")
    dimensions = {}
    for dim in REVIEW_DIMENSIONS:
        # Try various score formats
        dim_label = dim.replace("_", " ")
        score_match = re.search(
            rf"{dim_label}\s*[:\-]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?",
            text, re.IGNORECASE,
        )
        if score_match:
            dimensions[dim] = float(score_match.group(1))

    # Extract overall score
    overall_score = None
    overall_match = re.search(
        r"(?:overall|final)\s*(?:score|rating)\s*[:\-]\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?",
        text, re.IGNORECASE,
    )
    if overall_match:
        overall_score = float(overall_match.group(1))
    elif dimensions:
        overall_score = sum(dimensions.values()) / len(dimensions)

    # paperreview.ai gives qualitative reviews without numeric scores.
    # The caller (review.py) uses a CLI agent to assign a score.

    return PaperReviewResult(
        dimensions=dimensions,
        overall_score=overall_score,
        summary=sections["summary"],
        strengths=sections["strengths"],
        weaknesses=sections["weaknesses"],
        detailed_comments=sections["detailed_comments"],
        questions=sections["questions"],
        overall_assessment=sections["overall_assessment"],
        raw_html=html,
    )
