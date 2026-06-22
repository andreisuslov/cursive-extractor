"""Fetch an American Diary Project diary's ground-truth transcript.

Many diaries on americandiaryproject.com carry a human transcription embedded in
their collection page (the ``TRANSCRIPTION`` section). It is the whole-diary text
-- authoritative labels for the (word-image, text) pairs the OCR pipeline produces
from the same diary's scans. About a third of diaries instead show "Coming soon!"
(no transcript yet); for those this writes nothing and reports it.

    python fetch_transcript.py --url <diary collection URL>
    # -> outputs/<document>/transcript.txt   (whole-diary ground truth, beside its PDF)

Pure ``requests`` (no browser): the transcript is in the page's server HTML.
"""

import argparse
import glob
import html
import os
import re

import requests
from scrape_diary import OUTPUT_DIR, diary_slug


def default_transcript_out(slug):
    """Place the transcript inside this diary's document folder (beside its PDF).

    The scraper makes ``outputs/<slug>_pages_X-Y/`` first, so prefer an existing
    document folder for this diary; fall back to ``outputs/<slug>/`` if none yet.
    """
    docdirs = sorted(d for d in glob.glob(os.path.join(OUTPUT_DIR, slug + "*")) if os.path.isdir(d))
    docdir = docdirs[0] if docdirs else os.path.join(OUTPUT_DIR, slug)
    return os.path.join(docdir, "transcript.txt")


def extract_transcript(page_html):
    """Pull the diary prose out of a collection page's ``TRANSCRIPTION`` section.

    Returns the cleaned transcript text, or "" if the diary has no transcript yet
    ("Coming soon!" placeholder)."""
    h = page_html.rfind("TRANSCRIPTION")
    if h < 0:
        return ""
    # The footer's social block ("Facebook ...") reliably follows the transcript.
    foot = page_html.find("Facebook", h)
    seg = page_html[h : foot if foot > 0 else h + 60000]
    seg = re.sub(r"<br\s*/?>", "\n", seg)
    seg = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", seg)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", seg))
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n[ \t]*", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    # Drop the section heading + collapsible-summary boilerplate.
    txt = re.sub(r"^TRANSCRIPTION\s*", "", txt)
    txt = re.sub(r"^Click to view or hide\s*", "", txt).strip()
    if "coming soon" in txt.lower() and len(txt.split()) < 60:
        return ""
    return txt


def fetch_transcript(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return extract_transcript(r.text)


def _selftest():
    has = (
        '<h2 id="transcript">TRANSCRIPTION</h2><details><summary>Click to view or hide</summary>'
        "<p>June 14, 1981 Woke up at noon &amp; went shopping.</p></details>"
        '<a href="#">Facebook</a>'
    )
    out = extract_transcript(has)
    assert out == "June 14, 1981 Woke up at noon & went shopping.", repr(out)
    none = (
        '<h2 id="transcript">TRANSCRIPTION</h2><details><summary>Click to view or hide</summary>'
        "<p>Coming soon! If you're interested in volunteering to transcribe, reach out.</p>"
        "</details><a>Facebook</a>"
    )
    assert extract_transcript(none) == "", repr(extract_transcript(none))
    assert extract_transcript("<p>no section here</p>") == ""
    print("selftest ok")


def main():
    parser = argparse.ArgumentParser(description="Fetch a diary's ground-truth transcript")
    parser.add_argument("--url", help="Diary collection page URL")
    parser.add_argument(
        "--out", default=None, help="Output .txt path (default: outputs/<document>/transcript.txt)"
    )
    parser.add_argument(
        "--selftest", action="store_true", help="Run the offline self-check and exit"
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.url:
        parser.error("--url is required (or pass --selftest)")

    txt = fetch_transcript(args.url)
    if not txt:
        print(f"No transcript available for {diary_slug(args.url)} (Coming soon / none).")
        return
    out = args.out or default_transcript_out(diary_slug(args.url))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(txt + "\n")
    print(f"Saved {len(txt.split())} words -> {out}")


if __name__ == "__main__":
    main()
