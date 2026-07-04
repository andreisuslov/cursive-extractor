#!/bin/bash
# Double-click this (or run it) to open the letter labeller with in-place Save working.
# It serves letter_label.html over http://localhost so the File System Access API is allowed,
# starting the tiny server ONLY if it isn't already up (so re-opening reuses it), then opens Chrome.
DIR="/Users/ansuslov/Documents/Development/cursivetransformer/ocr/experiments"
PORT=8417
URL="http://localhost:${PORT}/letter_label.html"

# Reuse a server that's already serving the labeller; otherwise start one detached (survives this window).
if ! curl -fsS -o /dev/null "$URL" 2>/dev/null; then
  ( cd "$DIR" && nohup python3 -m http.server "$PORT" >/tmp/labeller_server.log 2>&1 & )
  for _ in $(seq 1 30); do curl -fsS -o /dev/null "$URL" 2>/dev/null && break; sleep 0.2; done
fi

# Open in a Chromium browser (in-place save needs Chrome/Edge, not Safari); fall back to default.
if   open -a "Google Chrome"  "$URL" 2>/dev/null; then :
elif open -a "Microsoft Edge" "$URL" 2>/dev/null; then :
elif open -a "Brave Browser"  "$URL" 2>/dev/null; then :
else open "$URL"; fi
