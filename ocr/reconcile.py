"""Reconcile INDEXED grounded detection against the authoritative transcript.

Grounded detection returns boxes tagged with their token NUMBER (``{index: box}``),
because the model drifts off strict order on a dense page (it skips tokens), so
positional mapping is unsafe. We map each box to its token by the returned index,
force the text to that token (authoritative), and **infill** a geometric box for
any token the model didn't return (interpolated from neighbours; the content-aware
crop then snaps it to ink). Result: exactly one box per transcript token.
"""


def _estimate(prev, nxt, k, span):
    """Estimate a box for a gap, from neighbouring boxes [ymin,xmin,ymax,xmax]."""
    if prev and nxt:
        f = (k / span) if span else 0.5
        return [int(prev[i] + (nxt[i] - prev[i]) * f) for i in range(4)]
    if prev:                                   # trailing gap: place to the right
        w = max(20, prev[3] - prev[1])
        return [prev[0], min(1000, prev[3]), prev[2], min(1000, prev[3] + w)]
    if nxt:                                    # leading gap: place to the left
        w = max(20, nxt[3] - nxt[1])
        return [nxt[0], max(0, nxt[1] - w), nxt[2], max(0, nxt[1])]
    return [450, 450, 550, 550]                # whole page empty: center-ish


def _infill(result):
    n, count = len(result), 0
    for idx in range(n):
        if result[idx]["box_2d"] is not None:
            continue
        p = idx - 1
        while p >= 0 and result[p]["box_2d"] is None:
            p -= 1
        q = idx + 1
        while q < n and result[q]["box_2d"] is None:
            q += 1
        prev = result[p]["box_2d"] if p >= 0 else None
        nxt = result[q]["box_2d"] if q < n else None
        result[idx]["box_2d"] = _estimate(prev, nxt, idx - p, q - p)
        result[idx]["estimated"] = True
        count += 1
    return count


def reconcile_indexed(index_map, tokens):
    """Map an ``{1-based index: box}`` detection onto ``tokens``.

    Returns ``(boxes, n_infilled)`` where ``boxes`` has exactly ``len(tokens)``
    entries ``{text, box_2d[, estimated]}`` in transcript order. Tokens the model
    didn't return are infilled geometrically.
    """
    result = []
    for i, tok in enumerate(tokens):
        box = index_map.get(i + 1)
        result.append({"text": tok,
                       "box_2d": [int(v) for v in box[:4]] if box and len(box) >= 4 else None})
    n_infilled = _infill(result)
    return result, n_infilled
