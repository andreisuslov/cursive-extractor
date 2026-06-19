#!/usr/bin/env python3
"""Split a stroke dataset into an "easybank" subset and a redo word list.

Items whose word contains none of i/j/t/x (the letters whose dots/crossbars are
hardest to vectorize) are kept as ``easy_output``; the rejected words are written
to ``redo_output`` as a ``const words = [...]`` JS list for re-collection.

    python setup_easybank.py synthbank.json easybank.json redobank.txt
"""
import json
import sys


def process_data(input_file, easy_output, redo_output):
    """Filter items containing i/j/t/x out of ``input_file``: write the rest to
    ``easy_output`` (JSON) and the rejected words to ``redo_output`` (JS list)."""
    with open(input_file) as f:
        data = json.load(f)
    easy_data = []
    redo_words = []
    for item in data:
        word = item["metadata"]["asciiSequence"]
        if not any(char in word for char in "ijtx"):
            easy_data.append(item)
        else:
            redo_words.append(word)
    with open(easy_output, "w") as f:
        json.dump(easy_data, f)
    with open(redo_output, "w") as f:
        f.write(f"const words = {json.dumps(redo_words)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 setup_easybank.py <input_file> <easy_output> <redo_output>")
        sys.exit(1)
    process_data(sys.argv[1], sys.argv[2], sys.argv[3])
