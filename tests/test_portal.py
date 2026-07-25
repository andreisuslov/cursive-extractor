"""Provenance rules for portal saves — the guard that keeps agent cuts out of training data."""

import copy

from ocr.experiments.portal import cut_xs, harvest, is_human_cut, match_seed


def _word(text, box, xs):
    h = box[3] - box[1]
    return {"text": text, "box": box, "cuts": [[[x, 0], [x, h]] for x in xs]}


SEED = [_word("cat", [0, 0, 90, 50], [30, 60]), _word("dog", [0, 100, 90, 150], [30, 60])]


def test_untouched_agent_cuts_are_not_human():
    assert not is_human_cut(copy.deepcopy(SEED[0]), SEED)


def test_dragged_cut_is_human():
    w = copy.deepcopy(SEED[1])
    w["cuts"][0] = [[41, 0], [41, 50]]
    assert is_human_cut(w, SEED)


def test_subpixel_wobble_is_not_an_edit():
    """Re-serialisation must not masquerade as a human edit."""
    w = copy.deepcopy(SEED[0])
    w["cuts"][0] = [[30.4, 0], [30.4, 50]]
    assert not is_human_cut(w, SEED)


def test_added_or_far_moved_word_counts_as_human():
    assert is_human_cut(_word("cat", [500, 500, 590, 550], [30]), SEED)


def test_changed_cut_count_is_human():
    w = copy.deepcopy(SEED[0])
    w["cuts"] = w["cuts"][:1]
    assert is_human_cut(w, SEED)


def test_word_with_no_cuts_is_never_human():
    assert not is_human_cut(_word("cat", [0, 0, 90, 50], []), SEED)


def test_match_seed_picks_same_text_nearest_box():
    assert match_seed(_word("dog", [2, 102, 92, 152], [30, 60]), SEED)["box"] == [0, 100, 90, 150]
    assert match_seed(_word("fox", [0, 0, 90, 50], [30]), SEED) is None


def test_harvest_keeps_only_worker_cut_words():
    untouched, moved = copy.deepcopy(SEED[0]), copy.deepcopy(SEED[1])
    moved["cuts"][0] = [[41, 0], [41, 50]]
    keep, stats = harvest({"words": [untouched, moved]}, SEED)
    assert [w["text"] for w in keep] == ["dog"]
    assert stats["human"] == 1 and stats["untouched"] == 1


def test_harvest_drops_human_words_with_wrong_cut_count():
    """cut_predictor needs exactly len(text)-1 interior cuts; count them but don't ship them."""
    w = copy.deepcopy(SEED[1])
    w["cuts"] = [[[10, 0], [10, 50]]]
    keep, stats = harvest({"words": [w]}, SEED)
    assert keep == []
    assert stats["human"] == 1 and stats["wrong_count"] == 1


def test_skipped_words_are_ignored():
    w = copy.deepcopy(SEED[1])
    w["cuts"][0] = [[41, 0], [41, 50]]
    w["skip"] = True
    assert harvest({"words": [w]}, SEED)[0] == []


def test_cut_xs_returns_sorted_polyline_means():
    assert cut_xs(_word("ab", [0, 0, 90, 50], [60, 30])) == [30, 60]
