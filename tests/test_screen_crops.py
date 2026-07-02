"""Pure decision logic of the pre-GPU crop screen (no API, no PDFs)."""

from ocr.screen_crops import best_transcript_match, decide

TOKENS = ["Uncle", "John", "went", "to", "the", "morning,", "sun", "wealth."]


# --- keep ---------------------------------------------------------------------


def test_exact_match_keeps():
    assert decide("Uncle", "Uncle", TOKENS) == ("keep", None)


def test_case_insensitive_keeps():
    assert decide("Uncle", "uncle", TOKENS) == ("keep", None)


def test_fuzzy_label_match_keeps():
    # "wealth" vs "wealth." -> ratio 12/13 ~ 0.92 >= 0.7
    assert decide("wealth.", "wealth", TOKENS) == ("keep", None)


def test_keep_wins_over_transcript():
    # The read matches the label AND is a transcript token: keep, not relabel.
    assert decide("the", "the", TOKENS) == ("keep", None)


def test_punct_noise_in_read_keeps():
    # Regression: a stray comma in the read (', to' for label 'to') failed the raw
    # keep compare (ratio 0.67 < 0.7) and bounced into a relabel of the same word.
    assert decide("to", ", to", TOKENS) == ("keep", None)


def test_relabel_to_own_token_is_keep():
    # Regression: read matches the label modulo punctuation and the transcript
    # holds a variant of the SAME word -- that is a keep, not a relabel that
    # churns the label's punctuation ('Mrs.' -> 'Mrs').
    assert decide("Mrs.", ", Mrs", ["Mrs", "Smith", "went"]) == ("keep", None)


# --- relabel ------------------------------------------------------------------


def test_read_of_other_transcript_word_relabels():
    decision, new_text = decide("the", "morning", TOKENS)
    assert decision == "relabel"
    assert new_text == "morning,"  # the token VERBATIM, punctuation kept


def test_relabel_strips_punctuation_both_sides():
    decision, new_text = decide("sun", "wealth.", TOKENS)
    assert decision == "relabel"
    assert new_text == "wealth."


def test_fuzzy_transcript_match_relabels():
    # "mornig" vs "morning" -> ratio 12/13 ~ 0.92 >= 0.8
    decision, new_text = decide("the", "mornig", TOKENS)
    assert decision == "relabel"
    assert new_text == "morning,"


def test_relabel_is_case_insensitive():
    decision, new_text = decide("the", "UNCLE", TOKENS)
    assert decision == "relabel"
    assert new_text == "Uncle"


# --- drop ---------------------------------------------------------------------


def test_empty_read_drops():
    assert decide("Uncle", "", TOKENS) == ("drop", None)
    assert decide("Uncle", "   ", TOKENS) == ("drop", None)


def test_garbage_read_drops():
    assert decide("Uncle", "zzqx", TOKENS) == ("drop", None)


def test_weak_transcript_similarity_drops():
    # "mrng" vs "morning" -> ratio 8/11 ~ 0.73 < 0.8: too weak to rewrite a label.
    assert decide("the", "mrng", TOKENS) == ("drop", None)


def test_between_ratios_drops():
    # Matches a transcript token at ~0.7 (enough to keep, NOT enough to relabel):
    # "went" vs "want" -> ratio 6/8 = 0.75.
    assert decide("the", "want", ["went", "the"]) == ("drop", None)


# --- best_transcript_match ----------------------------------------------------


def test_best_transcript_match_prefers_exact():
    assert best_transcript_match("morning", TOKENS) == "morning,"


def test_best_transcript_match_prefers_verbatim_instance():
    # Regression: the first stripped-exact token won even when a later instance
    # matched the read's punctuation verbatim ('Mr.' relabeled to 'Mr').
    assert best_transcript_match("Mr.", ["Mr", "Mr."]) == "Mr."
    assert decide("Millard", "Mr.", ["Mr", "Mr.", "Millard"]) == ("relabel", "Mr.")


def test_best_transcript_match_picks_highest_ratio():
    assert best_transcript_match("mornings", ["morn", "morning,"]) == "morning,"


def test_best_transcript_match_none_for_empty_or_punct():
    assert best_transcript_match("", TOKENS) is None
    assert best_transcript_match("...", TOKENS) is None


def test_punct_only_transcript_tokens_ignored():
    assert best_transcript_match("word", ["--", "..."]) is None
