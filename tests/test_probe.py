"""Tests for the language-conditioning probe (pure construction + paired ΔSR); rollout is a seam."""

import pytest

from eval.probe import (
    INSTRUCTION_VARIANTS,
    _probe_rollout,
    blank_instruction,
    build_instruction_variants,
    feature_shift_probe,
    language_sensitivity,
    language_sensitivity_paired,
    mismatch_instruction,
    shuffle_instruction,
)


def test_blank_instruction() -> None:
    assert blank_instruction("pick up the bowl") == ""


def test_shuffle_preserves_words_and_is_deterministic() -> None:
    instr = "pick up the black bowl and place it"
    s1 = shuffle_instruction(instr, seed=1)
    s2 = shuffle_instruction(instr, seed=1)
    assert s1 == s2                                   # deterministic
    assert sorted(s1.split()) == sorted(instr.split())  # same multiset of words
    assert s1 != instr                                # reordered


def test_shuffle_single_word_unchanged() -> None:
    assert shuffle_instruction("grasp") == "grasp"


def test_mismatch_picks_other_or_returns_self() -> None:
    assert mismatch_instruction("a", ["a", "b", "c"], seed=0) in {"b", "c"}
    assert mismatch_instruction("a", []) == "a"        # nothing to mismatch
    assert mismatch_instruction("a", ["a"]) == "a"     # only itself available


def test_build_instruction_variants() -> None:
    v = build_instruction_variants("open the door", others=["close the box"], seed=0)
    assert set(v) == set(INSTRUCTION_VARIANTS)
    assert v["correct"] == "open the door"
    assert v["blank"] == ""
    assert v["mismatched"] == "close the box"


def test_language_sensitivity_point() -> None:
    assert language_sensitivity(0.8, 0.3) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        language_sensitivity(1.5, 0.3)


def test_language_sensitivity_paired() -> None:
    correct = [1, 1, 1, 0]
    ablated = [0, 1, 0, 0]
    res = language_sensitivity_paired(correct, ablated, n_resamples=500, seed=0)
    assert res["sr_correct"] == pytest.approx(0.75)
    assert res["sr_ablated"] == pytest.approx(0.25)
    assert res["delta"] == pytest.approx(0.5)
    assert res["ci_lo"] <= res["delta"] <= res["ci_hi"]
    assert res["n"] == 4


def test_probe_rollout_and_feature_shift_are_seams() -> None:
    with pytest.raises(NotImplementedError):
        _probe_rollout(object(), "libero_spatial:0", "pick up", 0)
    with pytest.raises(NotImplementedError):
        feature_shift_probe()
