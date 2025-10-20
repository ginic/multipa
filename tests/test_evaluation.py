from collections import Counter

import pandas as pd
import pytest

from multipa.evaluation import ModelEvaluator, compute_edit_distance_errors, calculate_by_token_error_rates


@pytest.mark.parametrize(
    "ref,pred,subs,dels,inserts,token_counts",
    [
        ("po", "bo", {("p", "b"): 1}, {}, {}, {"p": 1, "o": 1}),
        ("dʒɹɪŋk", "dɹɪnkə", {("ŋ", "n"): 1}, {"ʒ": 1}, {"ə": 1}, {"d": 1, "ʒ": 1, "ɹ": 1, "ɪ": 1, "ŋ": 1, "k": 1}),
        ("tuːt", "", {}, {"t": 2, "uː": 1}, {}, {"t": 2, "uː": 1}),
        ("", "tuːt", {}, {}, {"t": 2, "uː": 1}, {}),
    ],
)
def test_compute_edit_distance_errors(ref, pred, subs, dels, inserts, token_counts):
    actual_subs, actual_dels, actual_inserts, actual_tokens = compute_edit_distance_errors(pred, ref)
    assert actual_subs == subs
    assert actual_dels == dels
    assert actual_inserts == inserts
    assert actual_tokens == token_counts


@pytest.mark.parametrize(
    "ref,pred,subs,dels,inserts,token_counts",
    [
        ("po", "bo", {("p", "b"): 1}, {}, {}, {"p": 1, "o": 1}),
        ("dʒɹɪŋk", "dɹɪnkə", {("ŋ", "n"): 1}, {"ʒ": 1}, {"ə": 1}, {"d": 1, "ʒ": 1, "ɹ": 1, "ɪ": 1, "ŋ": 1, "k": 1}),
        ("tuːt", "", {}, {"t": 2, "u": 1, "ː": 1}, {}, {"t": 2, "u": 1, "ː": 1}),
        ("", "tuːt", {}, {}, {"t": 2, "u": 1, "ː": 1}, {}),
    ],
)
def test_computes_edit_distance_no_ipa_tok(ref, pred, subs, dels, inserts, token_counts):
    actual_subs, actual_dels, actual_inserts, actual_tokens = compute_edit_distance_errors(pred, ref, use_ipa_tokenise=False)
    assert actual_subs == subs
    assert actual_dels == dels
    assert actual_inserts == inserts
    assert actual_tokens == token_counts


@pytest.mark.parametrize(
    "token_counts,substitutions,deletions,expected_error_rates",
    [
        # Basic error rate calculation with substitutions and deletions
        (
            {"a": 10, "b": 5, "c": 3},
            {("a", "x"): 2, ("b", "y"): 1},
            {"a": 1, "c": 3},
            {"a": 0.3, "b": 0.2, "c": 1.0},
        ),
        # No errors - all tokens should have 0.0 error rate
        (
            {"a": 5, "b": 3},
            {},
            {},
            {"a": 0.0, "b": 0.0},
        ),
        # Multiple different substitutions for same reference token
        (
            {"ð": 10},
            {("ð", "d"): 3, ("ð", "t"): 2},
            {"ð": 1},
            {"ð": 0.6},
        ),
        # Only deletions
        (
            {"p": 8, "t": 4},
            {},
            {"p": 2, "t": 1},
            {"p": 0.25, "t": 0.25},
        ),
        # Only substitutions
        (
            {"k": 6, "g": 4},
            {("k", "g"): 3, ("g", "k"): 2},
            {},
            {"k": 0.5, "g": 0.5},
        ),
        # Empty inputs
        (
            {},
            {},
            {},
            {},
        ),
        # Realistic IPA phoneme data with common confusions
        (
            {"p": 10, "b": 8, "t": 15, "d": 12, "ɹ": 20, "ə": 25},
            {("p", "b"): 2, ("t", "d"): 3, ("ɹ", "l"): 1, ("ə", "ʌ"): 2},
            {"ə": 5, "t": 2},
            {"p": 0.2, "b": 0.0, "t": 1 / 3, "d": 0.0, "ɹ": 0.05, "ə": 0.28},
        ),
        # Token appears in reference but has no errors
        (
            {"a": 5, "b": 3, "c": 2},
            {("a", "x"): 1},
            {"b": 1},
            {"a": 0.2, "b": 1 / 3, "c": 0.0},
        ),
    ],
)
def test_by_token_error_rates(token_counts, substitutions, deletions, expected_error_rates):
    """Test error rate calculation for various scenarios"""
    error_rates = calculate_by_token_error_rates(Counter(token_counts), Counter(substitutions), Counter(deletions))

    assert set(error_rates.keys()) == set(expected_error_rates.keys())

    for token, expected_rate in expected_error_rates.items():
        assert error_rates[token] == pytest.approx(expected_rate)


def test_model_evaluator(tmp_path):
    model_eval = ModelEvaluator()
    ground_truth = ["po"]
    prediction = ["bo"]

    metrics = model_eval.eval_non_empty_transcriptions("test_model", prediction, ground_truth)
    non_empty_expected = {
        "test_model": {
            "mean_phone_error_rate": 0.5,
            "mean_phone_feature_error_rate": 0.041666666666666664,
            "mean_feature_error_rate": 0.020833333333333332,
            "substitutions": {("p", "b"): 1},
            "insertions": {},
            "deletions": {},
            "by_token_error_rates": {"o": 0.0, "p": 1.0},
        }
    }
    expected_metrics_keys = set(
        [
            "mean_phone_error_rate",
            "mean_phone_feature_error_rate",
            "mean_feature_error_rate",
            "phone_error_rates",
            "phone_feature_error_rates",
            "feature_error_rates",
            "substitutions",
            "insertions",
            "deletions",
        ]
    )
    assert set(metrics.keys()) == expected_metrics_keys
    assert model_eval.results_to_write == non_empty_expected

    hallucinations = model_eval.eval_empty_transcriptions("test_model", prediction)
    assert hallucinations == {
        "panphone_phone_hallucinations": [2],
        "insertions": [{"b": 1, "o": 1}],
        "substitutions": [{}],
        "deletions": [{}],
    }
    all_expected = {
        "test_model": {
            "mean_phone_error_rate": 0.5,
            "mean_phone_feature_error_rate": 0.041666666666666664,
            "mean_feature_error_rate": 0.020833333333333332,
            "total_phone_hallucinations": 2,
            "substitutions": {("p", "b"): 1},
            "insertions": {"b": 1, "o": 1},
            "deletions": {},
            "by_token_error_rates": {"o": 0.0, "p": 1.0},
        }
    }
    assert model_eval.results_to_write == all_expected

    test_csv = tmp_path / "test.csv"
    model_eval.to_csv(test_csv)

    test_df = pd.read_csv(test_csv)
    assert test_df.shape == (1, 9)
    expected_columns = [
        "model",
        "mean_phone_error_rate",
        "mean_phone_feature_error_rate",
        "mean_feature_error_rate",
        "total_phone_hallucinations",
        "substitutions",
        "insertions",
        "deletions",
        "by_token_error_rates",
    ]
    assert set(test_df.columns) == set(expected_columns)
