from collections import Counter

import pandas as pd
import pytest

from multipa.evaluation import (
    ModelEvaluator,
    compute_edit_distance_errors,
    calculate_by_token_error_rates,
    get_token_confusion_matrix,
    EPS,
)


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


@pytest.mark.parametrize(
    "substitutions,deletions,insertions,true_token_counts,default_keys,empty_symbol,expected_rows",
    [
        # Test 1: Perfect predictions - no errors
        (
            Counter(),
            Counter(),
            Counter(),
            Counter({"p": 5, "t": 3, "k": 2}),
            None,
            EPS,
            [
                ("p", "p", 5),
                ("t", "t", 3),
                ("k", "k", 2),
            ],
        ),
        # Test 2: Only substitutions
        (
            Counter({("p", "b"): 2, ("t", "d"): 1}),
            Counter(),
            Counter(),
            {"p": 5, "t": 3},
            None,
            EPS,
            [
                ("p", "p", 3),  # 5 total - 2 substituted
                ("p", "b", 2),  # 2 substitutions
                ("t", "t", 2),  # 3 total - 1 substituted
                ("t", "d", 1),  # 1 substitution
            ],
        ),
        # Test 3: Only deletions
        (
            Counter(),
            Counter({"ə": 3, "ɹ": 1}),
            Counter(),
            {"ə": 10, "ɹ": 5, "t": 2},
            None,
            EPS,
            [
                ("ə", "ə", 7),  # 10 total - 3 deleted
                ("ə", EPS, 3),  # 3 deletions
                ("ɹ", "ɹ", 4),  # 5 total - 1 deleted
                ("ɹ", EPS, 1),  # 1 deletion
                ("t", "t", 2),  # no errors
            ],
        ),
        # Test 4: Only insertions
        (
            Counter(),
            Counter(),
            Counter({"x": 2, "y": 1}),
            {"p": 3, "t": 2},
            None,
            EPS,
            [
                ("p", "p", 3),
                ("t", "t", 2),
                (EPS, "x", 2),  # 2 insertions
                (EPS, "y", 1),  # 1 insertion
            ],
        ),
        # Test 5: Mixed errors
        (
            Counter({("p", "b"): 1, ("ð", "d"): 2}),
            Counter({"t": 1, "ə": 2}),
            Counter({"ʌ": 1, "ɪ": 1}),
            {"p": 4, "ð": 5, "t": 3, "ə": 8},
            None,
            EPS,
            [
                ("p", "p", 3),  # 4 - 1 substituted
                ("p", "b", 1),
                ("ð", "ð", 3),  # 5 - 2 substituted
                ("ð", "d", 2),
                ("t", "t", 2),  # 3 - 1 deleted
                ("t", EPS, 1),
                ("ə", "ə", 6),  # 8 - 2 deleted
                ("ə", EPS, 2),
                (EPS, "ʌ", 1),
                (EPS, "ɪ", 1),
            ],
        ),
        # Test 6: Empty inputs
        (
            Counter(),
            Counter(),
            Counter(),
            {},
            None,
            EPS,
            [],
        ),
        # Test 7: Using default_keys parameter
        (
            Counter({("a", "b"): 1}),
            Counter(),
            Counter(),
            {"a": 3},
            ["a", "c", "d"],  # Include tokens not in true_token_counts
            EPS,
            [
                ("a", "a", 2),  # 3 - 1 substituted
                ("a", "b", 1),
                ("c", "c", 0),  # From default_keys, no occurrences
                ("d", "d", 0),  # From default_keys, no occurrences
            ],
        ),
        # Test 8: Custom empty_token_symbol
        (
            Counter(),
            Counter({"p": 1}),
            Counter({"x": 1}),
            {"p": 5, "t": 2},
            None,
            "<NULL>",  # Custom empty symbol
            [
                ("p", "p", 4),
                ("p", "<NULL>", 1),
                ("t", "t", 2),
                ("<NULL>", "x", 1),
            ],
        ),
        # Test 9: IPA phonemes with realistic errors
        (
            Counter({("θ", "f"): 2, ("ð", "d"): 3}),
            Counter({"ɹ": 1}),
            Counter({"ə": 2}),
            {"θ": 10, "ð": 12, "ɹ": 8, "æ": 5},
            None,
            EPS,
            [
                ("θ", "θ", 8),  # 10 - 2 substituted
                ("θ", "f", 2),
                ("ð", "ð", 9),  # 12 - 3 substituted
                ("ð", "d", 3),
                ("ɹ", "ɹ", 7),  # 8 - 1 deleted
                ("ɹ", EPS, 1),
                ("æ", "æ", 5),  # no errors
                (EPS, "ə", 2),
            ],
        ),
        # Test 10: All correct values are substituted
        (
            Counter({("p", "b"): 5}),
            Counter(),
            Counter(),
            {"p": 5},  # All instances substituted
            None,
            EPS,
            [
                ("p", "p", 0),
                ("p", "b", 5),
            ],
        )

    ],
)
def test_get_token_confusion_matrix(
    substitutions, deletions, insertions, true_token_counts, default_keys, empty_symbol, expected_rows
):
    df = get_token_confusion_matrix(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        true_token_counts=true_token_counts,
        default_keys=default_keys,
        empty_token_symbol=empty_symbol,
    )

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["reference", "predicted", "count"]

    # Check number of rows
    assert len(df) == len(expected_rows), f"Expected {len(expected_rows)} rows, got {len(df)}"

    # Convert DataFrame to set of tuples for comparison (order doesn't matter)
    actual_rows = set(df.itertuples(index=False, name=None))
    expected_rows_set = set(expected_rows)

    assert actual_rows == expected_rows_set, f"Rows mismatch.\nExpected: {expected_rows_set}\nActual: {actual_rows}"