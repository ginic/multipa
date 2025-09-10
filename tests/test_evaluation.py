import pandas as pd
import pytest

from multipa.evaluation import ModelEvaluator, compute_edit_distance_errors


@pytest.mark.parametrize(
    "ref,pred,subs,dels,inserts",
    [
        ("po", "bo", {("p", "b"): 1}, {}, {}),
        ("dʒɹɪŋk", "dɹɪnkə", {("ŋ", "n"): 1}, {"ʒ": 1}, {"ə": 1}),
        ("tuːt", "", {}, {"t": 2, "uː": 1}, {}),
        ("", "tuːt", {}, {}, {"t": 2, "uː": 1}),
    ],
)
def test_compute_edit_distance_errors(ref, pred, subs, dels, inserts):
    actual_subs, actual_dels, actual_inserts = compute_edit_distance_errors(pred, ref)
    assert actual_subs == subs
    assert actual_dels == dels
    assert actual_inserts == inserts


@pytest.mark.parametrize(
    "ref,pred,subs,dels,inserts",
    [
        ("po", "bo", {("p", "b"): 1}, {}, {}),
        ("dʒɹɪŋk", "dɹɪnkə", {("ŋ", "n"): 1}, {"ʒ": 1}, {"ə": 1}),
        ("tuːt", "", {}, {"t": 2, "u": 1, "ː": 1}, {}),
        ("", "tuːt", {}, {}, {"t": 2, "u": 1, "ː": 1}),
    ],
)
def test_computes_edit_distance_no_ipa_tok(ref, pred, subs, dels, inserts):
    actual_subs, actual_dels, actual_inserts = compute_edit_distance_errors(pred, ref, use_ipa_tokenise=False)
    assert actual_subs == subs
    assert actual_dels == dels
    assert actual_inserts == inserts


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
        }
    }
    assert model_eval.results_to_write == all_expected

    test_csv = tmp_path / "test.csv"
    model_eval.to_csv(test_csv)

    test_df = pd.read_csv(test_csv)
    assert test_df.shape == (1, 8)
    expected_columns = [
        "model",
        "mean_phone_error_rate",
        "mean_phone_feature_error_rate",
        "mean_feature_error_rate",
        "total_phone_hallucinations",
        "substitutions",
        "insertions",
        "deletions",
    ]
    assert set(test_df.columns) == set(expected_columns)
