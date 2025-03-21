import pandas as pd

from multipa.evaluate import ModelEvaluator

def test_model_evaluator(tmp_path):
    model_eval = ModelEvaluator()
    ground_truth = ["po"]
    prediction = ["bo"]

    metrics = model_eval.eval_non_empty_transcriptions("test_model", prediction, ground_truth)
    non_empty_expected = {"test_model": {
        "mean_phone_error_rate": 0.5,
        "mean_phone_feature_error_rate": 0.041666666666666664,
        "mean_feature_error_rate": 0.020833333333333332
    }}
    expected_metrics_keys = set(["mean_phone_error_rate", 
                                 "mean_phone_feature_error_rate", 
                                 "mean_feature_error_rate", 
                                 "phone_error_rates", 
                                 "phone_feature_error_rates", 
                                 "feature_error_rates"])
    assert metrics.keys() == expected_metrics_keys
    assert model_eval.results_to_write == non_empty_expected

    hallucinations = model_eval.eval_empty_transcriptions("test_model", prediction)
    assert hallucinations == [2]
    all_expected = {"test_model": {
        "mean_phone_error_rate": 0.5,
        "mean_phone_feature_error_rate": 0.041666666666666664,
        "mean_feature_error_rate": 0.020833333333333332,
        "total_phone_hallucinations": 2
    }}
    assert model_eval.results_to_write == all_expected 

    test_csv = tmp_path / "test.csv"
    model_eval.to_csv(test_csv)

    test_df = pd.read_csv(test_csv)
    assert test_df.shape == (1, 5)
    expected_columns = [
        "model", 
        "mean_phone_error_rate",
        "mean_phone_feature_error_rate",
        "mean_feature_error_rate",
        "total_phone_hallucinations"
    ]
    assert set(test_df.columns) == set(expected_columns)
        
        
        
