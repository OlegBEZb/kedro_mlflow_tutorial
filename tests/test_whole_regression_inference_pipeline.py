import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models.utils import validate_schema


@pytest.fixture(params=[
    'models:/spaceflight_price_predictor@challenger',
    'models:/spaceflight_price_predictor@champion',
])
def model_uri(request):
    """Fixture to provide the MLflow model URI after verifying its existence."""
    model_uri = request.param

    # Check if the model exists in MLflow
    try:
        mlflow.pyfunc.load_model(model_uri)
    except MlflowException as e:
        if 'Registered model alias' in str(e) or 'not found' in str(e):
            pytest.skip(f"Model '{model_uri}' not found or alias missing. Skipping...")
        else:
            pytest.fail(f"Unexpected MlflowException for model '{model_uri}': {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error while checking model '{model_uri}': {e}")

    return model_uri


@pytest.fixture
def loaded_model(model_uri):
    """Fixture to load the MLflow model."""
    return mlflow.pyfunc.load_model(model_uri)


@pytest.fixture
def real_unseen_input_df():
    return pd.read_parquet('tests/unseen_data.parquet')


def test_pipeline_inference_on_real_unseen_input(loaded_model, real_unseen_input_df):
    input_schema = loaded_model.metadata.get_input_schema()

    # Check input schema
    validate_schema(real_unseen_input_df, input_schema)

    predictions = loaded_model.predict(real_unseen_input_df)

    # Check predictions
    assert predictions is not None, "Model returned no predictions."
    assert len(predictions) == len(real_unseen_input_df), "Mismatch between input and prediction row counts."
    assert predictions.dtype in [int, float], "Predictions must be numeric."
    assert predictions.min() >= 0, "Predictions should be non-negative."
    assert not np.isnan(predictions).any(), "Predictions should not contain NaN values."
