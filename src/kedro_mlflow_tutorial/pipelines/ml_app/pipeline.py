from kedro.pipeline import Pipeline, node, pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory
from kedro_mlflow.pipeline.pipeline_ml import PipelineML

from .nodes import (
    evaluate_model,
    extract_target,
    fit_numerical_features_scaler,
    generate_numerical_features,
    predict_node,
    select_training_columns,
    split_data,
    train_model,
    transform_numerical_features,
)


def create_preprocessing_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=extract_target,
            inputs=["spaceflight_primary_table", "params:target_column"],
            outputs="model_input_target",
            name="extract_target_node",
            tags=["training"],
        ),

        node(
            func=split_data,
            inputs=["spaceflight_primary_table", "model_input_target", "params:split_parameters"],
            outputs=["X_train_raw", "X_test_raw", "y_train", "y_test"],
            name="split_data_node",
            tags=["training"],
        ),

        node(
            func=select_training_columns,
            inputs=["X_train_raw", "params:features"],
            outputs="X_train",
            name="select_train_columns_node",
            tags=["training"],
        ),

        node(
            func=select_training_columns,
            inputs=["X_test_raw", "params:features"],
            outputs="X_test",
            name="select_test_columns_node",
            tags=["training", "inference"],
        ),

        node(
            func=fit_numerical_features_scaler,
            inputs="X_train",
            outputs="fitted_scaler",
            name="fit_numerical_scaler_node",
            tags=["training"],
        ),  # can't be done on the whole dataset

        node(
            func=transform_numerical_features,
            inputs=["X_train", "fitted_scaler"],
            outputs="X_train_scaled",
            name="transform_train_node",
            tags=["training"],
        ),
        node(
            func=transform_numerical_features,
            inputs=["X_test", "fitted_scaler"],
            outputs="X_test_scaled",
            name="transform_test_node",
            tags=["training", "inference"],
        ),

        node(
            func=generate_numerical_features,
            inputs=["X_train_scaled"],
            outputs="X_train_features",
            name="generate_train_features_node",
            tags=["training"],
        ),
        node(
            func=generate_numerical_features,
            inputs=["X_test_scaled"],
            outputs="X_test_features",
            name="generate_test_features_node",
            tags=["training", "inference"],
        ),
    ])


def create_training_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["X_train_features", "y_train", "params:model_params"],
            outputs="regressor",
            name="train_model_node",
            tags=["training"],
        ),
        node(
            func=evaluate_model,
            inputs=["regressor", "X_test_features", "y_test"],
            outputs="metrics",
            name="evaluate_model_node",
            tags=["training"],
        ),
    ])


def create_inference_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=predict_node,
            inputs=["regressor", "X_test_features"],
            outputs="predictions",
            name="inference_node",
            tags=["inference"],
        ),
    ])


def create_pipeline(**kwargs) -> PipelineML:
    preprocessing_pipeline = create_preprocessing_pipeline()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()

    training_pipeline_with_preprocessing = (
        preprocessing_pipeline + training_pipeline
    ).only_nodes_with_tags("training")

    inference_pipeline_with_preprocessing = (
        preprocessing_pipeline + inference_pipeline
    ).only_nodes_with_tags("inference")

    pipeline_ml = pipeline_ml_factory(
        training=training_pipeline_with_preprocessing,
        inference=inference_pipeline_with_preprocessing,
        input_name="X_test_raw",
    )

    return pipeline_ml
