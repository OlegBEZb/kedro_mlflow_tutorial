from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, extract_unseen_data, preprocess_companies, preprocess_shuttles


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",
                name="preprocess_companies_node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
                outputs="spaceflight_primary_table_full",
                name="create_model_input_table_node",
            ),
            node(func=extract_unseen_data,
                 inputs=["spaceflight_primary_table_full", "params:target_column"],
                 outputs="spaceflight_primary_table",
                 name="extract_unseen_data_node",
                 ),
        ]
    )
