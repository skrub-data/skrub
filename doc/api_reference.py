"""Configuration for the API reference documentation.

This file controls the content and ordering of the API reference section of the
documentation. Edit this file to edit the documentation.

CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to a dictionary that consists of the following
components:

title (required)
    The section title. It can't be `None`.
short_summary (required)
    The text to be printed on the index page; it has nothing to do the API reference
    page of each module.
description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_title }}                                                          |
|     ==================                                                          |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_description_1 }}                                                 |
|     {{ section_autosummary_1 }}                                                 |
|                                                                                 |
|     {{ section_description_2 }}                                                 |
|     {{ section_autosummary_2 }}                                                 |
|                                                                                 |
|     More sections...                                                            |
|---------------------------------------------------------------------------------|

Hooks will be automatically generated for each module and each section. For a module,
e.g., `skrub.selectors`, the hook would be `selectors_ref`; for a
section, e.g., "Building a pipeline" under `skrub`, the hook would be
`_skrub_ref-building-a-pipeline`. However, note that a better way is to refer using
the :mod: directive, e.g., :mod:`skrub.selectors` for the module and
:mod:`skrub.text` for the section.
"""

API_REFERENCE = {
    "pipeline": {
        "title": "Building a pipeline",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": (
                    "See :ref:`end_to_end_pipeline <end_to_end_pipeline>` for "
                    "further details. For more flexibility and control to build "
                    "pipelines, see the :ref:`skrub expressions <expressions_ref>`."
                ),
                "autosummary": [
                    "tabular_pipeline",
                    "tabular_learner",
                    "TableVectorizer",
                    "SelectCols",
                    "DropCols",
                    "ApplyToCols",
                    "ApplyToFrame",
                ],
            }
        ],
    },
    "encoders": {
        "title": "Encoding a column",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": "See :ref:`encoding <encoding>` for further details.",
                "autosummary": [
                    "StringEncoder",
                    "TextEncoder",
                    "MinHashEncoder",
                    "GapEncoder",
                    "SimilarityEncoder",
                    "ToCategorical",
                    "DatetimeEncoder",
                    "ToDatetime",
                    "to_datetime",
                ],
            }
        ],
    },
    "reporting": {
        "title": "Exploring a dataframe",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": None,
                "autosummary": [
                    "TableReport",
                    "patch_display",
                    "unpatch_display",
                    "column_associations",
                ],
            }
        ],
    },
    "cleaning": {
        "title": "Cleaning a dataframe",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": None,
                "autosummary": [
                    "deduplicate",
                    "Cleaner",
                    "DropUninformative",
                ],
            },
        ],
    },
    "joining": {
        "title": "Joining dataframes",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": None,
                "autosummary": [
                    "Joiner",
                    "AggJoiner",
                    "MultiAggJoiner",
                    "AggTarget",
                    "InterpolationJoiner",
                    "fuzzy_join",
                ],
            },
        ],
    },
    "selectors": {
        "title": "Selectors",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": (
                    "Contains method to select columns in a dataframe. "
                    "See the :ref:`selectors <selectors>` section for further details."
                ),
                "autosummary": [
                    "selectors.all",
                    "selectors.any_date",
                    "selectors.boolean",
                    "selectors.cardinality_below",
                    "selectors.categorical",
                    "selectors.cols",
                    "selectors.filter",
                    "selectors.filter_names",
                    "selectors.float",
                    "selectors.glob",
                    "selectors.has_nulls",
                    "selectors.integer",
                    "selectors.inv",
                    "selectors.make_selector",
                    "selectors.numeric",
                    "selectors.regex",
                    "selectors.select",
                    "selectors.string",
                ],
            }
        ],
    },
    "expressions": {
        "title": "Expressions",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": (
                    "Generalizing the scikit-learn pipeline. "
                    "See :ref:`skrub expression <skrub_pipeline>` for further details."
                ),
                "autosummary": ["var", "X", "y", "as_expr", "deferred"],
                "template": "base.rst",
            },
            {
                "description": "The expression object.",
                "autosummary": ["Expr"],
                "template": "expr_class.rst",
            },
            {
                "description": (
                    "Inline hyperparameters selection within your expressions."
                ),
                "autosummary": [
                    "choose_bool",
                    "choose_float",
                    "choose_int",
                    "choose_from",
                    "optional",
                ],
                "template": "base.rst",
            },
            {
                "description": "Evaluate your expressions.",
                "autosummary": ["cross_validate", "eval_mode"],
                "template": "base.rst",
            },
            {
                "description": (
                    "The ``skb`` accessor exposes all expressions methods and "
                    "attributes."
                ),
                "autosummary": [
                    "Expr.skb.apply",
                    "Expr.skb.apply_func",
                    "Expr.skb.clone",
                    "Expr.skb.concat",
                    "Expr.skb.cross_validate",
                    "Expr.skb.describe_defaults",
                    "Expr.skb.describe_param_grid",
                    "Expr.skb.describe_steps",
                    "Expr.skb.draw_graph",
                    "Expr.skb.drop",
                    "Expr.skb.eval",
                    "Expr.skb.freeze_after_fit",
                    "Expr.skb.full_report",
                    "Expr.skb.get_data",
                    "Expr.skb.get_pipeline",
                    "Expr.skb.get_grid_search",
                    "Expr.skb.get_randomized_search",
                    "Expr.skb.if_else",
                    "Expr.skb.iter_pipelines_grid",
                    "Expr.skb.iter_pipelines_randomized",
                    "Expr.skb.mark_as_X",
                    "Expr.skb.mark_as_y",
                    "Expr.skb.match",
                    "Expr.skb.preview",
                    "Expr.skb.select",
                    "Expr.skb.set_description",
                    "Expr.skb.set_name",
                    "Expr.skb.subsample",
                    "Expr.skb.train_test_split",
                ],
                "template": "autosummary/accessor_method.rst",
            },
            {
                "description": "Accessor attributes.",
                "autosummary": [
                    "Expr.skb.description",
                    "Expr.skb.is_X",
                    "Expr.skb.is_y",
                    "Expr.skb.name",
                    "Expr.skb.applied_estimator",
                ],
                "template": "autosummary/accessor_attribute.rst",
            },
            {
                "description": "Objects generated by the expressions.",
                "autosummary": [
                    "SkrubPipeline",
                    "ParamSearch",
                ],
                "template": "base.rst",
            },
        ],
    },
    "config": {
        "title": "Configuration",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": None,
                "autosummary": [
                    "get_config",
                    "set_config",
                    "config_context",
                ],
            }
        ],
    },
    "datasets": {
        "title": "Datasets",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": "Downloading a dataset.",
                "autosummary": [
                    "datasets.fetch_bike_sharing",
                    "datasets.fetch_country_happiness",
                    "datasets.fetch_credit_fraud",
                    "datasets.fetch_drug_directory",
                    "datasets.fetch_employee_salaries",
                    "datasets.fetch_flight_delays",
                    "datasets.fetch_ken_embeddings",
                    "datasets.fetch_ken_table_aliases",
                    "datasets.fetch_ken_types",
                    "datasets.fetch_medical_charge",
                    "datasets.fetch_midwest_survey",
                    "datasets.fetch_movielens",
                    "datasets.fetch_open_payments",
                    "datasets.fetch_toxicity",
                    "datasets.fetch_traffic_violations",
                    "datasets.fetch_videogame_sales",
                    "datasets.get_data_dir",
                    "datasets.make_deduplication_data",
                    "datasets.toy_orders",
                ],
            }
        ],
    },
}
