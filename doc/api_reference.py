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
                    "See :ref:`End-to-End pipeline"
                    " <user_guide_building_pipeline_index>` for further details. For"
                    " more flexibility and control to build pipelines, see the"
                    " :ref:`skrub DataOps <user_guide_data_ops_index>`."
                ),
                "autosummary": [
                    "tabular_pipeline",
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
                "description": (
                    "See :ref:`encoding <user_guide_encoders_index>` for further"
                    " details."
                ),
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
                    "SquashingScaler",
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
                    "Contains method to select columns in a dataframe. See the"
                    " :ref:`selectors <user_guide_selectors>` section for further"
                    " details."
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
    "data_ops": {
        "title": "DataOps",
        "short_summary": None,
        "description": None,
        "sections": [
            {
                "description": (
                    "Generalizing the scikit-learn pipeline. See :ref:`skrub DataOps"
                    " <user_guide_data_ops_index>` for further details."
                ),
                "autosummary": ["var", "X", "y", "as_data_op", "deferred"],
                "template": "base.rst",
            },
            {
                "description": "The DataOp object.",
                "autosummary": ["DataOp"],
                "template": "data_op_class.rst",
            },
            {
                "description": "Inline hyperparameters selection in your DataOps plan.",
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
                "description": "Evaluate your DataOps plan.",
                "autosummary": ["cross_validate", "eval_mode"],
                "template": "base.rst",
            },
            {
                "description": (
                    "The ``skb`` accessor exposes all DataOps methods and attributes."
                ),
                "autosummary": [
                    "DataOp.skb.apply",
                    "DataOp.skb.apply_func",
                    "DataOp.skb.clone",
                    "DataOp.skb.concat",
                    "DataOp.skb.cross_validate",
                    "DataOp.skb.describe_defaults",
                    "DataOp.skb.describe_param_grid",
                    "DataOp.skb.describe_steps",
                    "DataOp.skb.draw_graph",
                    "DataOp.skb.drop",
                    "DataOp.skb.eval",
                    "DataOp.skb.freeze_after_fit",
                    "DataOp.skb.full_report",
                    "DataOp.skb.get_data",
                    "DataOp.skb.make_learner",
                    "DataOp.skb.make_grid_search",
                    "DataOp.skb.make_randomized_search",
                    "DataOp.skb.if_else",
                    "DataOp.skb.iter_learners_grid",
                    "DataOp.skb.iter_learners_randomized",
                    "DataOp.skb.mark_as_X",
                    "DataOp.skb.mark_as_y",
                    "DataOp.skb.match",
                    "DataOp.skb.preview",
                    "DataOp.skb.select",
                    "DataOp.skb.set_description",
                    "DataOp.skb.set_name",
                    "DataOp.skb.subsample",
                    "DataOp.skb.train_test_split",
                ],
                "template": "autosummary/accessor_method.rst",
            },
            {
                "description": "Accessor attributes.",
                "autosummary": [
                    "DataOp.skb.description",
                    "DataOp.skb.is_X",
                    "DataOp.skb.is_y",
                    "DataOp.skb.name",
                    "DataOp.skb.applied_estimator",
                ],
                "template": "autosummary/accessor_attribute.rst",
            },
            {
                "description": "Objects generated by the DataOps.",
                "autosummary": [
                    "SkrubLearner",
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
