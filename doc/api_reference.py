"""Configuration for the API reference documentation."""


def _get_guide(*refs, is_developer=False):
    """Get the rst to refer to user/developer guide.

    `refs` is several references that can be used in the :ref:`...` directive.
    """
    if len(refs) == 1:
        ref_desc = f":ref:`{refs[0]}` section"
    elif len(refs) == 2:
        ref_desc = f":ref:`{refs[0]}` and :ref:`{refs[1]}` sections"
    else:
        ref_desc = ", ".join(f":ref:`{ref}`" for ref in refs[:-1])
        ref_desc += f", and :ref:`{refs[-1]}` sections"

    guide_name = "Developer" if is_developer else "User"
    return f"**{guide_name} guide.** See the {ref_desc} for further details."


def _get_submodule(module_name, submodule_name):
    """Get the submodule docstring and automatically add the hook.

    `module_name` is e.g. `sklearn.feature_extraction`, and `submodule_name` is e.g.
    `image`, so we get the docstring and hook for `sklearn.feature_extraction.image`
    submodule. `module_name` is used to reset the current module because autosummary
    automatically changes the current module.
    """
    lines = [
        f".. automodule:: {module_name}.{submodule_name}",
        f".. currentmodule:: {module_name}",
    ]
    return "\n\n".join(lines)


"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to a dictionary that consists of the following
components:

short_summary (required)
    The text to be printed on the index page; it has nothing to do the API reference
    page of each module.
description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_name }}                                                           |
|     =================                                                           |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_title_1 }}   <-------------- Optional if one wants the first     |
|     ---------------------                   section to directly follow          |
|     {{ section_description_1 }}             without a second-level heading.     |
|     {{ section_autosummary_1 }}                                                 |
|                                                                                 |
|     {{ section_title_2 }}                                                       |
|     ---------------------                                                       |
|     {{ section_description_2 }}                                                 |
|     {{ section_autosummary_2 }}                                                 |
|                                                                                 |
|     More sections...                                                            |
|---------------------------------------------------------------------------------|

Hooks will be automatically generated for each module and each section. For a module,
e.g., `sklearn.feature_extraction`, the hook would be `feature_extraction_ref`; for a
section, e.g., "From text" under `sklearn.feature_extraction`, the hook would be
`feature_extraction_ref-from-text`. However, note that a better way is to refer using
the :mod: directive, e.g., :mod:`sklearn.feature_extraction` for the module and
:mod:`sklearn.feature_extraction.text` for the section. Only in case that a section
is not a particular submodule does the hook become useful, e.g., the "Loaders" section
under `sklearn.datasets`.
"""

API_REFERENCE = {
    "skrub": {
        "short_summary": "Column-wise encoders",
        "description": "",
        "sections": [
            {
                "title": "Building a pipeline",
                "description": (
                    "See :ref:`end_to_end_pipeline <end_to_end_pipeline>` for "
                    "further details. For more flexibility and control to build "
                    "pipelines, see the :ref:`skrub expressions <expressions_ref>`."
                ),
                "autosummary": [
                    "tabular_learner",
                    "TableVectorizer",
                    "Cleaner",
                    "SelectCols",
                    "DropCols",
                    "DropUninformative",
                ],
            },
            {
                "title": "Encoding a column",
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
            },
            {
                "title": "Generating an HTML report",
                "description": None,
                "autosummary": [
                    "TableReport",
                    "patch_display",
                    "unpatch_display",
                    "column_associations",
                ],
            },
            {
                "title": "Cleaning a dataframe",
                "description": None,
                "autosummary": [
                    "deduplicate",
                ],
            },
            {
                "title": "Joining dataframes",
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
    "skrub.selectors": {
        "short_summary": "Selecting columns in a DataFrame",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "all",
                    "any_date",
                    "boolean",
                    "cardinality_below",
                    "categorical",
                    "cols",
                    "Filter",
                    "filter",
                    "filter_names",
                    "float",
                    "glob",
                    "has_nulls",
                    "integer",
                    "inv",
                    "make_selector",
                    "NameFilter",
                    "numeric",
                    "regex",
                    "select",
                    "Selector",
                    "string",
                ],
            }
        ],
    },
    "expressions": {
        "short_summary": "",
        "description": (
            "Generalization the scikit-learn pipeline. "
            "See :ref:`skrub expression <skrub_pipeline>` for further details."
        ),
        "autosummary": [
            {
                "title": "Creating expressions",
                "description": None,
                "autosummary": ["var", "X", "y", "as_expr", "deferred"],
            },
            {
                "title": "Hyperparameters choices",
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
            },
            {
                "title": "Evaluate your expressions",
                "description": None,
                "autosummary": ["cross_validate", "train_test_split", "eval_mode"],
            },
            {
                "title": "Working with expressions",
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
            },
        ],
    },
}
