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
    "encoder": {
        "short_summary": "Column-wise encoders",
        "description": None,
        "sections": [
            {
                "title": "Encoding a column",
                "autosummary": [
                    "StringEncoder",
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
                "title": "Deep learning",
                "description": """
    These encoders require installing additional dependencies around torch.
    See the "deep learning dependencies" section in the :ref:`installation_instructions`
    guide for more details.""",
                "autosummary": [
                    "TextEncoder",
                ],
            },
        ],
    },
    # "selectors": {
    #     "short_summary": "Selecting columns in a DataFrame",
    #     "description": (
    #         """
    # The srkub selectors provide a flexible way to specify the columns on which a
    # transformation should be applied. They are meant to be used for the ``cols``
    # argument of :meth:`Expr.skb.apply`, :meth:`Expr.skb.select`,
    # :meth:`Expr.skb.drop`, :class:`SelectCols` or :class:`DropCols`."""
    #     ),
    #     "sections": [
    #         "selectors.all",
    #         "selectors.any_date",
    #         "selectors.boolean",
    #         "selectors.cardinality_below",
    #         "selectors.categorical",
    #         "selectors.cols",
    #         "selectors.Filter",
    #         "selectors.filter",
    #         "selectors.filter_names",
    #         "selectors.float",
    #         "selectors.glob",
    #         "selectors.has_nulls",
    #         "selectors.integer",
    #         "selectors.inv",
    #         "selectors.make_selector",
    #         "selectors.NameFilter",
    #         "selectors.numeric",
    #         "selectors.regex",
    #         "selectors.select",
    #         "selectors.Selector",
    #         "selectors.string",
    #     ],
    # },
}
