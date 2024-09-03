import itertools

import sklearn
from sklearn.utils.fixes import parse_version

sklearn_version = parse_version(sklearn.__version__)

# TODO: remove when scikit-learn 1.6 is the minimum supported version
# TODO: subsequently, we should remove the inheritance from _HTMLDocumentationLinkMixin
# for each estimator then.
if sklearn_version > parse_version("1.6"):
    from sklearn.utils._estimator_html_repr import _HTMLDocumentationLinkMixin
else:

    class _HTMLDocumentationLinkMixin:
        """Mixin class allowing to generate a link to the API documentation."""

        _doc_link_module = "sklearn"
        _doc_link_url_param_generator = None

        @property
        def _doc_link_template(self):
            sklearn_version = parse_version(sklearn.__version__)
            if sklearn_version.dev is None:
                version_url = f"{sklearn_version.major}.{sklearn_version.minor}"
            else:
                version_url = "dev"
            return getattr(
                self,
                "__doc_link_template",
                (
                    f"https://scikit-learn.org/{version_url}/modules/generated/"
                    "{estimator_module}.{estimator_name}.html"
                ),
            )

        @_doc_link_template.setter
        def _doc_link_template(self, value):
            setattr(self, "__doc_link_template", value)

        def _get_doc_link(self):
            """Generates a link to the API documentation for a given estimator.

            This method generates the link to the estimator's documentation page
            by using the template defined by the attribute `_doc_link_template`.

            Returns
            -------
            url : str
                The URL to the API documentation for this estimator. If the estimator
                does not belong to module `_doc_link_module`, the empty string (i.e.
                `""`) is returned.
            """
            if self.__class__.__module__.split(".")[0] != self._doc_link_module:
                return ""

            if self._doc_link_url_param_generator is None:
                estimator_name = self.__class__.__name__
                # Construct the estimator's module name, up to the first private
                # submodule. This works because in scikit-learn all public estimators
                # are exposed at that level, even if they actually live in a private
                # sub-module.
                estimator_module = ".".join(
                    itertools.takewhile(
                        lambda part: not part.startswith("_"),
                        self.__class__.__module__.split("."),
                    )
                )
                return self._doc_link_template.format(
                    estimator_module=estimator_module, estimator_name=estimator_name
                )
            return self._doc_link_template.format(
                **self._doc_link_url_param_generator()
            )


doc_link_template = (
    "https://skrub-data.org/{version}/reference/generated/"
    "{estimator_module}.{estimator_name}.html"
)
doc_link_module = "skrub"


def doc_link_url_param_generator(estimator):
    from skrub import __version__

    skrub_version = parse_version(__version__)
    if skrub_version.dev is None:
        version_url = f"{skrub_version.major}.{skrub_version.minor}"
    else:
        version_url = "dev"
    estimator_name = estimator.__class__.__name__
    estimator_module = ".".join(
        itertools.takewhile(
            lambda part: not part.startswith("_"),
            estimator.__class__.__module__.split("."),
        )
    )
    return {
        "version": version_url,
        "estimator_module": estimator_module,
        "estimator_name": estimator_name,
    }
