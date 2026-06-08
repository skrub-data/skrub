from sklearn.base import BaseEstimator


class BaseTransformer(BaseEstimator):
    _doc_link_module = "skrub"

    # Defining this as a property because it inherits from _HTMLDocumentationLinkMixin,
    # which also defines _doc_link_template as a property, and we want to be able
    # to override it.
    @property
    def _doc_link_template(self):
        return getattr(
            self,
            "__doc_link_template",
            "https://skrub-data.org/stable/reference/generated/"
            "{estimator_module}.{estimator_name}.html",
        )
