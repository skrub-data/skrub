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

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        # This method should be overridden by subclasses. We raise an error here to
        # make it clear to users that they need to implement this method if they are
        # creating a custom transformer class. We also catch the error in check_output
        # to provide a more informative error message if the output of transform has the
        # wrong type.
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the 'transform' method."
        )
