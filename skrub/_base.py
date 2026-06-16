from sklearn.base import BaseEstimator


class SkrubBaseTransformer(BaseEstimator):
    """Base class for all skrub estimators.

    This is a class that all skrub transformers inherit from.
    For the moment, it's only used to set the documentation url for estimator diagrams.
    
    Think twice before adding anything to this class: it is a base class of *all* skrub estimators, including meta-estimators like ApplyToCols, the SingleColumnTransformer base class, and the SkrubLearners created by DataOps.
    """

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
