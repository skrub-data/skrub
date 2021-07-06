import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PretrainedFastText(BaseEstimator, TransformerMixin):
    """ Category embedding using a fastText pretrained model.
    
    In a nutshell, fastText learns embeddings for character n-grams based on
    their context. Sequences of words are then embedded by averaging the
    n-gram representations it is made of. FastText embeddings thus capture
    both semantic and morphological information.
    
    The code is largely based on the fasttext package, for which this class
    provides a simple interface. It is designed to download and use fastText
    models pretrained on Common Crawls and Wikipedia in 157 different
    languages and stored in the form of binary files "cc.{language}.300.bin"
    (see <https://fasttext.cc/docs/en/crawl-vectors.html>).
    
    Parameters
    ----------
    
    bin_dir : str
        The folder containing the fastText models in the form of binary files.
        Models downloaded or saved with the 'download_model' or 'save_model'
        methods are stored in bin_dir.
    
    n_components : int, default=300
        The size of the fastText embeddings (300 for the downloaded models).
        If n_components < 300, the fastText model is automatically reduced to
        output vectors of the desired size.
        If n_components > 300, it is set back to 300.
        
    language : str, default='en'
        The training language of the fastText model to load.
        See the list of models trained in 157 different languages here:
        <https://fasttext.cc/docs/en/crawl-vectors.html>.
    
    file_name : str or None, default=None
        Only used for testing purposes. If given, indicates the file to load
        instead of "cc.{language}.{n_components}.bin". This allows to load
        different fastText models that have been manually downloaded.
        If given, n_components must be set appropriately to the embedding size
        of the model.
        
    References
    ----------
    For a detailed description of the method, see
    `Enriching Word Vectors with Subword Information
    <https://arxiv.org/abs/1607.04606>`_ by Bojanowski et al. (2017).
    
    Additional information about pretrained models and the fasttext package
    can be found here <https://fasttext.cc>.
    """

    def __init__(self, bin_dir, n_components=300, language='en',
                 file_name=None):
        
        self.bin_dir = bin_dir
        self.n_components = n_components if n_components < 300 else 300
        self.language = language
        # Load the model from binary file
        if file_name == None:
            file_name = f"cc.{language}.{n_components}.bin"
        self.file_path = os.path.join(bin_dir, file_name)
        self.load_model()
        return
    
    def download_model(self):
        """
        Download pre-trained common-crawl vectors from fastText's website
        <https://fasttext.cc/docs/en/crawl-vectors.html>.
        Downloaded models are stored in self.bin_dir.
        """

        import fasttext.util
        cwd = os.getcwd()
        os.chdir(self.bin_dir)
        # Download fastText model in bin_dir
        fasttext.util.download_model(self.language, if_exists='ignore')
        os.chdir(cwd)
        return
    
    def load_model(self):
        """
        Load the binary file "cc.{language}.{n_components}.bin" in bin_dir.
        If there exists no file with the appropriate embedding dimension
        (n_components), we instead load the raw model (whose embedding dim is
        300) and reduce it to n_components afterwards.
        """
        
        import fasttext
        file_path_300 = os.path.join(
            self.bin_dir, f"cc.{self.language}.300.bin")
        # Load model with dim = n_components
        if os.path.isfile(self.file_path):
            self.ft_model = fasttext.load_model(self.file_path)
        # Otherwise, load the raw model with dim = 300 and reduce it
        elif os.path.isfile(file_path_300):
            self.ft_model = fasttext.load_model(file_path_300)
            fasttext.util.reduce_model(self.ft_model, self.n_components)
        else: # Else, raise an error
            raise FileNotFoundError(
                f"The file {self.file_path} doesn't exist and cannot be\
                reduced from the raw model {file_path_300}.\
                Download the raw model with self.download_model()\
                and load it manually with self.load_model().")
        return
    
    def save_model(self, file_name=None):
        """
        Save the current fastText model in self.bin_dir.
        This is particularly useful to save reduced models, which
        require less memory to be stored/loaded.
        
        Parameters
        ----------
        
        file_name : str or None, default=None
            If given, indicates the filename under which the model
            must be saved. By default, the default file_name is
            "cc.{self.language}.{self.n_components}.bin".
        """
        
        cwd = os.getcwd()
        os.chdir(self.bin_dir)
        # Save fastText model in bin_dir
        if file_name == None:
            file_name = f"cc.{self.language}.{self.n_components}.bin"
        self.ft_model.save_model(file_name)
        os.chdir(cwd)
        return
    
    def reduce_model(self, n_components):
        """
        Reduce the embedding dimension of the loaded fastText model.
        
        Parameters
        ----------
        
        n_components : int
            The new embedding size. Must be smaller than the previous one.
        """
        
        import fasttext.util
        assert n_components < self.n_components, f"Cannot expand embedding\
            size from {self.n_components} to {n_components}."
        self.n_components = n_components
        fasttext.util.reduce_model(self.ft_model, self.n_components)
        return
                    
    def fit(self, X=None, y=None): 
        """
        Since the model is already trained, simply checks that is has been
        loaded properly.
        """

        assert hasattr(self, 'ft_model'), f"The fastText model hasn't been\
            automatically loaded. Load it manually with self.load_model()."
        return self

    def transform(self, X):
        """
        Return fastText embeddings of input strings in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The string data to encode.

        Returns
        -------
        X_out : 2-d array, shape (n_samples, n_components)
            Transformed input.
        """
        
        # Check if a fastText model has been loaded
        assert hasattr(self, 'ft_model'), f"The fastText model hasn't been\
            automatically loaded. Load it manually with self.load_model()."
        # Check input data shape
        X = np.asarray(X)
        assert X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1), f"ERROR:\
        shape {X.shape} of input array is not supported."
        if X.ndim == 2:
            X = X[:, 0]
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "ERROR: Input data is not string."
        # Remove '\n' from X
        X = np.array([x.replace('\n', ' ') for x in X])
        # Get unique categories and store the associated embeddings in a dict
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_X_out = np.empty((len(unq_X), self.n_components))
        for idx, x in enumerate(unq_X):
            unq_X_out[idx] = self.ft_model.get_sentence_vector(x)
        X_out = unq_X_out[lookup]
        return X_out