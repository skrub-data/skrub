import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from dirty_cat import PretrainedFastText

# Change working directory to this file's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def download_test_model():
    import urllib.request
    url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/dbpedia.ftz'
    urllib.request.urlretrieve(url, './dbpedia.ftz')
    return    

def test_pretrained_fastText(n_samples=70):
    
    download_test_model()
    X_txt = fetch_20newsgroups(subset='train')['data']
    X = X_txt[:n_samples]
    bin_dir, file_name = '.', 'dbpedia.ftz'
    enc = PretrainedFastText(bin_dir, n_components=10, file_name=file_name)
    X_enc = enc.transform(X)
    assert X_enc.shape == (n_samples, 10), str(X_enc.shape)
    os.remove(file_name) # Remove test model
    return

def test_input_type():
    
    download_test_model()
    bin_dir, file_name = '.', 'dbpedia.ftz'
    # Numpy array
    X = np.array(['alice', 'bob'])
    enc = PretrainedFastText(bin_dir, n_components=10, file_name=file_name)
    X_enc_array = enc.transform(X)
    # List
    X = ['alice', 'bob']
    enc = PretrainedFastText(bin_dir, n_components=10, file_name=file_name)
    X_enc_list = enc.fit_transform(X)
    # Check if the encoded vectors are the same
    np.testing.assert_array_equal(X_enc_array, X_enc_list)
    os.remove(file_name) # Remove test model
    return

def test_save_model(n_samples=70):
    
    download_test_model()
    X_txt = fetch_20newsgroups(subset='train')['data']
    X = X_txt[:n_samples]
    bin_dir, file_name = '.', 'dbpedia.ftz'
    saved_file_name = 'dbpedia2.ftz'
    # Save model
    enc = PretrainedFastText(bin_dir, n_components=10, file_name=file_name)
    X_enc = enc.transform(X)
    enc.save_model(saved_file_name)
    # Check if file exists
    saved_file_path = os.path.join(bin_dir, saved_file_name)
    assert os.path.isfile(saved_file_path), saved_file_path
    # Load saved model
    enc2 = PretrainedFastText(
        bin_dir, n_components=10, file_name=saved_file_name)
    X_enc2 = enc2.transform(X)
    np.testing.assert_array_equal(X_enc, X_enc2)
    # Delete saved model
    os.remove(saved_file_path)
    os.remove(file_name) # Remove test model
    return

# Quantized models like "dbpedia.ftz" cannot be reduced.
# def test_reduce_model(reduced_n_components=5, n_samples=70):
    
#     X_txt = fetch_20newsgroups(subset='train')['data']
#     X = X_txt[:n_samples]
#     bin_dir, file_name = '../data/fasttext/', 'dbpedia.ftz'
#     # Use 'reduce_model' method
#     enc = PretrainedFastText(
#         n_components=10, bin_dir=bin_dir, file_name=file_name)
#     enc.reduce_model(reduced_n_components)
#     X_enc = enc.transform(X)
#     assert enc.n_components == reduced_n_components, str(enc.n_components)
#     assert X_enc.shape == (n_samples, reduced_n_components), str(X_enc.shape)
#     return

if __name__ == '__main__':
    
    # Run test functions
    for test_function in (test_pretrained_fastText, test_input_type,
                          test_save_model):
        
        print(f'start {test_function.__name__}')
        test_function()
        print(f'{test_function.__name__} passed')