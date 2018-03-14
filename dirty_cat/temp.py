import numpy as np

from similarity_encoder import SimilarityEncoder

model = SimilarityEncoder(
    similarity_type='ngram', handle_unknown='ignore')
X = np.array(['aa', 'aaa', 'aaab']).reshape(-1, 1)
X_test = np.array([['aa', 'aaa', 'aaa', 'aaab', 'aaac']]).reshape(-1, 1)
model.fit(X)
encoder = model.transform(X_test)
print(encoder)
