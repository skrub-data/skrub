import pandas as pd

from skrub import TableVectorizer

data = pd.DataFrame(
    {"ville": ["Paris", "Lyon", "Marseille"], "population": [2148000, 513000, 861000]}
)

vectorizer = TableVectorizer()
X = vectorizer.fit_transform(data)
print(X)
