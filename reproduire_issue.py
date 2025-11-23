# Script pour reproduire l'issue https://github.com/skrub-data/skrub/issues/1763
# sentence_transformers devrait être non installé
# vérifié avec pip show sentence_transformers

import pandas as pd

from skrub import TextEncoder

text_enc = TextEncoder()
df = pd.DataFrame({"text": ["hello", "world"]})

# text_enc.fit_transform(df["text"])

try:
    text_enc.fit_transform(df["text"])
except ImportError as e:
    print("Message d'erreur actuel:")
    print(str(e))
