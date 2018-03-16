"""
base example script. the midwest survey datasets will be included in the
package because of its small size (500kb)
"""

from dirty_cat.datasets import fetch_midwest_survey
from dirty_cat.similarity_encoder import SimilarityEncoder

if __name__ == '__main__':
    data = fetch_midwest_survey().astype(str)
    se = SimilarityEncoder()
    # take only first 100 rows for speed limitations
    transformed_data = se.fit_transform(
        data.iloc[:, 1].values.reshape((-1, 1)))
