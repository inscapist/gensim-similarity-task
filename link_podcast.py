import pandas as pd
import pdb
import json
import numpy as np


df = pd.read_csv("podcasts.csv")
df = df.astype(object).replace(np.nan, "None")
podcasts = df.to_dict("records")

df = pd.read_csv("products.csv")
df = df.astype(object).replace(np.nan, "None")
products = df.to_dict("records")

print(
    json.dumps(
        {
            "podcasts": podcasts,
            "products": products,
        },
        default=str,
    )
)

# https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05
# https://github.com/massanishi/document_similarity_algorithms_experiments/blob/master/bert/process_bert_similarity.py
