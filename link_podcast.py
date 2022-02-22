import pandas as pd
import pdb
import json
import numpy as np
from gensim.utils import simple_preprocess as preprocess
from gensim import corpora
from gensim import models
from gensim import similarities


df = pd.read_csv("podcasts_with_transcript.csv")
df = df.astype(object).replace(np.nan, "None")
podcasts_raw = df.to_dict("records")

df = pd.read_csv("products.csv")
df = df.astype(object).replace(np.nan, "None")
products_raw = df.to_dict("records")


podcasts = [
    {
        "host": p.get("-"),
        "title": p.get("Title"),
        "doc": " ".join([p["Title"], p["transcripts"]]),
    }
    for p in podcasts_raw
]
podcast_docs = [preprocess(p["doc"]) for p in podcasts]


# pdb.set_trace()

products = [
    {
        "name": p.get("ProdName"),
        "sku": p.get("SKU"),
        "doc": " ".join([p["ProdName"], p["Desc"], p["LongDesc"]]),
    }
    for p in products_raw
]
product_docs = [preprocess(p["doc"]) for p in products]

# create common dictionary for both products and podcasts
docs = podcast_docs + product_docs
dictionary = corpora.Dictionary(docs)

# build separate corpus
podcast_corpus = [dictionary.doc2bow(doc) for doc in podcast_docs]
product_corpus = [dictionary.doc2bow(doc) for doc in product_docs]

# create podcast index
tfidf = models.TfidfModel(podcast_corpus)
podcast_index = similarities.MatrixSimilarity(tfidf[podcast_corpus])

# TEST with a sample query
query = "weight loss"
query_bow = dictionary.doc2bow(preprocess(query))
query_model = tfidf[query_bow]
sims = podcast_index[query_model]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
query_result = [(doc_score, podcasts[doc_position]) for doc_position, doc_score in sims]


def find_top_3(i, bow):
    product = products[i]
    model = tfidf[bow]
    sims = podcast_index[model]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return {
        "product": product,
        "matches": [
            (doc_score, podcasts[doc_position]) for doc_position, doc_score in sims
        ][0:3],
    }


# compute top 3 podcasts for all products
product_top3_podcasts = [find_top_3(i, bow) for i, bow in enumerate(product_corpus)]


# pdb.set_trace()


# debug excel
print(
    json.dumps(
        {
            "podcasts": podcasts,
            "products": products,
            "docs": docs,
            "dictionary": dictionary.token2id,
            "product_top3_podcasts": product_top3_podcasts,
            "query": query,
            "query_result": query_result,
        },
        default=str,
    )
)

# https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05
# https://github.com/massanishi/document_similarity_algorithms_experiments/blob/master/bert/process_bert_similarity.py
