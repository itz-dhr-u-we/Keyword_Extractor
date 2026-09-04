import pandas as pd
import re
import pickle
import stopwordsiso as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keybert import KeyBERT

kw_model = KeyBERT()
df = pd.read_csv("papers.csv")
df = df.iloc[:5000, :]

# Pre-processing
english_stopwords = stopwords.stopwords("en")

def pre_process(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(\d|\W)+", " ", text)
    text = text.split()
    text = [word for word in text if word not in english_stopwords and len(word) >= 3]
    return " ".join(text)

df['paper_text'] = df['paper_text'].fillna('')
docs = df['paper_text'].apply(lambda x: pre_process(str(x))).tolist()

# Vectorization & TF-IDF
cv = CountVectorizer(max_df=0.95, max_features=10000, ngram_range=(1, 3))
word_count_vector = cv.fit_transform(docs)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)
feature_names = cv.get_feature_names_out()

# TF-IDF Helper Functions
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

def get_keywords(idx, docs):
    tf_idf_vector = tfidf_transformer.transform(cv.transform([docs[idx]]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    return keywords

# KeyBERT Helper Function
def get_keybert_keywords(text, top_n=10):
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 3), 
        stop_words='english', 
        top_n=top_n
    )
    return {kw: round(score, 3) for kw, score in keywords}
    
with open('cv.pkl', 'wb') as f:
    pickle.dump(cv, f)

with open('tfidf_transformer.pkl', 'wb') as f:
    pickle.dump(tfidf_transformer, f)

print("Models successfully trained and saved as pickle files!")