import re
import pickle
from flask import Flask, render_template, request
import stopwordsiso as stopwords
from keybert import KeyBERT

app = Flask(__name__, template_folder='.')

english_stopwords = stopwords.stopwords("en")
kw_model = KeyBERT()

with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)

feature_names = cv.get_feature_names_out()

def pre_process(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(\d|\W)+", " ", text)
    text = text.split()
    text = [word for word in text if word not in english_stopwords and len(word) >= 3]
    return " ".join(text)

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
    return {feature_vals[i]: score_vals[i] for i in range(len(feature_vals))}

def get_keybert_keywords(text, top_n=10):
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 3), 
        stop_words='english', 
        top_n=top_n
    )
    return {kw: round(score, 3) for kw, score in keywords}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['message']
        mode = request.form.get('mode', 'auto')
        
        if len(text_input.strip().split()) < 3:
            return render_template('index.html', prediction="Please provide a longer text corpus.")
        
        word_count = len(text_input.split())
        
        if mode == "fast" or (mode == "auto" and word_count > 300):
            processed_doc = pre_process(text_input)
            tf_idf_vector = tfidf_transformer.transform(cv.transform([processed_doc]))
            sorted_items = sort_coo(tf_idf_vector.tocoo())
            keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
            selected_method = "TF-IDF (Statistical - Optimized for Scale)"
        else:
            keywords = get_keybert_keywords(text_input, top_n=10)
            selected_method = "KeyBERT (Semantic - Optimized for Context)"
        
        return render_template(
            'index.html', 
            prediction=keywords, 
            method_used=selected_method
        )

if __name__ == '__main__':
    app.run(debug=True)