# Keyword Extraction

An end-to-end Natural Language Processing web application that contrasts traditional statistical frequency-based extraction with modern deep-learning semantic embeddings.

## Key Features
- Adaptive Routing Engine: Automatically routes text processing based on length and scale requirements.
- Statistical Pipeline (TF-IDF): Utilizes CountVectorizer and TfidfTransformer for rapid exact-match term frequency scoring.
- Semantic Pipeline (KeyBERT): Employs pre-trained transformer embeddings (all-MiniLM-L6-v2) via sentence-transformers to capture deep contextual meaning.
- Full-Stack Integration: Built with a clean HTML/Flask architecture and model serialization.

## Tech Stack
- Python, Flask (Backend Web Framework)
- Scikit-Learn, NLTK (Statistical NLP & Vectorization)
- KeyBERT, Hugging Face Transformers (Deep Learning Semantic Extraction)

## Local Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/keyword-extractor.git](https://github.com/your-username/keyword-extractor.git)
   cd keyword-extractor
2. Create and activate a virtual environment:
   
    python -m venv venv
    source venv/Scripts/activate
4. Install dependencies:
   
    pip install flask pandas scikit-learn keybert sentence-transformers stopwordsiso
6. Run the application:
   
    python app.py
