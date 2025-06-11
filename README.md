# Quora-Duplicate-Question-Detection-with-Hybrid-NLP-Model-
🔹 Project Flow & Key Technical Steps\
📂 Data & Problem Understanding\
•	Processed 350K+ question pairs from Quora’s dataset, addressing class imbalance (60:40 non-duplicate/duplicate split).\
•	Engineered 22 linguistic features (lexical, syntactic, and semantic) to capture question similarity.\
⚙️ Advanced Feature Engineering\
•	Text Normalization: Contraction handling, HTML stripping, and fuzzy string matching (e.g., fuzz.ratio, token_set_ratio).\
•	Embeddings: Leveraged Word2Vec (Google News 300D) for semantic similarity via sentence vectors.\
•	Feature Fusion: Combined handcrafted features (e.g., longest_substring_ratio, cwc_min) with embeddings for richer context.\
🤖 Model Architecture & Training\
•	Hybrid Neural Network: Built a custom TensorFlow model with:\
o	Dense layers + L2 regularization for embeddings.\
o	Feature concatenation + BatchNorm/Dropout (up to 50%) to prevent overfitting.\
•	Class Weighting: Mitigated imbalance, achieving 90% recall for duplicates.\
•	Performance: 82% accuracy (74% precision, 90% recall) on test data.\
🚀 Deployment\
•	Packaged the pipeline (scaler, Word2Vec, model) using Dill and deployed via Streamlit for real-time predictions.\
•	Optimized inference speed (<1s response time) for scalable use.\
💡 Why It Matters: This project demonstrates how feature engineering + deep learning can solve nuanced NLP challenges. Perfect for content moderation or FAQ systems!\
Tools: Python, TensorFlow, NLTK, FuzzyWuzzy, Streamlit, Gensim
