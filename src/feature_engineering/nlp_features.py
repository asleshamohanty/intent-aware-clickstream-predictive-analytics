import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER sentiment resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def extract_keywords(text, top_n=3):
    """
    Simple keyword extraction: take top N words from cleaned query
    """
    tokens = text.split()
    return " ".join(tokens[:top_n])

def extract_nlp_features(input_csv, tfidf_path="models/tfidf_vectorizer.pkl"):
    df = pd.read_csv(input_csv)

    # 1️⃣ TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df['CleanQuery'].astype(str))

    # Save TF-IDF vectorizer for later inference
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)

    # 2️⃣ Sentiment Score (VADER)
    df['Sentiment'] = df['CleanQuery'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # 3️⃣ Keyword Tags (simple top 3 words)
    df['Keywords'] = df['CleanQuery'].apply(extract_keywords)

    # 4️⃣ Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    # Save final processed dataset
    output_csv = "data/processed/synthetic_clickstream_nlp_features.csv"
    df.to_csv(output_csv, index=False)
    print(f"NLP features dataset saved to {output_csv}")

    return df

# Example usage
if __name__ == "__main__":
    extract_nlp_features("data/processed/synthetic_clickstream_clean.csv")
