import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean search query text:
    - lowercase
    - remove punctuation
    - remove stopwords
    - lemmatize words
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation/special chars
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # lemmatize
    return " ".join(tokens)

def clean_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['CleanQuery'] = df['SearchQuery'].apply(clean_text)
    df.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    clean_dataset("data/raw/synthetic_clickstream_nlp.csv",
                  "data/processed/synthetic_clickstream_clean.csv")
