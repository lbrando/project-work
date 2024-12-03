import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def keywords(text):
    max_ngram_length = 1
    top_n = 5

    vectorizer = TfidfVectorizer(
        ngram_range=(1, max_ngram_length),
        stop_words='english'
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform([text])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get TF-IDF scores
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Create keyword list with scores
    keywords = [
        feature_names[idx] 
        for idx, score in enumerate(tfidf_scores) 
        if score > 0
    ]
    
    # Sort and return top N keywords
    return sorted(keywords, key=lambda x: len(x), reverse=True)[:top_n]
    
if __name__ == "__main__":
    import pandas as pd

    dataset = pd.read_table("src/dataset/speech-a.tsv", sep="\t", names=["Surname", "Code", "Speech"])
    
    dataset["Keywords"] = dataset["Speech"].apply(keywords)

    print(dataset[["Surname", "Keywords"]])

    dataset.to_csv("src/dataset/keywords.csv", index=False)