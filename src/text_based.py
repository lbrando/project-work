from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import textstat
from textblob import TextBlob

# Scarica le risorse di NLTK necessarie in background
nltk.download('punkt', quiet=True)

def is_formal(speech):
    if len(speech) <= 150:
        return "N/A"
    
    blob = TextBlob(speech)
    # Bassa soggettività indica formalità
    if blob.sentiment.subjectivity < 0.5:
        return "Formal"
    else:
        return "Informal"

# Funzione per calcolare metriche di leggibilità
def calcola_legibilita(speech):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(speech), # facilità di lettura basandosi sulla lunghezza delle frasi e sul numero di sillabe per parola
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(speech), #  anni di istruzione necessari per comprendere il testo
        "gunning_fog": textstat.gunning_fog(speech), # complessità del speech in base alla lunghezza delle frasi e alla percentuale di parole complesse
        "smog_index": textstat.smog_index(speech), # come quello precedente
        "text_standard": textstat.text_standard(speech) # sintetizzazione dei risultati in un livello scolastico approssimativo
    }

def analyze_narrative_structure(speech):
    sentences = sent_tokenize(speech)
    
    # Analyze sentence length variation
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    
    narrative_metrics = {
        'Average Sentence Length': np.mean(sentence_lengths),
        'Sentence Length Variation': np.std(sentence_lengths),
        'Total Sentences': len(sentences),
        'Complexity Score': np.mean(sentence_lengths) / (np.std(sentence_lengths) + 1)
    }

    return "\n".join(["%s: %.3f" % (key, value) if isinstance(value, float) else "%s: %d" % (key, value) for key, value in narrative_metrics.items()])
    
def tfidf(speech):
    # Definisce il numero di parole massimo per le keywords
    max_ngram_length = 1
    # Definisce quante keywords dobbiamo trovare
    top_n = 5

    # Inserisce i parametri di ricerca e il dizionario delle stop words
    vectorizer = TfidfVectorizer(
        ngram_range=(1, max_ngram_length),
        stop_words='english'
    )
    
    # Prende la matrice TF-IDF
    tfidf_matrix = vectorizer.fit_transform([speech])
    
    # Prende i nomi delle parole
    feature_names = vectorizer.get_feature_names_out()
    
    # Prende gli score TF-IDF
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Crea una lista di tuple (keyword, score)
    keyword_scores = [
        (feature_names[idx], score)
        for idx, score in enumerate(tfidf_scores)
        if score > 0
    ]
    
    # Ordina per score decrescente
    keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    
    # Prende solo le top_n keywords
    top_keywords = keyword_scores[:top_n]
    
    # Restituisce le parole chiave con i rispettivi punteggi
    return "\n".join(["%s: %.3f" % (kw.title(), round(score, 3)) for kw, score in top_keywords])

if __name__ == "__main__":
    import time
    import pandas as pd
    
    # Carica il dataset
    data_folder = "src/dataset/speech-filtered.tsv"
    dataset = pd.read_table(data_folder, sep='\t', header=None)

    # Assegna i nomi delle colonne
    dataset.columns = ["Surname", "Code", "Speech"]
    
    # Inizio della misurazione
    start_time = time.time()

    # Aggiunge le parole con score TF-IDF più alto
    dataset["TF-IDF"] = dataset["Speech"].apply(tfidf)

    # Aggiunge la struttura narrativa
    dataset["Narrative structure"] = dataset["Speech"].apply(analyze_narrative_structure)

    # Verifica il tono del discorso
    dataset["Tone"] = dataset["Speech"].apply(is_formal)

    # Calcola le metriche di leggibilità
    readability_metrics = dataset["Speech"].apply(calcola_legibilita)
    dataset["Flesch Reading Ease"] = readability_metrics.apply(lambda x: x["flesch_reading_ease"])
    dataset["Flesch Kincaid Grade"] = readability_metrics.apply(lambda x: x["flesch_kincaid_grade"])
    dataset["Gunning Fog"] = readability_metrics.apply(lambda x: x["gunning_fog"])
    dataset["Smog Index"] = readability_metrics.apply(lambda x: x["smog_index"])
    dataset["Text Standard"] = readability_metrics.apply(lambda x: x["text_standard"])

    # Fine della misurazione
    end_time = time.time()
        
    # Calcolo del tempo di esecuzione
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print("Tempo di esecuzione: %d ore, %d minuti, %.2f secondi" % (hours, minutes, seconds))

    print(dataset[["Surname", "TF-IDF", "Narrative structure"]])
    
    # Salva il dataset
    dataset.to_csv("src/dataset/keywords.csv", index=False, sep=",")