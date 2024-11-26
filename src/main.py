import time
import threading
import itertools
import pandas as pd
from pandas import concat
from transformers import pipeline
    
from analysis import load_model_and_tokenizer, classify_text, detect_propaganda_type, classify_toxicity

# Funzione per visualizzare l'animazione di caricamento
def loading_animation():
    for frame in itertools.cycle(["|", "/", "-", "\\"]):
        print(f"\rSto elaborando... {frame}", end="")
        time.sleep(0.2)

# Carica il dataset
data_folder = "speech-a.tsv"
dataset = pd.read_table(data_folder, sep='\t', names=["name", "code", "speech"])

# Carica modelli e tokenizer
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer, sentiment_model = load_model_and_tokenizer(sentiment_model_name)
sentiment_labels = ["very negative", "negative", "neutral", "positive", "very positive"]

emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer, emotion_model = load_model_and_tokenizer(emotion_model_name)
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

propaganda_model_name = "IDA-SERICS/PropagandaDetection"
propaganda_classifier = pipeline("text-classification", model=propaganda_model_name)

toxicity_model_name = "citizenlab/distilbert-base-multilingual-cased-toxicity"
toxicity_classifier = pipeline("text-classification", model=toxicity_model_name)

# Thread per l'animazione di caricamento
loading_thread = threading.Thread(target=loading_animation, daemon=True)

try:
    loading_thread.start()
    # Applica le funzioni al dataset
    dataset["toxicity"] = dataset["speech"].apply(lambda x: classify_toxicity(x, toxicity_classifier))
    dataset["type of propaganda"] = dataset["speech"].apply(lambda x: detect_propaganda_type(x, propaganda_classifier))
    #dataset["is propaganda"] = dataset.apply(classify_is_propaganda, axis=1)
    #dataset["sentiment"] = dataset["speech"].apply(lambda x: classify_text(x, sentiment_model, sentiment_tokenizer, sentiment_labels))
    #dataset["emotion"] = dataset["speech"].apply(lambda x: classify_text(x, emotion_model, emotion_tokenizer, emotion_labels))

    # Mostra il dataset risultante
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.width', 1000)
    #print(dataset)

    # Salva il dataset risultante in un file CSV
    dataset.to_csv("toxicity.csv", index=False)
finally:
    # Ferma l'animazione di caricamento e stampa il messaggio finale
    print("\rElaborazione completata!. Salvataggio completato")