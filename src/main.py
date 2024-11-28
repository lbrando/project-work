import time
import threading
import itertools
import pandas as pd
from pandas import concat
from transformers import pipeline
    
from analysis import load_model_and_tokenizer, classify_text, detect_propaganda_type, classify_toxicity
from analysis import sentiment_model, sentiment_tokenizer, sentiment_labels
from analysis import emotion_model, emotion_tokenizer, emotion_labels

# Funzione per visualizzare l'animazione di caricamento
def loading_animation():
    for frame in itertools.cycle(["|", "/", "-", "\\"]):
        print(f"\rSto elaborando... {frame}", end="")
        time.sleep(0.2)

# Carica il dataset
data_folder = "speech-a.tsv"
dataset = pd.read_table(data_folder, sep='\t', names=["surname", "code", "speech"])

# Thread per l'animazione di caricamento
loading_thread = threading.Thread(target=loading_animation, daemon=True)

try:
    loading_thread.start()
    # Applica le funzioni al dataset
    dataset["toxicity_text_count"] = dataset["speech"].apply(classify_toxicity)
    dataset["type of propaganda"] = dataset["speech"].apply(detect_propaganda_type)
    dataset["sentiment"] = dataset["speech"].apply(lambda x: classify_text(x, sentiment_model, sentiment_tokenizer, sentiment_labels))
    dataset["emotion"] = dataset["speech"].apply(lambda x: classify_text(x, emotion_model, emotion_tokenizer, emotion_labels))
    
    # Salva il dataset risultante in un file CSV
    dataset.to_csv("toxicity.csv", index=False)
finally:
    
    # Ferma l'animazione di caricamento e stampa il messaggio finale
    print("\rElaborazione completata!. Salvataggio completato")