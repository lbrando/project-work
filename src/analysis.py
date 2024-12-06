import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Determina il dispositivo appropriato per il backend
if torch.backends.mps.is_available():  # Per macOS con Apple Silicon (MPS)
    device = "mps"
elif torch.cuda.is_available():  # Per GPU Nvidia (CUDA)
    device = "cuda"
else:  # CPU come fallback
    device = "cpu"

# Funzione per caricare un modello di classificazione
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Modello per il sentiment analysis
sentiment_tokenizer, sentiment_model = load_model_and_tokenizer("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_labels = ["very negative", "negative", "neutral", "positive", "very positive"]

# Modello per l'emotion analysis
emotion_tokenizer, emotion_model = load_model_and_tokenizer("bhadresh-savani/distilbert-base-uncased-emotion")
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Modello per la propaganda - determinare tipo di propaganda
propaganda_classifier = pipeline("text-classification", model="IDA-SERICS/PropagandaDetection", device=0 if device != "cpu" else -1)

# Modello per la toxicity
toxicity_classifier = pipeline("text-classification", model="citizenlab/distilbert-base-multilingual-cased-toxicity", device=0 if device != "cpu" else -1)

# Funzione unificata per classificare il testo
def classify_text(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return labels[prediction].capitalize()

# Funzione per rilevare i criteri di propaganda
def contains_propaganda_criteria(text):
    criteria = {
        "accusatory_tone": ["fault of"],
        "opposition_distortion": ["enemies", "traitor", "dishonest", "corrupt"],
        "slogan_repetition": ["you have to believe", "the truth is", "don't forget", "do not forget"],
        "fear_appeals": ["danger", "threat", "crisis", "chaos"],
        "flagwaving": ["our nation", "homeland", "defend values"],
        "black_white_fallacy": ["either with us or against us", "only choice"],
        "clichés": ["golden times", "like once upon a time", "old times"]
    }
    # Conta quante frasi/parole chiave sono presenti per ciascun criterio
    type_counts = {key: sum(phrase in text.lower() for phrase in phrases) for key, phrases in criteria.items()}
    # Filtra i criteri che hanno almeno una corrispondenza
    detected_types = {key: count for key, count in type_counts.items() if count > 0}
    # Se non ci sono criteri rilevati, ritorna None
    if not detected_types:
        return None, None
    predominant_type = max(detected_types, key=detected_types.get)
    return detected_types, predominant_type

# Funzione per rilevare il tipo di propaganda
def detect_propaganda_type(text, classifier=propaganda_classifier, chunk_size=512, overlap=128):
    import re
    from collections import Counter
 
    # Funzione per creare chunk con finestra scorrevole
    def split_text_sliding_window(text, chunk_size=512, overlap=128):
        words = text.split()
        chunks = []
        start = 0
 
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append((chunk, start, end))  # Include gli indici di partenza e fine
            start = end - overlap  # Sposta la finestra avanti
 
        return chunks
 
    # Dividi il testo in frasi usando una semplice regex
    def split_into_sentences(text):
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        return sentence_endings.split(text)
 
    # Divide il testo in chunk
    text_chunks = split_text_sliding_window(text, chunk_size, overlap)
    propaganda_results = []
 
    for chunk, start_idx, end_idx in text_chunks:
        # Classifica il chunk
        is_propaganda_by_model = classifier(chunk, truncation=True, max_length=chunk_size)[0]['label'] == "propaganda"
        criteria_counts, predominant_type = contains_propaganda_criteria(chunk)
 
        # Analizza le frasi del chunk
        if is_propaganda_by_model or criteria_counts:
            sentences = split_into_sentences(chunk)  # Dividi il chunk in frasi
            for sentence in sentences:
                # Classifica la singola frase
                sentence_is_propaganda = classifier(sentence, truncation=True)[0]['label'] == "propaganda"
                sentence_criteria, sentence_type = contains_propaganda_criteria(sentence)
 
                if sentence_is_propaganda or sentence_criteria:

                    propaganda_results.append({
                        "type": sentence_type,
                        "sentence": sentence
                    })
 
    # Se esistono risultati di propaganda
    if propaganda_results:
        # Conta la frequenza di ogni tipo di propaganda
        types = [result["type"] for result in propaganda_results]
        type_counts = Counter(types)
 
        # Prendi le 5 tipologie più comuni
        top_5_types = type_counts.most_common(5)
 
        # Estrai i risultati per le 5 tipologie più comuni
        results = []
        for propaganda_type, _ in top_5_types:
            sentences_of_type = [
                {"sentence": result["sentence"]}
                for result in propaganda_results if result["type"] == propaganda_type
            ]
            results.append({
                "type": propaganda_type,
                "sentences": sentences_of_type
            })
 
        return results
 
    return "Not identified"

'''
# Funzione per classificare se è propaganda
def classify_is_propaganda(row):
    if row['code'] == 1 and row['type of propaganda'] is not None:
        return "propaganda"
    elif row['code'] == 0 and row['type of propaganda'] is None:
        return "not propaganda"
    else:
        return "not defined"
'''

# Funzione per rilevare i criteri di tossicità
def contains_toxicity_criteria(text):
    criteria = {
        "insults": ["idiot", "stupid", "moron", "fool", "scum", "loser", "crooked", "degenerate", "ignoramus"],
        "threats": ["destroy", "kill", "eliminate", "eradicate", "you'll regret", "annihilate", "purge", "defeat utterly"],
        "violent_language": ["blood", "war", "fight", "smash", "knock them out", "battle", "no quarter"],
        "divisive_statements": ["us vs them", "enemy of the people", "deep state", "traitors", "fifth column"],
        "personal_attacks": ["you always fail", "your fault", "worthless", "incompetent", "phony", "betrayer"],
        "exaggerations": ["always lie", "never truthful", "the worst", "greatest challenge", "unprecedented threat"],
        "fear_appeals": ["danger", "threat", "crisis", "chaos", "they're coming for you", "crime wave"],
        "flagwaving": ["our nation", "homeland", "defend values", "Make America Great Again", "Fatherland", "motherland"]
    }
    # Conta quante frasi/parole chiave sono presenti per ciascun criterio
    type_counts = {key: sum(phrase in text.lower() for phrase in phrases) for key, phrases in criteria.items()}
    # Filtra i criteri che hanno almeno una corrispondenza
    detected_types = {key: count for key, count in type_counts.items() if count > 0}
    # Se non ci sono criteri rilevati, ritorna None
    if not detected_types:
        return None, None
    # Identifica il tipo di tossicità
    predominant_type = max(detected_types, key=detected_types.get)
    return detected_types, predominant_type

# Funzione per classificare la tossicità
def classify_toxicity(text):
    try:
        # Rilevamento criteri
        detected_criteria, _ = contains_toxicity_criteria(text)
 
        # Formattare i criteri rilevati uno per riga
        if detected_criteria:
            return '\n'.join(["%s: %s" % (k, v) for k, v in detected_criteria.items()]).replace("_", " ").title()
        else:
            return ""
    except Exception:
        raise
        return ""
    
def offsets(speech, data):

    # Verifica che la propaganda sia stata correttamente identificata
    if data == "Not identified":
        return "Not identified", None

    # Crea un dizionario per definire gli offset
    offsets = {
        # Crea una chiave per tipo e la formatta correttamente 
        entry["type"].replace("_", " ").capitalize(): [
            # Crea un offset per ogni frase
            "[%s:%s]" % (start, start + len(sentence["sentence"]))
            for sentence in entry["sentences"]
            # Trova le frasi e le misura
            if (start := speech.find(sentence["sentence"])) != -1
        ]
        for entry in data
    }

    # Restituice il tipo di propaganda e gli offset per ogni tipo di propaganda
    return "\n".join(offsets.keys()), "\n".join("%s: %s" % (key, ", ".join(element)) for key, element in offsets.items())