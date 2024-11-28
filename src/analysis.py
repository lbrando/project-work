import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Funzione per caricare un modello di classificazione
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


# Modello per il sentiment analysis
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer, sentiment_model = load_model_and_tokenizer(sentiment_model_name)
sentiment_labels = ["very negative", "negative", "neutral", "positive", "very positive"]

# Modello per l'emotion analysis
emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_tokenizer, emotion_model = load_model_and_tokenizer(emotion_model_name)
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Modello per la propaganda - determinare tipo di propaganda
propaganda_model_name = "IDA-SERICS/PropagandaDetection"
propaganda_classifier = pipeline("text-classification", model=propaganda_model_name)

# Modello per la toxicity
toxicity_model_name = "citizenlab/distilbert-base-multilingual-cased-toxicity"
toxicity_classifier = pipeline("text-classification", model=toxicity_model_name)

# Funzione unificata per classificare il testo
def classify_text(text, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return labels[prediction]

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
def detect_propaganda_type(text, classifier):
    is_propaganda_by_model = classifier(text, truncation=True, max_length=512)[0]['label'] == "propaganda"
    # Rileva i criteri di propaganda
    criteria_counts, predominant_type = contains_propaganda_criteria(text)
    # Se il modello classifica il testo come propaganda o ci sono criteri rilevati, ritorna il tipo
    if is_propaganda_by_model or criteria_counts:
        return predominant_type
    return None
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
def classify_toxicity(text, classifier):
    try:
        # Classificazione con il modello
        result = classifier(text, truncation=True, max_length=512)
 
        # Rilevamento criteri
        detected_criteria, _ = contains_toxicity_criteria(text)
 
        # Formattare i criteri rilevati uno per riga
        if detected_criteria:
            return '\n'.join([f"{k}: {v}" for k, v in detected_criteria.items()])
        else:
            return ""
    except Exception as e:
        print(f"Errore durante la classificazione del testo: {text}\n{e}")
        return ""