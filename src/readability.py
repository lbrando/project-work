import textstat
from textblob import TextBlob
import spacy
from typing import List, Dict


def main():
    import pandas as pd
    
    # Carica il modello di spaCy
    nlp = spacy.load("en_core_web_sm")

    # Leggi il dataset
    dataset = pd.read_csv('speech-a.tsv', sep='\t', header=None, names=["surname", "code", "speech"])

    # Funzione per determinare se il testo è formale o informale
    def is_formal_textblob(text):
        if isinstance(text, str):
            if len(text) > 150:
                blob = TextBlob(text)
                # Bassa soggettività indica formalità
                if blob.sentiment.subjectivity < 0.5:
                    return "Formal"
                else:
                    return "Informal"
            else:
                return "N/A"

    # Funzione per calcolare metriche di leggibilità
    def calcola_legibilita(testo):
        if isinstance(testo, str):  # Controlla che il testo non sia NaN
            return {
                "flesch_reading_ease": textstat.flesch_reading_ease(testo),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(testo),
                "gunning_fog": textstat.gunning_fog(testo),
                "smog_index": textstat.smog_index(testo),
                "text_standard": textstat.text_standard(testo)
            }
        else:
            return {"flesch_reading_ease": None, "flesch_kincaid_grade": None,
                    "gunning_fog": None, "smog_index": None, "text_standard": None}

    # Lists of persuasive words and techniques
    PERSUASIVE_WORDS = {
        "emotional": ["incredible", "amazing", "unique", "fantastic", "best", "exceptional", "unbelievable"],
        "urgency": ["now", "immediately", "instant", "quickly", "don't miss", "limited time"],
        "authority": ["experts", "proven", "scientifically", "research", "leaders", "professional"],
        "social_proof": ["everyone", "most", "majority", "common", "shared", "popular"]
    }

    def analyze_persuasion(text: str) -> Dict[str, float]:
        """
        Analyze text and calculate a detailed persuasion score.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict[str, float]: Dictionary with various persuasion scores
        """
        
        if not isinstance(text, str) or not text.strip():
            return {
                "total_score": 0.0, # totale di persuasione
                "superlatives": 0.0, # frequenza dei superlativi
                "rhetorical_questions": 0.0, # frequenza di domande retorica
                "emotional_words": 0.0, # frequenza di parole emotive
                "urgency_words": 0.0, # frequenza di parole di urgenza
                "authority_words": 0.0, # frequenza di parole di autorità
                "social_proof_words": 0.0 # frequenza di parole di prova sociale
            }
        
        # Analisi del testo usando spaCy
        doc = nlp(text.lower())
        
        # Calcola metriche persuasive
        # Per ogni tipo di elemento persuasivo, conta quante volte appare nel testo 
        metrics = {
            # conta i superlativi
            "superlatives": sum(1 for token in doc if token.tag_ in ["JJS", "RBS"]),
            # conta le domande retoriche, basandosi sul fatto che terminano con un punto di domanda
            "rhetorical_questions": sum(1 for sent in doc.sents if sent.text.strip().endswith("?")),
            # conta le parole emotive che appartengono alla lista predefinita
            "emotional_words": sum(1 for token in doc if token.text in PERSUASIVE_WORDS["emotional"]),
            # conta le parole legate all'urgenza dalla lista predefinita
            "urgency_words": sum(1 for token in doc if token.text in PERSUASIVE_WORDS["urgency"]),
            # conta le parole legate all'autorità
            "authority_words": sum(1 for token in doc if token.text in PERSUASIVE_WORDS["authority"]),
            # conta le parole di prova sociale
            "social_proof_words": sum(1 for token in doc if token.text in PERSUASIVE_WORDS["social_proof"])
        }
        
        # Calcola il numero totale di token
        total_tokens = len(doc)
        # Se il testo non contiene token, restituisce un dizionario vuoto
        if total_tokens == 0:
            return {key: 0.0 for key in metrics.keys() | {"total_score"}}
        
        # Definisce il peso per ogni elemento persuasivo, indicando quanto è importante ciascuna metrica
        weights = {
            "superlatives": 1.5, # peso medio
            "rhetorical_questions": 2.0, # sono più persuasive
            "emotional_words": 2.5, # sono le più persuasive
            "urgency_words": 2.0, # sono persuasive 
            "authority_words": 1.8, # peso medio-alto
            "social_proof_words": 1.7 # peso inferiore
        }
        
        # Calcola uno score normalizzato per ogni metrica
        # Per ogni metrica:
        # - Divido il conteggio per il numero totale di token (normalizzazione)
        # - MOltiplico per il peso associato per enfatizzare la sua importanza 
        normalized_scores = {
            key: (metrics[key] / total_tokens) * weights[key] for key in metrics
        }
        
        # Somma tutti gli score normalizzati per ottenere lo score totale di persuasione
        total_score = sum(normalized_scores.values())
        normalized_scores["total_score"] = round(total_score, 3) # arrotondare a 3 - scelta comune per rappresentare valori numerici in cui la precisione oltre il millesimo non è necessaria
        
        return normalized_scores # restituisce il dizionario con tutti gli score

    # Applica la funzione su ogni riga della colonna "speech"
    readability_metrics = dataset["speech"].apply(calcola_legibilita)
    persuasion_scores = dataset["speech"].apply(analyze_persuasion)

    # Estrai i risultati retorici
    dataset['tone'] = dataset['speech'].apply(is_formal_textblob)
    dataset['persuasion_score'] = persuasion_scores
    dataset["flesch_reading_ease"] = readability_metrics.apply(lambda x: x["flesch_reading_ease"])
    dataset["flesch_kincaid_grade"] = readability_metrics.apply(lambda x: x["flesch_kincaid_grade"])
    dataset["gunning_fog"] = readability_metrics.apply(lambda x: x["gunning_fog"])
    dataset["smog_index"] = readability_metrics.apply(lambda x: x["smog_index"])
    dataset["text_standard"] = readability_metrics.apply(lambda x: x["text_standard"])

    # Salva il dataset con i nuovi campi
    dataset.to_csv("readability1.csv", index=False)

if __name__ == "__main__":
    main()