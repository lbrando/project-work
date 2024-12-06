import textstat
from textblob import TextBlob

def main():
    import pandas as pd
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
                "flesch_reading_ease": textstat.flesch_reading_ease(testo), # facilità di lettura basandosi sulla lunghezza delle frasi e sul numero di sillabe per parola
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(testo), #  anni di istruzione necessari per comprendere il testo
                "gunning_fog": textstat.gunning_fog(testo), # complessità del testo in base alla lunghezza delle frasi e alla percentuale di parole complesse
                "smog_index": textstat.smog_index(testo), # come quello precedente
                "text_standard": textstat.text_standard(testo) # sintetizzazione dei risultati in un livello scolastico approssimativo
            }
        else:
            return {"flesch_reading_ease": None, "flesch_kincaid_grade": None,
                    "gunning_fog": None, "smog_index": None, "text_standard": None}

    # Applica la funzione su ogni riga della colonna "speech"
    readability_metrics = dataset["speech"].apply(calcola_legibilita)


    dataset['tone'] = dataset['speech'].apply(is_formal_textblob)
    dataset["flesch_reading_ease"] = readability_metrics.apply(lambda x: x["flesch_reading_ease"])
    dataset["flesch_kincaid_grade"] = readability_metrics.apply(lambda x: x["flesch_kincaid_grade"])
    dataset["gunning_fog"] = readability_metrics.apply(lambda x: x["gunning_fog"])
    dataset["smog_index"] = readability_metrics.apply(lambda x: x["smog_index"])
    dataset["text_standard"] = readability_metrics.apply(lambda x: x["text_standard"])


    dataset.to_csv("readibility1.csv", index=False)

if __name__ == "__main__":
    main()