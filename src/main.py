import time
import pandas as pd
import itertools
import threading
import concurrent.futures

# Importa i file come moduli
import analysis as an
import scraping_politician as sp
import scraping_speech as ss

### DEFINIZIONE DEI PARAMETRI ###
# Carica il percorso del dataset
data_folder = "src/dataset/speech-filtered.tsv"
# Seleziona il percorso del dataset ad salvare
data_output = "src/dataset/dataset-processato-1.csv"

# Legge il dataset e da il nome alle colonne
dataset = pd.read_table(data_folder, sep="\t", names=["Surname", "Code", "Speech"])

# Funzione per visualizzare un'animazione di caricamento
def loading_animation(message, stop_event, completed_message):
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not stop_event.is_set():
        print("\r%s  %s" % (next(spinner), message), end="", flush=True)
        time.sleep(0.1)
    if success:
        print("\r%s" % completed_message, flush=True)

# Inizia la misurazione
start_time = time.time()

success = True

# Fase 1: Scraping
print("\n🕵️  Eseguo lo scraping delle informazioni dei politici")
# Avvia l'animazione di caricamento
stop_event = threading.Event()
loading_thread = threading.Thread(
    target=loading_animation,
    args=("Scraping in corso", stop_event, "✅ Scraping completato")
)
loading_thread.start()

# Trova i dati dei politici e li aggiunge al dataset
try:
    dataset["Full name"] = dataset["Surname"].apply(lambda x: sp.get_info(x, "name"))
    dataset["Birthday"] = dataset["Surname"].apply(lambda x: sp.get_info(x, "birthday"))
    dataset["Birth place"] = dataset["Surname"].apply(lambda x: sp.get_info(x, "birthplace"))
    dataset["Death day"] = dataset["Surname"].apply(lambda x: sp.get_info(x, "deathday"))
    dataset["Death place"] = dataset["Surname"].apply(lambda x: sp.get_info(x, "deathplace"))
    dataset["Political party"] = dataset["Surname"].apply(lambda x: sp.get_info(x, "party"))

except Exception as e:
    print("\r❌ Errore durante lo scraping dei dati: %s" % e)
    success = False

finally:
    # Ferma l'animazione di caricamento
    stop_event.set()
    loading_thread.join()

# Fine della misurazione
end_time = time.time()

# Calcola il tempo di elaborazione
total_seconds = end_time - start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = total_seconds % 60
print("⏳ Tempo di esecuzione dello scraping: %d ore, %d minuti, %.2f secondi\n" % (hours, minutes, seconds))

# Inizia la misurazione
start_time = time.time()

success = True

# Fase 2: Ottenimento informazioni
print("🗣️  Ottengo le informazioni sui discorsi")
# Avvia l'animazione di caricamento
stop_event = threading.Event()
loading_thread = threading.Thread(
    target=loading_animation,
    args=("Ottengo informazioni", stop_event, "✅ Informazioni ottenute")
)
loading_thread.start()

# Trova i dati dei politici e li aggiunge al dataset
try:
    # Usa ThreadPoolExecutor per l'elaborazione parallela
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Mappa le chiamate API in parallelo
        results = list(executor.map(ss.speech_info, dataset["Full name"], dataset["Speech"]))
    
    # Crea un dizionario per mappare i risultati
    results_dict = dict(results)
    
    # Crea le nuove colonne in modo sicuro
    speech_data = dataset["Speech"].apply(lambda x: ss.get_speech_data(x, results_dict)).tolist()
    
    # Separa le colonne
    dataset["Speech Date"] = [row[0] for row in speech_data]
    dataset["Speech location"] = [row[1] for row in speech_data]
    dataset["Speech occasion"] = [row[2] for row in speech_data]

except Exception as e:
    print("\r❌ Errore nell'ottenimento delle informazioni: %s" % e)
    success = False

finally:
    # Ferma l'animazione di caricamento
    stop_event.set()
    loading_thread.join()

# Fine della misurazione
end_time = time.time()

# Calcola il tempo di elaborazione
total_seconds = end_time - start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = total_seconds % 60
print("⏳ Tempo di esecuzione per l'ottenimento delle informazioni: %d ore, %d minuti, %.2f secondi\n" % (hours, minutes, seconds))

# Avvia la misurazione
start_time = time.time()

success = True

# Fase 3: Analisi
print("🧠 Eseguo l'analisi dei discorsi")
# Avvia l'animazione di caricamento
stop_event = threading.Event()
loading_thread = threading.Thread(
    target=loading_animation,
    args=("Analisi in corso", stop_event, "✅ Analisi completata")
)
loading_thread.start()

# Applica le funzioni di analisi
try:
    dataset["Toxicity text count"] = dataset["Speech"].apply(an.classify_toxicity)
    dataset["Type of propaganda"] = dataset.apply(
        lambda row: an.detect_propaganda_type(row["Speech"]) if row["Code"] == 1 else "Not propaganda", axis=1
    )
    dataset["Sentiment"] = dataset["Speech"].apply(
        lambda x: an.classify_text(x, an.sentiment_model, an.sentiment_tokenizer, an.sentiment_labels)
    )
    dataset["Emotion"] = dataset["Speech"].apply(
        lambda x: an.classify_text(x, an.emotion_model, an.emotion_tokenizer, an.emotion_labels)
    )

except Exception as e:
    print("\r❌ Errore nell'analisi dei discorsi: %s" % e)
    success = False

finally:
    # Ferma l'animazione di caricamento
    stop_event.set()
    loading_thread.join()

# Fine della misurazione
end_time = time.time()

# Calcola il tempo di elaborazione
total_seconds = end_time - start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = total_seconds % 60
print("⏳ Tempo di esecuzione dell'analisi: %d ore, %d minuti, %.2f secondi\n" % (hours, minutes, seconds))

# Salva il dataset risultante in un file CSV
dataset.to_csv(data_output, index=False)
print("✅ Dataset salvato in %s\n" % data_output)