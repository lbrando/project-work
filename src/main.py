import time
import pandas as pd
import itertools
import threading
import concurrent.futures
import os
import funzioni as fn

### DEFINIZIONE DEI PARAMETRI ###
# Carica il percorso del dataset
data_folder = "src/dataset/speech-a.tsv"
# Seleziona il percorso del dataset da salvare
data_output = "src/dataset/speech-b.csv"
# Seleziona il percorso in cui salvare il word embedding
embedding_output = "src/word_embedding.png"

# Legge il dataset e da il nome alle colonne, eliminando eventuali colonne vuote o duplicate
dataset = pd.read_table(data_folder, sep="\t", names=["Surname", "Code", "Speech"]).dropna().drop_duplicates()

# Cleaning del dataset
# Rimuove i link dai testi
dataset["Speech"] = dataset["Speech"].apply(fn.remove_links)
# Filtra il dataset per mantenere solo le utilizzabili
dataset = dataset[(dataset["Speech"].str.len() >= 200) &
                  (dataset["Code"].isin([0, 1])) &
                  dataset["Surname"].apply(lambda x: isinstance(x, str))]

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
    dataset["Full name"] = dataset["Surname"].apply(lambda x: fn.get_info(x, "name"))
    dataset["Birthday"] = dataset["Surname"].apply(lambda x: fn.get_info(x, "birthday"))
    dataset["Birth place"] = dataset["Surname"].apply(lambda x: fn.get_info(x, "birthplace"))
    dataset["Death day"] = dataset["Surname"].apply(lambda x: fn.get_info(x, "deathday"))
    dataset["Death place"] = dataset["Surname"].apply(lambda x: fn.get_info(x, "deathplace"))
    dataset["Political party"] = dataset["Surname"].apply(lambda x: fn.get_info(x, "party"))

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
total_time = total_seconds
print("⏳ Tempo di esecuzione dello scraping: %d ore, %d minuti, %.2f secondi\n" % (int(total_seconds // 3600), int((total_seconds % 3600) // 60), total_seconds % 60))

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        # Mappa le chiamate API in parallelo
        results = list(executor.map(fn.speech_info, dataset["Full name"], dataset["Speech"]))
    
    # Crea un dizionario per mappare i risultati
    results_dict = dict(results)
    
    # Crea le nuove colonne in modo sicuro
    speech_data = dataset["Speech"].apply(lambda x: fn.get_speech_data(x, results_dict)).tolist()
    
    # Separa le colonne
    dataset["Speech Date"] = [row[0] for row in speech_data]
    dataset["Speech location"] = [row[1] for row in speech_data]
    dataset["Speech occasion"] = [row[2] for row in speech_data]
    dataset["Topic"] = [row[3] for row in speech_data]
    dataset["Cognitive Bias"] = [row[4] for row in speech_data]
    dataset["Summary"] = [row[5] for row in speech_data]
    dataset["Keywords"] = [row[6] for row in speech_data]
    dataset["Underlying narrative"] = [row[7].replace(";", "\n\n") for row in speech_data]

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
total_time += total_seconds
print("⏳ Tempo di esecuzione per l'ottenimento delle informazioni: %d ore, %d minuti, %.2f secondi\n" % (int(total_seconds // 3600), int((total_seconds % 3600) // 60), total_seconds % 60))

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
    dataset["Toxicity text count"] = dataset["Speech"].apply(fn.classify_toxicity)
    dataset["Type of propaganda"], dataset["Offsets"] = zip(*dataset.apply(
        lambda row: fn.offsets(row["Speech"], fn.detect_propaganda_type(row["Speech"])) if row["Code"] == 1 else ("Not propaganda", None), axis=1
    ))
    dataset["Sentiment"] = dataset["Speech"].apply(
        lambda x: fn.classify_text(x, fn.sentiment_model, fn.sentiment_tokenizer, fn.sentiment_labels)
    )
    dataset["Emotion"] = dataset["Speech"].apply(
        lambda x: fn.classify_text(x, fn.emotion_model, fn.emotion_tokenizer, fn.emotion_labels)
    )
    dataset["Persuasion score"] = dataset["Speech"].apply(fn.analyze_persuasion)

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
total_time += total_seconds
print("⏳ Tempo di esecuzione dell'analisi: %d ore, %d minuti, %.2f secondi\n" % (int(total_seconds // 3600), int((total_seconds % 3600) // 60), total_seconds % 60))

# Avvia la misurazione
start_time = time.time()

success = True

# Fase 4: Feature text-based
print("📝 Aggiungo feature text-based")
# Avvia l'animazione di caricamento
stop_event = threading.Event()
loading_thread = threading.Thread(
    target=loading_animation,
    args=("Aggiunta in corso", stop_event, "✅ Aggiunta completata")
)
loading_thread.start()

# Applica le funzioni di analisi
try:
    # Aggiunge le parole con score TF-IDF più alto
    dataset["TF-IDF"] = dataset["Speech"].apply(fn.tfidf)

    # Aggiunge la struttura narrativa
    dataset["Narrative structure"] = dataset["Speech"].apply(fn.analyze_narrative_structure)

    # Verifica il tono del discorso
    dataset["Tone"] = dataset["Speech"].apply(fn.is_formal)

    # Calcola metriche di leggibilità
    readability_metrics = dataset["Speech"].apply(fn.calcola_legibilita)
    dataset["Flesch Reading Ease"] = readability_metrics.apply(lambda x: x["flesch_reading_ease"])
    dataset["Flesch Kincaid Grade"] = readability_metrics.apply(lambda x: x["flesch_kincaid_grade"])
    dataset["Gunning Fog"] = readability_metrics.apply(lambda x: x["gunning_fog"])
    dataset["Smog Index"] = readability_metrics.apply(lambda x: x["smog_index"])
    dataset["Text Standard"] = readability_metrics.apply(lambda x: x["text_standard"])

except Exception as e:
    print("\r❌ Errore nell'aggiunta delle feature text-based: %s" % e)
    success = False

finally:
    # Ferma l'animazione di caricamento
    stop_event.set()
    loading_thread.join()

# Fine della misurazione
end_time = time.time()

# Calcola il tempo di elaborazione
total_seconds = end_time - start_time
total_time += total_seconds
print("⏳ Tempo di esecuzione dell'analisi: %d ore, %d minuti, %.2f secondi\n" % (int(total_seconds // 3600), int((total_seconds % 3600) // 60), total_seconds % 60))

while True:
    plt_show = input("📈 Visualizzare il grafico del word embedding? (s/n) ").lower()
    if plt_show == "s" or plt_show == "n":
        break
    else:
        print("❌ Inserire solo un valore tra 's' o 'n'")

# Avvia la misurazione
start_time = time.time()

success = True

# Fase 5: Word embedding
print("📊 Calcolo il word embedding")
# Avvia l'animazione di caricamento
stop_event = threading.Event()
loading_thread = threading.Thread(
    target=loading_animation,
    args=("Calcolo in corso", stop_event, "✅ Calcolo completato")
)
loading_thread.start()

# Applica le funzioni di analisi
try:
    plt = fn.visualize_embeddings(dataset, plt_show, embedding_output)
    dataset = dataset.drop(columns=["Embedding", "x", "y"])

except Exception as e:
    print("\r❌ Errore nel calcolo del word embedding: %s" % e)
    success = False

finally:
    # Ferma l'animazione di caricamento
    stop_event.set()
    loading_thread.join()

# Fine della misurazione
end_time = time.time()

# Calcola il tempo di elaborazione
total_seconds = end_time - start_time
total_time += total_seconds
print("⏳ Tempo di esecuzione del word embedding: %d ore, %d minuti, %.2f secondi\n" % (int(total_seconds // 3600), int((total_seconds % 3600) // 60), total_seconds % 60))

if plt_show == "s":
    plt.show()
    print("")

# Salva il dataset risultante in un file CSV
dataset.to_csv(data_output, index=False)
print("✅ Grafico del word embedding salvato in %s" % embedding_output)
print("✅ Dataset salvato in %s\n" % data_output)

# Calcola il tempo di elaborazione
total_seconds = end_time - start_time
total_time += total_seconds
print("⏳ Tempo di esecuzione totale: %d ore, %d minuti, %.2f secondi\n" % (int(total_time // 3600), int((total_time % 3600) // 60), total_time % 60))