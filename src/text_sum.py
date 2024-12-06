import os
import torch
from transformers import pipeline, logging, AutoTokenizer
import time

# Disabilita parallelismo dei tokenizer per evitare deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configura il logging di Hugging Face per sopprimere i messaggi di log
logging.set_verbosity_error()  # Sopprime tutti i messaggi di log di livello inferiore a ERROR

if torch.backends.mps.is_available():  # Per macOS con Apple Silicon (MPS)
    device = "mps"
elif torch.cuda.is_available():  # Per GPU Nvidia (CUDA)
    device = "cuda"
else:  # CPU come fallback
    device = "cpu"

summarizer = pipeline(
    "summarization",
    "pszemraj/led-base-book-summary",
    device=0 if device != "cpu" else -1,
)

# Inizializza il tokenizer per il modello
tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-base-book-summary")

# Funzione per dividere il testo in segmenti che rispettano il limite dei token
def split_text(text, max_tokens=16000):
    tokens = tokenizer.encode(text, truncation=False, padding=False)

    if len(tokens) <= max_tokens:
        return [text]  # Nessuna divisione necessaria
    
    # Se il testo è troppo lungo lo divide in segmenti di max_tokens
    split_texts = []
    for i in range(0, len(tokens), max_tokens):
        split_text = tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True)
        split_texts.append(split_text)
    return split_texts

def summarize_speech(speech):
    # Split del discorso in parti più piccole, se necessario
    split_speeches = split_text(speech)
    summaries = []
    for part in split_speeches:
        summary = summarizer(
            part,
            min_length=2,
            max_length=100,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            repetition_penalty=3.5,
            num_beams=1,
            length_penalty=None,
            do_sample=False,
            early_stopping=False,
        )[0]["summary_text"]
        summaries.append(summary)
    # Combina i riassunti delle parti
    return ' '.join(summaries)

def main():
    import pandas as pd
    import concurrent.futures

    dataset = pd.read_table("src/dataset/speech-a.tsv", sep="\t", names=["Surname", "Code", "Speech"])

    # Inizio della misurazione
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Applica direttamente la logica di riepilogo nel map
        results = list(executor.map(lambda x: summarize_speech(x) if len(x) > 150 else x, dataset["Speech"]))

    # Aggiunge il risultato alla colonna "Summary"
    dataset["Summary"] = results

    # Fine della misurazione
    end_time = time.time()

    print(dataset[["Surname", "Summary"]])

    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print("Tempo di esecuzione: %d ore, %d minuti, %.2f secondi" % (hours, minutes, seconds))

    dataset.to_csv("src/dataset/summary.csv", index=False)

if __name__ == "__main__":
    main()
