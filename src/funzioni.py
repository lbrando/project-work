import requests
import wikipedia
from bs4 import BeautifulSoup
import os
import openai
import torch
from collections import Counter
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
import textstat
from textblob import TextBlob
import spacy

# Scarica le risorse di NLTK necessarie in background
nltk.download('punkt', quiet=True)
# Carica il modello di spaCy
nlp = spacy.load("en_core_web_sm")

# Crea il dizionario di cache dei politici
politici = {}

# Imposta la pagina di wikipedia in inglese
wikipedia.set_lang("en")

# Legge la API key
os.environ["OPENAI_API_KEY"] = open("src/api key/API-GLHF.txt", "r").read()

# Disabilita parallelismo dei tokenizer per evitare deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Determina il dispositivo appropriato per il backend
if torch.cuda.is_available(): # Per GPU Nvidia (CUDA)
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Per macOS con Apple Silicon (MPS)
    device = "mps"
else:  # CPU come fallback
    device = "cpu"

# Verifica che la pagina si riferisca ad un politico
def verify_politician(titolo):

    try:
        # Prende la prima frase del riassunto della pagina
        summary = wikipedia.summary(titolo, sentences=1)
        # Filtra per le parole chiave contenute nella prima frase
        return any(keyword in summary.lower() for keyword in ["who", "politician", "statesman"])

    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return False

# Ottiene le informazioni dal polico
def get_info(cognome, parametro):
    # Verifica se il parametro esiste 
    if cognome in politici and parametro in politici[cognome]:
        return politici[cognome][parametro]
    
    # Verifica se il cognome esiste e crea il dizionario
    elif cognome not in politici:
        politici[cognome] = {}

    # Verifica se il soup non esiste e lo deve creare
    if "soup" not in politici[cognome]:
        # Cerca su wikipedia il cognome dato
        results = wikipedia.search(cognome)

        # Trova l'indice del risultato del politico tramite la funzione verify_politician
        index = next((i for i, result in enumerate(results) if verify_politician(result)), None)
        
        if index is not None:
            # Prende l'URL della pagina wikipedia
            url = wikipedia.page(results[index]).url

            # Effettua una richiesta GET alla pagina
            response = requests.get(url)

            # Verifica se la richiesta ha avuto successo
            if response.status_code == 200:
                # Crea l'oggetto BeautifulSoup per analizzare l'HTML e lo aggiunge dizionario
                politici[cognome]["soup"] = BeautifulSoup(response.text, features="html.parser")
            else:
                print("Errore nella richiesta della pagina:", response.status_code)

    # Se il soup è stato creato cerca il parametro da restituire
    if "soup" in politici[cognome]:
        # Prende la soup già esistente nel dizionario
        soup = politici[cognome]["soup"]
        # Trova l'infobox contenente le rows in cui sono presenti diversi dati
        infobox = soup.find("table", {"class": "infobox"}).find_all("tr")

        # Cerca il nome
        if parametro == "name":
            # Estrai il nome completo
            name = soup.find("div", {"class": "nickname"}).text

            if name:
                # Aggiunge il nome al dizionario
                politici[cognome]["name"] = name
                # Restituisce il nome
                return name
            return "N/A"
        
        # Cerca la data di nascita
        elif parametro == "birthday":
            # Estrae la data di nascita
            birth_date = soup.find("span", {"class": "bday"}).text

            if birth_date:
                # Aggiunge la data di nascita al dizionario
                politici[cognome]["birthday"] = birth_date
                # Restituisce la data di nascita
                return birth_date
            return "N/A"
        
        # Cerca il luogo di nascita
        elif parametro == "birthplace":
            # Trova il luogo di nascita in caso questo sia un link
            try:
                # Estrae il luogo di nascita
                birth_place = next(
                    # Prende il link del luogo di nascita e lo unisce allo Stato di nascita
                    ("%s%s" % (row.find("a").text, row.find("a").find_next_sibling(string=True).text)
                    for row in infobox if "Born" in row.get_text()), 
                    None
                )
            # Trova il luogo di nascita in caso questo sia una scritta semplice
            except AttributeError:
                # Estrae il luogo di nascita
                birth_place = next(
                    # Prende la stringa del luogo di nascita
                    ("%s" % row.find("br").find_next_sibling(string=True).text
                    for row in infobox if "Born" in row.get_text()), 
                    None
                )

            if birth_place:
                # Aggiunge il luogo di nascita al dizionario
                politici[cognome]["birthplace"] = birth_place
                # Restituisce il luogo di nascita
                return birth_place
            return "N/A"
        
        # Cerca la data di morte
        elif parametro == "deathday":
            # Estrae la data di morte
            death_day = next(
                # Prende lo span della data di morte
                (row.find("span").text
                for row in infobox if "Died" in row.get_text()), 
                None
            )

            if death_day:
                # Riformatta la data per togliere le parentesi ed avere il formato aaaa-mm-gg
                death_day = death_day[1:-1]
                # Aggiunge la data di morte al dizionario
                politici[cognome]["deathday"] = death_day
                # Restituisce la data di morte
                return death_day
            # Aggiunge None al dizionario nel caso in cui la persona non sia morta
            politici[cognome]["deathday"] = None
            return None
        
        # Cerca il luogo di morte
        elif parametro == "deathplace":
            # Trova il luogo di morte in caso questo sia un link
            try:
                # Estrae il luogo di morte
                death_place = next(
                    # Prende il link del luogo di morte e lo unisce allo Stato di morte
                    ("%s%s" % (row.find("a").text, row.find("a").find_next_sibling(string=True).text)
                    for row in infobox if "Died" in row.get_text()), 
                    None
                )
            # Trova il luogo di nascita in caso questo sia una scritta semplice
            except AttributeError:
                death_place = next(
                    # Prende la stringa del luogo di morte
                    ("%s" % row.find("br").find_next_sibling(string=True).text
                    for row in infobox if "Died" in row.get_text()), 
                    None
                )

            if death_place:
                # Aggiunge il luogo di morte al dizionario
                politici[cognome]["deathplace"] = death_place
                # Restituisce il luogo di morte
                return death_place
            # Aggiunge None al dizionario nel caso in cui la persona non sia morta
            politici[cognome]["deathplace"] = None
            return None
        
        # Cerca il partito politico
        elif parametro == "party":
            # Estrae il partito politico
            party = next(
            # Prende il link del partito politico
                ("%s" % row.find("a").text
                for row in infobox if "Political party" in row.get_text()), 
                None
            )

            if party:
                # Aggiunge il partito politico al dizionario
                politici[cognome]["party"] = party
                # Restituisce il partito politico
                return party
            return "N/A"
            
        else:
            print("Parametro selezionato non valido.")

def speech_info(politician, speech):
    try:
        client = openai.OpenAI(
            # Utilizza le API di GLHF per l'utilizzo del modello llama
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://glhf.chat/api/openai/v1"
        )
        completion = client.chat.completions.create(
            model="hf:meta-llama/Meta-Llama-3.1-405B-Instruct",
            # Prompt per il modello llama
            messages=[
                {"role": "system", "content": "Provide only the answer, do not add any element. If you cannot provide a specific information provide instead the value 'N/A'."},
                {"role": "user", "content": """
                Extract the following information from the provided speech or text, formatted as 'yyyy-mm-dd$location, state$occasion$topic$cognitive bias$summary$keywords$underlying narrative':

                Date: The exact date the speech was delivered, formatted as 'yyyy-mm-dd'.
                Location: The city or venue where the speech was delivered, followed by the state, separated by a comma (,). If the state is unavailable, provide only the city or venue, do not provide N/A if you do not find the state.
                Occasion: The specific event, purpose, or context for which the speech was delivered.
                Topic: The specific topic of the speech, or a general theme or subject, if applicable, in one or a few words.
                Cognitive bias: The bias of the speech, such as Illusory correlation, Causal Fallacy, Implicit association, etc. if there are any. If there is more then one bias provide all of them in a list formatted as 'bias1, bias2, bias3'. If the bias is unavailable, provide None.
                Summary: A short summary of the speech, in less then 100 words, if available. If the speech is too short to summarize, provide the whole speech. If there is no speech provide 'No speech provided'.
                Keywords: The keywords extracted from the speech, in one or a few words (at most 5), if available. If the keywords cannot be extracted, provide 'N/A'. Provide the keywords in a list formatted as 'keyword1, keyword2, keyword3'.
                Underlying narrative: Analyze the political speech and provide a comprehensive breakdown of its underlying narrative. Follow these steps and provide them in a bullet list, with the following format: 'Conext: description.;Rhetorical Structure: description.;Main Themes and Values: description.;...'. Always provide ";" between each step and do not use spaces alongside it. Use a fullstop at the end of every description. Do never provide '\n'. If the speech is too short to analyze, provide 'N/A'.
                    Context: Offer a brief analysis of the historical, cultural, and political context of the speech. Identify the target audience and the purpose of the message.
                    Rhetorical Structure: Identify the organization of the speech, highlighting key points and the sequence of arguments.
                    Main Themes and Values: List the central themes and implicit values. Analyze how the protagonist (positive actor), the antagonist (obstacle or problem), and the proposed solution are represented.
                    Use of Language: Examine the tone, rhetorical devices, and symbolic imagery used. Explain how these elements contribute to delivering the message.
                    Emotional Appeals and Persuasive Strategy: Analyze appeals to emotions such as hope, fear, or empathy, and assess any manipulative elements, including stereotypes, generalizations, or polarizations.
                    Conclusion: Summarize the speech’s effectiveness. Is the narrative coherent and convincing? How might it influence the audience?
                    
                Output Format:
                Always provide the information in the exact order: 'date$location, state$occasion$topic$cognitive bias$summary$keywords$underlying narrative'.
                Separate each piece of information strictly with the $ character, without spaces before or after it.
                If a specific piece of information cannot be determined, use N/A for that field.
                
                Rules for Extraction:
                Extract as much information as possible from the text. If one or tswo fields can be determined but others cannot, provide the known fields and use N/A only for those that are truly unavailable.
                Use logical inference when explicit information is missing. For example, infer the state if the city is well-known, or infer the occasion based on context (e.g., "July 4th" implies Independence Day). Ensure the inference is reasonable and based only on the provided text.
                Try to give an answer as much as you can.
                Avoid using N/A unless absolutely necessary, and only if no reasonable inference or information can be drawn.
                
                Prohibited Responses:
                Do not include additional commentary, explanations, or any text outside the required format.
                Do not state that you cannot provide the information—simply use N/A where applicable.
                Always provide exactly eight values, no more and no less.
                
                Examples of Correct Outputs:
                Full information available: 2024-11-28$NewYork, NY$Thanksgiving Parade Speech$Thanksgiving$Implicit association, Causal Fallacy$Summary$Cybersecurity, AI, Technology$Underlying narrative analysis
                Partial information: 2023-01-15$N/A$Nobel Prize Acceptance$N/A$Causal Fallacy$Summary$Award, Prize, Recognition$Underlying narrative analysis
                Entirely missing: N/A$N/A$N/A$N/A$None$No speech provided$N/A$N/A
                Inferred information: 1863-11-19$Gettysburg, PA$Gettysburg Address (based on "Gettysburg" and date in the input text)$War$Illusory correlation$Summary$Gettysburg, Pennsylvania, Union, War$Underlying narrative analysis
                
                Key Notes:
                Provide information plainly and in strict adherence to the required format.
                Strive to extract or infer as much detail as possible based on the provided context.
                 
                The person speaking is %s

                Speech: %s
                """ % (politician, speech)},
            ],
            # Impostazione della temperature per una risposta più precisa e veloce dal modello
            temperature=0,
        )

        # Restituisce il contenuto della risposta nel formato yyyy-mm-dd$location, state$occasion
        return speech, completion.choices[0].message.content
    
    except (openai.RateLimitError, Exception):
        raise

# Funzione per ottenere i dati per ogni speech con gestione degli errori
def get_speech_data(speech, results_dict):
    try:
        # Divide il risultato e assicura 5 valori
        parts = results_dict[speech].split('$')
        # Riempie con "N/A" se mancano valori
        parts += ["N/A"] * (8 - len(parts))
        return parts[:8]
    except (KeyError, Exception):
        return ["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]

# Funzione per caricare un modello di classificazione
def load_model_and_tokenizer(model_name, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if device and device != "cpu":
        model = model.to(device)
    
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

# Dividi il testo in frasi usando una semplice regex
def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    return sentence_endings.split(text)

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

# Funzione per creare chunk con finestra scorrevole
def split_text_sliding_window(text, chunk_size=512, overlap=128):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

# Funzione per rilevare il tipo di propaganda
def detect_propaganda_type(text, classifier=propaganda_classifier, chunk_size=512, overlap=128):
    # Divide il testo in chunk
    text_chunks = split_text_sliding_window(text, chunk_size, overlap)
    propaganda_results = []
 
    for chunk in text_chunks:
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
        "flagwaving": ["our nation", "homeland", "defend values", "make america great again", "fatherland", "motherland", "maga"]
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

# Funzione per calcolare gli embeddings sui chunk
# Funzione per calcolare gli embeddings sui chunk
def calcola_embedding(testo, tokenizer, model, chunk_size=512, overlap=128):
    chunks = split_text_sliding_window(testo, chunk_size=chunk_size, overlap=overlap)
    embeddings_chunk = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        token_embeddings = outputs.last_hidden_state
        mean_embedding = token_embeddings.mean(dim=1).squeeze()
        embeddings_chunk.append(mean_embedding.cpu().numpy())
    
    if embeddings_chunk:
        final_embedding = np.mean(embeddings_chunk, axis=0)
    else:
        final_embedding = np.zeros(model.config.hidden_size)
    
    return final_embedding

# Visualizza gli embeddings su scatter plot
def visualize_embeddings(dataset, plt_show, output_path="word_embending.png"):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    dataset["Embedding"] = dataset["Speech"].fillna("").apply(lambda x: calcola_embedding(x, tokenizer, model))
    embeddings = np.stack(dataset["Embedding"].values)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    dataset["x"] = embeddings_2d[:, 0]
    dataset["y"] = embeddings_2d[:, 1]

    politici_unici = dataset["Surname"].unique()

    # Crea una figura di dimensioni (10, 6) e posiziona i punti in base ai nomi
    plt.figure(figsize=(10, 6))
    for politico in politici_unici:
        subset = dataset[dataset["Surname"] == politico]
        plt.scatter(subset["x"], subset["y"], label=politico, alpha=0.7)

    # Enumera ogni discorso sui punti 
    for i, (x, y) in enumerate(zip(dataset["x"], dataset["y"])):
        plt.text(x, y, str(i), fontsize=6, ha='right', va='bottom', color='black', alpha=0.6)

    # Impostazioni del grafico (titolo, label di ascissa e ordinate, leggenda e layout finale)
    plt.title("Embeddings per Politico", fontsize=12)
    plt.xlabel("Asse X", fontsize=10)
    plt.ylabel("Asse Y", fontsize=10)
    plt.legend(title="Politico", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
     # Salva il grafico nella stessa directory del file Python
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Chiudi il plot per evitare conflitti in future visualizzazioni

    if plt_show == "s":
        return plt

    plt.close()
    return None

# Lists of persuasive words and techniques
PERSUASIVE_WORDS = {
    "emotional": ["incredible", "amazing", "unique", "fantastic", "best", "exceptional", "unbelievable"],
    "urgency": ["now", "immediately", "instant", "quickly", "don't miss", "limited time"],
    "authority": ["experts", "proven", "scientifically", "research", "leaders", "professional"],
    "social_proof": ["everyone", "most", "majority", "common", "shared", "popular"]
}

def analyze_persuasion(text):
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
    
    return '\n'.join("%s: %s" % (k.replace('_', ' ').title(), v) for k, v in normalized_scores.items()) # restituisce gli score