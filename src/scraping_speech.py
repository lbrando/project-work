import concurrent.futures
import os
import openai
import pandas as pd

# Legge la API key
os.environ["OPENAI_API_KEY"] = open("src/api key/API Key GLHF.txt", "r").read()

def speech_info(speech):
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
                Extract the following information from the provided speech or text, formatted as yyyy-mm-dd$location, state$occasion:

                Date: The exact date the speech was delivered, formatted as yyyy-mm-dd.
                Location: The city or venue where the speech was delivered, followed by the state, separated by a comma (,). If the state is unavailable, provide only the city or venue, do not provide N/A.
                Occasion: The specific event, purpose, or context for which the speech was delivered.
                
                Output Format:
                Always provide the information in the exact order: date$location, state$occasion.
                Separate each piece of information strictly with the $ character, without spaces before or after it.
                If a specific piece of information cannot be determined, use N/A for that field.
                
                Rules for Extraction:
                Extract as much information as possible from the text. If one or two fields can be determined but others cannot, provide the known fields and use N/A only for those that are truly unavailable.
                Use logical inference when explicit information is missing. For example, infer the state if the city is well-known, or infer the occasion based on context (e.g., "July 4th" implies Independence Day). Ensure the inference is reasonable and based only on the provided text.
                Avoid using N/A unless absolutely necessary, and only if no reasonable inference or information can be drawn.
                
                Prohibited Responses:
                Do not include additional commentary, explanations, or any text outside the required format.
                Do not state that you cannot provide the information—simply use N/A where applicable.
                Always provide exactly three values, no more and no less.
                
                Examples of Correct Outputs:
                Full information available: 2024-11-28$NewYork, NY$Thanksgiving Parade Speech
                Partial information: 2023-01-15$N/A$Nobel Prize Acceptance
                Entirely missing: N/A$N/A$N/A
                Inferred information: 1863-11-19$Gettysburg, PA$Gettysburg Address (based on "Gettysburg" and date in the input text).
                
                Key Notes:
                Provide information plainly and in strict adherence to the required format.
                Strive to extract or infer as much detail as possible based on the provided context.

                Speech: %s
                """ % speech},
            ],
            # Impostazione della temperature per una risposta più precisa e veloce dal modello
            temperature=0,
        )

        # Restituisce il contenuto della risposta nel formato yyyy-mm-dd$location, state$occasion
        return speech, completion.choices[0].message.content
    
    except openai.RateLimitError:
        print("Rate limit exceeded.")

# Funzione per ottenere i dati per ogni speech con gestione degli errori
def get_speech_data(speech):
    try:
        # Divide il risultato e assicura 3 valori
        parts = results_dict[speech].split('$')
        # Riempie con "N/A" se mancano valori
        parts += ["N/A"] * (3 - len(parts))
        return parts[:3]
    except (KeyError, Exception):
        return ["N/A", "N/A", "N/A"]


if __name__ == "__main__":
    import time
    
    # Carica il dataset
    data_folder = "src/dataset/speech-filtered.tsv"
    dataset = pd.read_table(data_folder, sep='\t', header=None)

    # Assegna i nomi delle colonne
    dataset.columns = ["Surname", "Code", "Speech"]
    
    # Inizio della misurazione
    start_time = time.time()
    
    # Usa ThreadPoolExecutor per l'elaborazione parallela
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Mappa le chiamate API in parallelo
        results = list(executor.map(speech_info, dataset["Speech"]))
    
    # Crea un dizionario per mappare i risultati
    results_dict = dict(results)
    
    # Crea le nuove colonne in modo sicuro
    speech_data = dataset["Speech"].apply(get_speech_data).tolist()
    
    # Separa le colonne
    dataset["Speech Date"] = [row[0] for row in speech_data]
    dataset["Speech location"] = [row[1] for row in speech_data]
    dataset["Speech occasion"] = [row[2] for row in speech_data]
    
    # Fine della misurazione
    end_time = time.time()
    
    # Stampa risultati
    print(dataset[["Surname", "Speech Date", "Speech location", "Speech occasion"]])
    
    # Calcolo del tempo di esecuzione
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print(f"Tempo di esecuzione: {hours} ore, {minutes} minuti, {seconds:.2f} secondi")
    
    # Salva il dataset
    dataset.to_csv("src/dataset/speech-info.csv", index=False, sep=",")