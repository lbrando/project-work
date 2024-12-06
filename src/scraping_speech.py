import os
import openai

# Legge la API key
os.environ["OPENAI_API_KEY"] = open("src/api key/API-GLHF.txt", "r").read()

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
                Story telling: Analyze the political speech and provide a comprehensive breakdown of its underlying narrative. Follow these steps and provide them in a bullet list, with the following format: 'Conext: description.;Rhetorical Structure: description.;Main Themes and Values: description.;...'. Always provide ";" between each step and do not use spaces alongside it. Use a fullstop at the end of every description. Do never provide '\n'. If the speech is too short to analyze, provide 'N/A'.
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

if __name__ == "__main__":
    import time
    import concurrent.futures
    import pandas as pd
    
    # Carica il dataset
    data_folder = "src/dataset/speech-filtered.tsv"
    dataset = pd.read_table(data_folder, sep='\t', header=None)

    # Assegna i nomi delle colonne
    dataset.columns = ["Surname", "Code", "Speech"]
    
    # Inizio della misurazione
    start_time = time.time()
    
    # Usa ThreadPoolExecutor per l'elaborazione parallela
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        # Mappa le chiamate API in parallelo
        results = list(executor.map(speech_info, dataset["Surname"], dataset["Speech"]))
    
    # Crea un dizionario per mappare i risultati
    results_dict = dict(results)
    
    # Crea le nuove colonne in modo sicuro
    speech_data = dataset["Speech"].apply(lambda x: get_speech_data(x, results_dict)).tolist()
    
    # Separa le colonne
    dataset["Speech Date"] = [row[0] for row in speech_data]
    dataset["Speech location"] = [row[1] for row in speech_data]
    dataset["Speech occasion"] = [row[2] for row in speech_data]
    dataset["Topic"] = [row[3] for row in speech_data]
    dataset["Cognitive Bias"] = [row[4] for row in speech_data]
    dataset["Summary"] = [row[5] for row in speech_data]
    dataset["Keywords"] = [row[6] for row in speech_data]
    dataset["Underlying narrative"] = [row[7].replace(";", "\n\n") for row in speech_data]

    # Fine della misurazione
    end_time = time.time()
    
    # Stampa risultati
    print(dataset[["Surname", "Speech Date", "Speech location", "Speech occasion", "Topic", "Cognitive Bias", "Summary", "Keywords", "Underlying narrative"]])
    
    # Calcolo del tempo di esecuzione
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print("Tempo di esecuzione: %d ore, %d minuti, %.2f secondi" % (hours, minutes, seconds))
    
    # Salva il dataset
    dataset.to_csv("src/dataset/speech-info-1.csv", index=False, sep=",")