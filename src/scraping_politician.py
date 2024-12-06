import pandas as pd
import requests
import wikipedia
from bs4 import BeautifulSoup

# Crea il dizionario di cache dei politici
politici = {}
'''
struttura = {
    cognome : {
        soup: codice,
        name: nome,
        birthday: data,
        birthplace: luogo,
        deathday: data,
        deathplace: luogo,
        party: partito,
    }
}
'''

# Imposta la pagina di wikipedia in inglese
wikipedia.set_lang("en")

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


if __name__ == "__main__":
    import time

    # Carica il percorso del dataset
    data_folder = "src/dataset/speech-a.tsv"

    # Legge il dataset tsv
    dataset = pd.read_table(data_folder, sep = '\t', header = None)
                
    # Da il nome alle colonne del dataset
    dataset.columns = ["Surname", "Code", "Speech"]

    # Inizio della misurazione
    start_time = time.time()

    # Trova i dati dei politici e li aggiunge al dataset
    dataset["Full name"] = dataset["Surname"].apply(lambda x: get_info(x, "name"))
    dataset["Birthday"] = dataset["Surname"].apply(lambda x: get_info(x, "birthday"))
    dataset["Birth place"] = dataset["Surname"].apply(lambda x: get_info(x, "birthplace"))
    dataset["Death day"] = dataset["Surname"].apply(lambda x: get_info(x, "deathday"))
    dataset["Death place"] = dataset["Surname"].apply(lambda x: get_info(x, "deathplace"))
    dataset["Political party"] = dataset["Surname"].apply(lambda x: get_info(x, "party"))

    # Fine della misurazione
    end_time = time.time()
    
    # Stampa i dati senza i discorsi
    print(dataset[["Surname", "Full name", "Birthday", "Birth place", "Death day", "Death place", "Political party"]])

     # Calcolo del tempo
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print("Tempo di esecuzione: %d ore, %d minuti, %.2f secondi" % (hours, minutes, seconds))