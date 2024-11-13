import pandas as pd
import requests, wikipedia
from bs4 import BeautifulSoup

# Carica il percorso del dataset
data_folder = "speech-a.tsv"

# Legge il dataset tsv
dataset = pd.read_table(data_folder, sep = '\t', header = None)

# Dizionario dei politici
'''
struttura = {
    cognome : {
        wiki: link,
        name: nome,
        birthday: data,
        birthplace: luogo,
    }
}
'''

politici = {}

# Imposta la pagina di wikipedia in inglese
wikipedia.set_lang("en")

# Verifica che la pagina si riferisca ad un politico
def verify_politician(titolo):
    try:
        # Prende la prima frase del riassunto della pagina
        summary = wikipedia.summary(titolo, sentences=1)
        keywords = ["who", "president", "politician", "president-elect", "statesman"]
        # Filtra per le parole chiave contenute nella prima frase
        return any(keyword in summary.lower() for keyword in keywords)
    except wikipedia.exceptions.DisambiguationError as e:
        pass
    except wikipedia.exceptions.PageError as e:
        pass

# Funzione per trovare la pagina di wikipedia
def get_wiki(cognome):

    if cognome not in politici:
        # Cerca il cognome dato
        results = wikipedia.search(cognome)

        # Trova l'indice del risultato del politico
        index = next((i for i, result in enumerate(results) if verify_politician(result)), None)

        # URL della pagina wikipedia
        url = wikipedia.page(results[index]).url

        # Effettua una richiesta GET alla pagina
        response = requests.get(url)

        # Verifica che la richiesta ha avuto successo
        if response.status_code == 200:
            # Aggiunge il link della pagina al dizionario
            politici[cognome] =  {"wiki" : response}
            # Restituisce la pagina scaricata con request
            return response
        
        else:
            print("Errore nella richiesta della pagina:", response.status_code)  
    else:
        return politici[cognome]["wiki"]

def get_name(wiki, cognome):

    if "name" not in politici[cognome]:
        # Crea l'oggetto BeautifulSoup per analizzare l'HTML
        soup = BeautifulSoup(wiki.text, "html.parser")
        # Estrai il nome completo
        name = soup.find("div", {"class": "nickname"}).text
        # Aggiunge il nome al dizionario
        politici[cognome]["name"] = name
        # Restituisce il nome
        return name
    else:
        return politici[cognome]["name"]

def get_birthday(wiki, cognome):

    if "birthday" not in politici[cognome]:
        # Crea l'oggetto BeautifulSoup per analizzare l'HTML
        soup = BeautifulSoup(wiki.text, "html.parser")
        # Estrai la data di nascita
        birth_date = soup.find("span", {"class": "bday"}).text
        # Aggiunge il compleanno al dizionario
        politici[cognome]["birthday"] = birth_date
        # Restituisce la data di compleanno
        return birth_date
    else:
        return politici[cognome]["birthday"]

def get_birtplace(wiki, cognome):

    if "birthplace" not in politici[cognome]:
        # Crea l'oggetto BeautifulSoup per analizzare l'HTML
        soup = BeautifulSoup(wiki.text, "html.parser")

        # Trova l'infobox contenente le rows in cui è presente il luogo di nascita
        infobox_rows = soup.find("table", {"class": "infobox"}).find_all("tr")

        # Estrae il luogo di nascita
        birth_place = next(
            # Prende il link del luogo di nascita e lo unisce allo Stato di nascita
            (f"{row.find('a').text}{row.find('a').find_next_sibling(string=True).text}"
            for row in infobox_rows if "Born" in row.get_text()), 
            None
        )

        # Aggiunge il luogo di nascita al dizionario
        politici[cognome]["birthplace"] = birth_place
        # Restituisce il luogo di nascita
        return birth_place
    else:
        return politici[cognome]["birthplace"]
    

# Da il nome alle colonne del dataset
dataset.columns = ["Name", "Code", "Speech"]

# Test per trovare la data di compleanno
# print(dataset["Name"].apply(lambda x: get_birthday(get_wiki(x), x)))

candidato = "Trump"
get_name(get_wiki(candidato), candidato)
get_birthday(get_wiki(candidato), candidato)
get_birtplace(get_wiki(candidato), candidato)
candidato = "Obama"
get_name(get_wiki(candidato), candidato)
get_birthday(get_wiki(candidato), candidato)
get_birtplace(get_wiki(candidato), candidato)

print(politici)