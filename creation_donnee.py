import pandas as pd
import numpy as np

champions = {

    "NULL": -1,
    "Aatrox": 0, 
    "Ahri": 1, 
    "Akali": 2, 
    "Akshan": 3, 
    "Alistar": 4, 
    "Amumu": 5, 
    "Anivia": 6,
    "Annie": 7, 
    "Aphelios": 8, 
    "Ashe": 9, 
    "Aurelion Sol": 10, 
    "Azir": 11, "Bard": 12,
    "Bel'Veth": 13, 
    "Blitzcrank": 14, 
    "Brand": 15, 
    "Braum": 16, 
    "Briar": 17, 
    "Caitlyn": 18,
    "Camille": 19, 
    "Cassiopeia": 20, 
    "Cho'Gath": 21, 
    "Corki": 22, 
    "Darius": 23, 
    "Diana": 24,
    "Dr. Mundo": 25, 
    "Draven": 26, 
    "Ekko": 27, 
    "Elise": 28, 
    "Evelynn": 29, 
    "Ezreal": 30,
    "Fiddlesticks": 31, 
    "Fiora": 32, 
    "Fizz": 33, 
    "Galio": 34, 
    "Gangplank": 35, 
    "Garen": 36,
    "Gnar": 37, 
    "Gragas": 38, 
    "Graves": 39, 
    "Gwen": 40, 
    "Hecarim": 41, 
    "Heimerdinger": 42,
    "Hwei": 43, 
    "Illaoi": 44, 
    "Irelia": 45, 
    "Ivern": 46, 
    "Janna": 47, 
    "Jarvan IV": 48,
    "Jax": 49, 
    "Jayce": 50, 
    "Jhin": 51, 
    "Jinx": 52, 
    "K'Sante": 53, 
    "Kai'Sa": 54, 
    "Kalista": 55,
    "Karma": 56, 
    "Karthus": 57, 
    "Kassadin": 58, 
    "Katarina": 59, 
    "Kayle": 60, 
    "Kayn": 61,
    "Kennen": 62, 
    "Kha'Zix": 63, 
    "Kindred": 64, 
    "Kled": 65, 
    "Kog'Maw": 66, 
    "LeBlanc": 67,
    "Lee Sin": 68, 
    "Leona": 69, 
    "Lillia": 70, 
    "Lissandra": 71, 
    "Lucian": 72, 
    "Lulu": 73,
    "Lux": 74, 
    "Maître Yi": 75, 
    "Malphite": 76, 
    "Malzahar": 77, 
    "Maokai": 78, 
    "Milio": 79,
    "Miss Fortune": 80, 
    "Mordekaiser": 81, 
    "Morgana": 82, 
    "Naafiri": 83, 
    "Nami": 84,
    "Nasus": 85, 
    "Nautilus": 86, 
    "Neeko": 87, 
    "Nidalee": 88, 
    "Nilah": 89, 
    "Nocturne": 90,
    "Nunu et Willump": 91, 
    "Olaf": 92, 
    "Orianna": 93, 
    "Ornn": 94, 
    "Pantheon": 95, 
    "Poppy": 96,
    "Pyke": 97, 
    "Qiyana": 98, 
    "Quinn": 99, 
    "Rakan": 100, 
    "Rammus": 101, 
    "Rek'Sai": 102,
    "Rell": 103, 
    "Renata Glasc": 104, 
    "Renekton": 105, 
    "Rengar": 106, 
    "Riven": 107,
    "Rumble": 108, 
    "Ryze": 109, 
    "Samira": 110, 
    "Sejuani": 111, 
    "Senna": 112, 
    "Seraphine": 113,
    "Sett": 114, 
    "Shaco": 115, 
    "Shen": 116, 
    "Shyvana": 117, 
    "Singed": 118, 
    "Sion": 119,
    "Sivir": 120, 
    "Skarner": 121, 
    "Smolder": 122, 
    "Sona": 123, 
    "Soraka": 124, 
    "Swain": 125,
    "Sylas": 126, 
    "Syndra": 127, 
    "Tahm Kench": 128, 
    "Taliyah": 129, 
    "Talon": 130, 
    "Taric": 131,
    "Teemo": 132, 
    "Thresh": 133, 
    "Tristana": 134, 
    "Trundle": 135, 
    "Tryndamere": 136,
    "Twisted Fate": 137, 
    "Twitch": 138, 
    "Udyr": 139, 
    "Urgot": 140, 
    "Varus": 141, 
    "Vayne": 142,
    "Veigar": 143, 
    "Vel'Koz": 144, 
    "Vex": 145, 
    "Vi": 146, 
    "Viego": 147,
    "Viktor": 148,
    "Vladimir": 149, 
    "Volibear": 150, 
    "Warwick": 151, 
    "Wukong": 152, 
    "Xayah": 153, 
    "Xerath": 154,
    "Xin Zhao": 155, 
    "Yasuo": 156, 
    "Yone": 157, 
    "Yorick": 158, 
    "Yuumi": 159, 
    "Zac": 160,
    "Zed": 161, 
    "Zeri": 162, 
    "Ziggs": 163, 
    "Zilean": 164, 
    "Zoe": 165, 
    "Zyra": 166
}





def read_column_in_chunks(file_path, column_letter, chunk_size=12):
    # Lire le fichier Excel
    df = pd.read_excel(file_path)

    # Convertir la lettre de colonne en indice de colonne (0-indexé)
    column_index = ord(column_letter.upper()) - ord('A')

    colonne_win = ord('Y'.upper()) - ord('A')


    # Nombre total de lignes
    total_rows = df.shape[0]

    # Exclure les deux dernières lignes
    total_rows -= 2

    # Liste pour stocker les chunks
    chunks = []
    win_chunks = []
    # Lire par blocs de chunk_size lignes
    for start_row in range(0, total_rows, chunk_size):
        
        end_row = min(start_row + chunk_size, total_rows)

        chunk = df.iloc[start_row:end_row, column_index].tolist()

        first_element = df.iloc[start_row, colonne_win]

        # Supprimer les 'nan' à la fin du chunk
        while chunk and pd.isna(chunk[-1]):
            chunk.pop()

        # Ajouter le chunk à la liste des chunks si ce n'est pas vide
        if chunk:
            chunks.append(chunk)

        if first_element == 1:
            win_chunks.append(chunk[:5].copy())
        else:
            win_chunks.append(chunk[5:].copy())



    return chunks, win_chunks

def chunks_to_one_hot(chunks, taille_equipe):
    num_champions = 168
    one_hot_chunks = []

    for chunk in chunks:
        one_hot_vector = [0] * (num_champions * taille_equipe)  # Initialiser une liste de zéros de taille 1680
        for i, champion in enumerate(chunk):
            index = champions.get(champion, -1)  # Obtenir l'indice du champion
            if index != -1:
                one_hot_vector[i * num_champions + index] = 1  # Mettre 1 à la position correspondante

        one_hot_chunks.append(one_hot_vector)

    return one_hot_chunks

def display_one_hot_chunks(one_hot_chunks):
    for chunk in one_hot_chunks:
        for i in range(0, len(chunk), 168):
            print(chunk[i:i+168])
        print()  # Ajouter une ligne vide entre les chunks pour une meilleure lisibilité


if __name__ == "__main__":

    # Spécifiez le chemin vers votre fichier Excel
    file_path = "2024_game.xlsx"
    
    # Lettre de la colonne à lire (R dans ce cas)
    column_letter = 'R'
 

######################################## creer le dataset
    # Lire les chunks de la colonne et les stocker dans une liste
    # chunks, win_chunks = read_column_in_chunks(file_path, column_letter)

    # one_hot_chunks = chunks_to_one_hot(chunks, 10)
    # one_hot_win = chunks_to_one_hot(win_chunks, 5)
    # # print(one_hot_chunks[0])
    # # display_one_hot_chunks(one_hot_chunks)
    # # print(one_hot_win)
    # one_hot_all_chunks_np = np.array(one_hot_chunks)
    # one_hot_chunks_gagnants_np = np.array(one_hot_win)

    # np.save('one_hot_all_chunks.npy', one_hot_all_chunks_np)
    # np.save('one_hot_chunks_gagnants.npy', one_hot_chunks_gagnants_np)
    ######################

    test = [["NULL","Jarvan IV","NULL","NULL","NULL","K'sante","Nidalee","Syndra","Smolder","Karma"]]

    one = chunks_to_one_hot(test, 10)
    print(one)