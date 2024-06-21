import torch
import torch.nn as nn
from creation_donnee import champions
from creation_donnee import chunks_to_one_hot
import numpy as np
# Définir les hyperparamètres
input_size = 168 * 10  # 168 personnages * 10 (5 alliés + 5 ennemis)
hidden_size = 10000  # Taille de la couche cachée
output_size = 168 * 5  # 168 personnages * 5 (pour les 5 choix de sortie)

# Définir le réseau de neurones
class LoLCharacterPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LoLCharacterPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 5000)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(5000, output_size)
        self.sigmoid = nn.Sigmoid()  # Utiliser sigmoid pour une sortie entre 0 et 1

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Initialiser le modèle
model = LoLCharacterPredictor(input_size, hidden_size, output_size)

# Charger les poids du modèle sauvegardé
model.load_state_dict(torch.load('lol_character_predictor.pth'))

# Mettre le modèle en mode évaluation
model.eval()

# Exemple de données d'entrée
# Remplacez ceci par les nouvelles données de jeu pour lesquelles vous souhaitez faire des prédictions
champ = [["NULL","NULL","NULL","NULL","NULL",
          
          "NULL","Vi","NULL","Aphelios","NULL"]]

new_data = chunks_to_one_hot(champ,10)
# Convertir les données en tenseur PyTorch
input_tensor = torch.tensor(new_data, dtype=torch.float32)

# Faire des prédictions avec le modèle
with torch.no_grad():
    predictions = model(input_tensor)

# Transformer les prédictions
transformed_predictions = predictions.clone()

transformed_arrays = np.reshape(transformed_predictions, (5, 168))
argmax_indices = np.argmax(transformed_arrays, axis=1)

champ_clone = new_data.copy()
champ_clone = np.reshape(champ_clone, (10, 168))

binary_arrays = np.zeros_like(transformed_arrays)
for i in range(transformed_arrays.shape[0]):
    binary_arrays[i, argmax_indices[i]] = 1


def encode_to_champion_list(encoded_array, champions_dict):
    champion_names = []
    champ_numbers = []

    num_rows = encoded_array.shape[0]

    for i in range(num_rows):
        # Trouver l'indice du maximum dans la ligne actuelle du tableau encodé en one-hot
        max_index = np.argmax(encoded_array[i])

        if np.max(encoded_array[i]) == 0:
            champ_numbers.append(-1)
        else:
            champ_numbers.append(max_index)
        
        # Trouver le nom du champion correspondant à cet indice
        champion_name = list(champions_dict.keys())[list(champions_dict.values()).index(max_index)]
        
        # Ajouter le nom du champion à la liste
        champion_names.append(champion_name)

    return champion_names, champ_numbers


test_out, champ_numbers_out = encode_to_champion_list(binary_arrays, champions)
test_in, champ_number_in = encode_to_champion_list(champ_clone, champions)

print(champ_number_in, champ_numbers_out)


for i in range(5):
    if champ_number_in[5+i] >=0:
        if champ_number_in[5+i] in champ_numbers_out:
            print('aaa ',champ_number_in[5+i])
            for j in range(5):
                transformed_arrays[j][champ_number_in[5+i]] = 0 
                argmax_indices = np.argmax(transformed_arrays, axis=1)


print(transformed_arrays[1][146])

argmax_indices = np.argmax(transformed_arrays, axis=1)
print('info ', argmax_indices)

binary_arrays = np.zeros_like(transformed_arrays)
for i in range(transformed_arrays.shape[0]):
    binary_arrays[i, argmax_indices[i]] = 1

test_out, champ_numbers_out = encode_to_champion_list(binary_arrays, champions)

print(test_out)