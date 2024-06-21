import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Définir les hyperparamètres
input_size = 168 * 10  # 168 personnages * 10 (5 alliés + 5 ennemis)
hidden_size = 10000  # Taille de la couche cachée
output_size = 168 * 5  # 168 personnages * 5 (pour les 5 choix de sortie)
learning_rate = 0.001
num_epochs = 500

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

# Initialiser le modèle, la fonction de perte et l'optimiseur
model = LoLCharacterPredictor(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Exemple de données d'entraînement
# Les données doivent être sous forme de vecteurs one-hot pour les 10 personnages sélectionnés
# Ici, nous utilisons des données fictives pour illustration
# Remplacez ceci par vos propres données de jeu

loaded_one_hot_all_chunks_np = np.load('one_hot_all_chunks.npy')
loaded_one_hot_chunks_gagnants_np = np.load('one_hot_chunks_gagnants.npy')

x_train = torch.tensor(loaded_one_hot_all_chunks_np, dtype=torch.float32)
y_train = torch.tensor(loaded_one_hot_chunks_gagnants_np, dtype=torch.float32)

# x_train = torch.rand(100, input_size)  # 100 échantillons d'entraînement
# y_train = torch.rand(100, output_size)  # 100 échantillons de sorties cibles

# Entraînement du modèle
for epoch in range(num_epochs):
    model.train()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Entraînement terminé.")

# Sauvegarder le modèle
torch.save(model.state_dict(), 'lol_character_predictor.pth')

