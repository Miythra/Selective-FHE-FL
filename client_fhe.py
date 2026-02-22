import sys
import flwr as fl
import torch
from torch.utils.data import DataLoader
import tenseal as ts
import numpy as np

import model
import data_utils

# --- 1. IDENTIFICATION DU CLIENT ---
try:
    client_id = int(sys.argv[1])
except IndexError:
    client_id = 0

print(f"--- Démarrage du Client Sécurisé n°{client_id} ---")

# --- 2. CHARGEMENT DU CONTEXTE SECRET ---
print("Chargement du contexte cryptographique complet...")
with open("secret_context.bytes", "rb") as f:
    # Ce fichier contient déjà tout ce qu'il faut, y compris la clé secrète !
    context = ts.context_from(f.read())

# --- 3. PRÉPARATION DU CERVEAU ET DES DONNÉES ---
datasets_per_client, testset = data_utils.load_datasets(num_clients=3)
trainloader = DataLoader(datasets_per_client[client_id], batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32)
net = model.Net()

# =====================================================================
# --- LA MAGIE DU CHIFFREMENT (LES FONCTIONS COMPLEXES) ---
# =====================================================================

def get_encrypted_parameters(net, context):
    """Étape 1 : Démonter le château, le couper en morceaux et le chiffrer."""
    
    # 1. APLATIR LE MODÈLE (Le mettre en file indienne)
    tous_les_poids = [] # Une grande liste vide pour accueillir nos pièces de LEGO
    
    # On parcourt toutes les couches du cerveau (.values() récupère juste les nombres)
    for val in net.state_dict().values():
        # .flatten() : Écrase le bloc 2D/3D en une ligne 1D.
        # .tolist() : Transforme le format PyTorch complexe en une simple liste Python.
        # .extend() : Ajoute ces pièces à la suite de notre grande liste (contrairement à .append qui ferait des listes dans des listes).
        tous_les_poids.extend(val.flatten().tolist())

    # 2. DÉCOUPER ET CHIFFRER
    taille_max = 4096 # C'est la limite physique de notre boîte CKKS
    blocs_chiffres_numpy = [] # La liste des paquets qu'on va envoyer au serveur
    
    # Boucle spéciale : On avance de 4096 en 4096. 
    # Si on a 62000 poids, i vaudra 0, puis 4096, puis 8192, etc.
    for i in range(0, len(tous_les_poids), taille_max):
        # On découpe une "tranche" de la grande liste (de 'i' jusqu'à 'i + 4096')
        bloc = tous_les_poids[i : i + taille_max]
        
        # On chiffre ce bloc avec CKKS ! Le résultat est un objet mathématique complexe.
        vecteur_chiffre = ts.ckks_vector(context, bloc)
        
        # Problème : Le réseau internet et Flower ne savent pas transporter cet objet complexe.
        # Solution : On le sérialise (.serialize()), c'est-à-dire qu'on le transforme en une suite d'octets de base.
        # Puis on le met dans un tableau NumPy (np.uint8) car Flower adore NumPy.
        bytes_array = np.frombuffer(vecteur_chiffre.serialize(), dtype=np.uint8)
        
        # On ajoute ce colis prêt à être expédié à notre pile de colis
        blocs_chiffres_numpy.append(bytes_array)
        
    return blocs_chiffres_numpy # On donne la pile de colis chiffrés à Flower


def set_encrypted_parameters(net, context, parameters_bytes):
    """Étape 2 : Déchiffrer les colis reçus du serveur et reconstruire le château."""
    
    # 1. DÉCHIFFRER ET REMETTRE EN FILE INDIENNE
    tous_les_poids_dechiffres = []
    
    # Pour chaque colis chiffré reçu du serveur...
    for b in parameters_bytes:
        # On retransforme le simple tableau NumPy en véritable objet mathématique CKKS
        vecteur_chiffre = ts.ckks_vector_from(context, b.tobytes())
        
        # On déchiffre ! La fonction utilise toute seule la clé secrète qu'on a chargée au début.
        bloc_clair = vecteur_chiffre.decrypt() 
        
        # On ajoute les nombres déchiffrés à notre grande file indienne
        tous_les_poids_dechiffres.extend(bloc_clair)

    # 2. RECONSTRUIRE LE CHÂTEAU 3D (Redonner la bonne forme)
    nouveau_state_dict = {} # Le nouveau plan du cerveau
    index_actuel = 0 # Notre position dans la file indienne
    
    # .items() nous donne le nom de la couche (ex: 'conv1.weight') et sa forme (val)
    for key, val in net.state_dict().items():
        # Combien de pièces de LEGO faut-il pour reconstruire cette couche précise ?
        nb_elements = val.numel() 
        
        # On prend exactement le bon nombre de pièces dans notre file indienne
        tranche = tous_les_poids_dechiffres[index_actuel : index_actuel + nb_elements]
        
        # .reshape(val.shape) : La vraie magie de PyTorch ! On ordonne à cette ligne 
        # plate de reprendre la forme exacte (2D ou 3D) de la couche d'origine.
        nouveau_tenseur = torch.tensor(tranche).reshape(val.shape)
        
        # On range la couche reconstruite dans le nouveau plan
        nouveau_state_dict[key] = nouveau_tenseur
        
        # On avance notre curseur pour la prochaine couche
        index_actuel += nb_elements
        
    # On remplace l'ancien cerveau par ce nouveau cerveau tout neuf
    net.load_state_dict(nouveau_state_dict, strict=True)


# =====================================================================
# --- LE CLIENT FLOWER (Le messager) ---
# =====================================================================
class FlowerClientSecure(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        # Quand le serveur réclame les poids, on lance notre grosse fonction de chiffrement !
        return get_encrypted_parameters(net, context)

    def fit(self, parameters, config):
        print(">> 🔒 Réception du modèle chiffré... Déchiffrement en cours...")
        # On déchiffre et reconstruit le modèle envoyé par le serveur
        set_encrypted_parameters(net, context, parameters)
        
        print(">> 🧠 Entraînement local...")
        # L'IA s'entraîne normalement en clair sur ton ordinateur
        model.train(net, trainloader, epochs=1)
        
        print(">> 🔐 Chiffrement total du modèle pour envoi au serveur... (Ça va prendre du temps !)")
        # On rechiffre tout pour le renvoyer discrètement
        return get_encrypted_parameters(net, context), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Même chose, mais juste pour passer l'examen (le test)
        set_encrypted_parameters(net, context, parameters)
        loss, accuracy = model.test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

# Lancement du client sur le réseau local
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClientSecure())