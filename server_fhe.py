import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import tenseal as ts
import numpy as np

print("--- ☁️  Démarrage du Serveur Sécurisé ---")
print("Chargement du contexte public (Le serveur n'a pas la clé secrète !)")

# Le serveur charge UNIQUEMENT le contexte public. 
# Il a le droit de calculer, mais pas le droit de déchiffrer.
with open("public_context.bytes", "rb") as f:
    context = ts.context_from(f.read())


# --- CRÉATION DE NOTRE PROPRE ALGORITHME D'AGRÉGATION ---
# On "hérite" de la stratégie classique FedAvg, mais on va réécrire sa fonction principale.
class SecureFedAvg(fl.server.strategy.FedAvg):
    
    # C'est cette fonction que Flower appelle quand tous les clients ont envoyé leurs calculs
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
            
        print(f"\n[Round {server_round}] 🔐 Agrégation homomorphe en cours...")
        
        # 1. EXTRAIRE LES DONNÉES
        # "results" contient les colis de tous les clients. 
        # On les transforme en tableaux NumPy utilisables.
        donnees_clients = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        
        nb_clients = len(donnees_clients)
        fraction = 1.0 / nb_clients # Pour faire la moyenne (ex: diviser par 3, c'est multiplier par 1/3)
        
        blocs_agreges_chiffres = []
        nb_blocs_total = len(donnees_clients[0]) # Combien de blocs de 4096 a-t-on par client ?
        
        # 2. LA MATHÉMATIQUE HOMOMORPHE AVEUGLE
        # Pour chaque bloc du modèle (du 1er au dernier)
        for i in range(nb_blocs_total):
            somme_chiffree = None
            
            # On additionne ce bloc précis pour les 3 clients
            for blocs_d_un_client in donnees_clients:
                
                # On transforme le colis NumPy (octets) en véritable objet mathématique CKKS
                bytes_array = blocs_d_un_client[i]
                bloc_ckks = ts.ckks_vector_from(context, bytes_array.tobytes())
                
                # L'ADDITION MAGIQUE
                if somme_chiffree is None:
                    somme_chiffree = bloc_ckks
                else:
                    somme_chiffree = somme_chiffree + bloc_ckks # On additionne deux éléments chiffrés !
            
            # LA MULTIPLICATION MAGIQUE (Pour faire la moyenne)
            # On multiplie notre résultat chiffré par un nombre en clair (1/3)
            moyenne_chiffree = somme_chiffree * fraction
            
            # On remet le résultat dans un colis NumPy (octets) pour le renvoyer
            bytes_res = np.frombuffer(moyenne_chiffree.serialize(), dtype=np.uint8)
            blocs_agreges_chiffres.append(bytes_res)
            
        print(f"[Round {server_round}] ✅ Agrégation terminée ! (Le serveur n'a rien vu en clair)")
        
        # On renvoie les paramètres sous le format spécial de Flower
        return ndarrays_to_parameters(blocs_agreges_chiffres), {}

# --- LANCEMENT DU SERVEUR ---
# On configure notre nouvelle stratégie
strategy = SecureFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=2), # On limite à 2 rounds car c'est long !
        strategy=strategy,
    )