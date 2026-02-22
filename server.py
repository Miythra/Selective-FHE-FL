import flwr as fl

# On choisit l'algorithme "FedAvg" (Faire la moyenne de tous les clients)
strategy = fl.server.strategy.FedAvg(
    # On demande à 100% (1.0) des clients de s'entraîner à chaque round
    fraction_fit=1.0,  
    
    # On demande à 100% (1.0) des clients de tester le modèle à chaque round
    fraction_evaluate=1.0,  
    
    # Le serveur refusera de lancer l'entraînement s'il n'a pas exactement 3 clients connectés
    min_fit_clients=3,  
    
    # Pareil pour l'évaluation
    min_evaluate_clients=3, 
    
    # Nombre total de clients qu'on attend sur notre réseau local
    min_available_clients=3, 
)

# 2. LANCEMENT DU SERVEUR
if __name__ == "__main__":
    print("🟢 Démarrage du Serveur Flower sur le port 8080...")
    print("⏳ En attente de la connexion des 3 clients...")
    
    # On allume la machine !
    fl.server.start_server(
        server_address="127.0.0.1:8080", # 127.0.0.1 = L'adresse de ton propre ordinateur
        # "num_rounds=3" : On va faire 3 allers-retours (entraînements) complets
        config=fl.server.ServerConfig(num_rounds=3), 
        strategy=strategy,
    )