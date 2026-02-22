import sys
import flwr as fl
import torch
from torch.utils.data import DataLoader

import model
import data_utils

try:
    client_id = int(sys.argv[1])
except IndexError:
    print("Oups ! Tu as oublié de donner un numéro de client. J'utilise le client 0 par défaut.")
    client_id = 0

print(f"--- Démarrage du Client n°{client_id} ---")

datasets_per_client, testset = data_utils.load_datasets(num_clients=3)
client_dataset = datasets_per_client[client_id]

trainloader = DataLoader(client_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32)
net = model.Net()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(">> Réception du modèle global... Début de l'entraînement local.")
        self.set_parameters(parameters)
        model.train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(">> Évaluation du modèle global...")
        self.set_parameters(parameters)
        loss, accuracy = model.test(net, testloader)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
    

if __name__ == "__main__":
    # On connecte ce client au serveur local (127.0.0.1) sur le port 8080
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=FlowerClient()
    )