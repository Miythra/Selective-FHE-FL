import torch 
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def load_datasets(num_clients=3):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    # Split the training set into num_clients subsets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    # on met les images de trop pour le 3eme cleint
    lengths[-1] += len(trainset) % num_clients

    datasets_per_client= random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))
    return datasets_per_client, testset

if __name__ == "__main__":
    clients_data, test_data = load_datasets()
    print(f"Nombre de clients : {len(clients_data)}")