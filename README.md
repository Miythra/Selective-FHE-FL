# Selective-FHE-FL: Privacy-Preserving Federated Learning

## Overview
This research project explores the intersection of **Federated Learning (FL)** and **Fully Homomorphic Encryption (FHE)**. The primary goal is to train a Convolutional Neural Network (CNN) collaboratively on the CIFAR-10 dataset across multiple clients, while ensuring that the model updates remain strictly confidential.

Currently, the project implements a baseline standard FL pipeline and a Full FHE pipeline. The ultimate objective of this research is to implement **Selective FHE**—encrypting only the most sensitive model parameters (e.g., 10%)—to overcome the massive computational and communication bottlenecks caused by Full FHE.

## Project Structure

### Phase 1: Standard Federated Learning (Plaintext)
* `data_utils.py`: Downloads and splits the CIFAR-10 dataset among 3 clients.
* `model.py`: Defines the PyTorch CNN architecture (based on LeNet) and the local train/test functions.
* `client.py` & `server.py`: Standard Flower client and server implementations using plaintext weights (FedAvg strategy).

### Phase 2: Full FHE Federated Learning (Encrypted)
* `crypto_utils.py`: Generates and saves the TenSEAL CKKS cryptographic keys (`secret_context.bytes`, `public_context.bytes`).
* `client_fhe.py`: Flattens the PyTorch model into 1D arrays, chunks them, and encrypts 100% of the weights using CKKS before sending them to the server.
* `server_fhe.py`: A custom Flower server implementing a SecureFedAvg strategy to perform homomorphic aggregation on encrypted bytes without ever decrypting them.

## Installation & Requirements

1. Clone this repository.
2. Create a virtual environment and install the required dependencies:
   ```bash
   pip install flwr torch torchvision tenseal numpy
   ```

## How to Run the Simulations

### 1. Standard FL Pipeline (Fast)
Open 4 separate terminals and run the following commands:
* **Terminal 1 (Server):** `python server.py`
* **Terminal 2 (Client 0):** `python client.py 0`
* **Terminal 3 (Client 1):** `python client.py 1`
* **Terminal 4 (Client 2):** `python client.py 2`

### 2. Full FHE Pipeline (Slow - Proof of Bottleneck)
*Note: This simulation takes significantly longer due to the encryption of 100% of the model's parameters.*

First, generate the cryptographic keys:
```bash
python crypto_utils.py
```
Then, open 4 terminals and run:
* **Terminal 1 (Server):** `python server_fhe.py`
* **Terminal 2 (Client 0):** `python client_fhe.py 0`
* **Terminal 3 (Client 1):** `python client_fhe.py 1`
* **Terminal 4 (Client 2):** `python client_fhe.py 2`

## Research Roadmap
- [x] **Phase 1:** Build a robust Standard FL pipeline.
- [x] **Phase 2:** Implement Full FHE using TenSEAL/CKKS (Demonstrating the latency bottleneck).
- [ ] **Phase 3:** Implement Selective FHE (Encrypting only a fraction of the weights to optimize performance while maintaining privacy).