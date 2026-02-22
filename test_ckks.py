import torch
import tenseal as ts

print("--- 1. CRÉATION DU CONTEXTE CKKS (Génération des clés) ---")
# On définit les paramètres mathématiques du chiffrement
# poly_modulus_degree = 8192 est un standard pour un bon compromis sécurité/vitesse
context = ts.context(
    ts.SCHEME_TYPE.CKKS, 
    poly_modulus_degree=8192, 
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
# Le paramètre d'échelle (scale) gère la précision des nombres à virgule
context.global_scale = 2**40
# On génère les clés (Secrète pour déchiffrer, Publique pour chiffrer et calculer)
context.generate_galois_keys()

print("\n--- 2. CHIFFREMENT DES DONNÉES ---")
# Imaginons que ce sont les poids envoyés par le Client 1 et le Client 2
poids_client_1 = torch.tensor([0.5, -1.2, 3.14])
poids_client_2 = torch.tensor([1.5, 2.2, -1.14])

print(f"Poids en clair Client 1 : {poids_client_1.tolist()}")
print(f"Poids en clair Client 2 : {poids_client_2.tolist()}")

# On chiffre ces poids avec la clé publique
enc_client_1 = ts.ckks_vector(context, poids_client_1.tolist())
enc_client_2 = ts.ckks_vector(context, poids_client_2.tolist())

print("Les données sont maintenant chiffrées ! Le serveur ne peut pas les lire.")

print("\n--- 3. CALCUL HOMOMORPHE SUR LE SERVEUR ---")
# Le serveur additionne les poids SANS les déchiffrer (C'est la magie du CKKS !)
enc_resultat = enc_client_1 + enc_client_2

print("\n--- 4. DÉCHIFFREMENT CHEZ LE CLIENT ---")
# Le client reçoit le résultat et utilise sa clé secrète pour le lire
resultat_clair = enc_resultat.decrypt()

# On arrondit un peu car le CKKS est un chiffrement "approximatif"
resultat_arrondi = [round(v, 4) for v in resultat_clair]

print(f"Résultat déchiffré : {resultat_arrondi}")
print("On s'attendait à : [2.0, 1.0, 2.0]")