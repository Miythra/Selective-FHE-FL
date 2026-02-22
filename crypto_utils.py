import tenseal as ts

def generate_and_save_keys():
    print("⏳ Génération des clés mathématiques CKKS... (Ça peut prendre quelques secondes)")
    
    # 1. Création du contexte mathématique de base
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 
        poly_modulus_degree=8192, 
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # 2. Sauvegarde du contexte COMPLET (Il contient la clé secrète pour les clients)
    # L'argument 'save_secret_key=True' fait toute la magie ici !
    with open("secret_context.bytes", "wb") as f:
        f.write(context.serialize(save_secret_key=True))

    # 3. On supprime la clé secrète de la mémoire pour le rendre 100% public
    context.make_context_public()
    
    # 4. On sauvegarde ce contexte public (Pour le serveur)
    with open("public_context.bytes", "wb") as f:
        f.write(context.serialize())

    print("✅ Clés générées avec succès !")
    print(" -> 'secret_context.bytes' créé (Contient la clé secrète).")
    print(" -> 'public_context.bytes' créé (100% public).")

if __name__ == "__main__":
    generate_and_save_keys()