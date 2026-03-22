import random
import roadmap_core.base_script as core
import matplotlib.pyplot as plt
import os

# Fonction simulant le tirage d'une carte (entre 1 et 10)
def tirer_carte():
    nombre = random.randint(1, 10)
    return nombre

def jouer_partie(montant, mise):
    """
    Joue une partie contre le croupier.

    Args:
        montant (float): Capital actuel du joueur en euros
        mise (float): Montant misé pour cette partie en euros

    Returns:
        montant (float): Nouveau capital après la partie
        msg (str): Message décrivant le résultat
        mon_score (int): Carte tirée par le joueur
        score_croupier (int): Carte tirée par le croupier
    """
    mon_score = tirer_carte()
    score_croupier = tirer_carte()
    if mon_score > score_croupier:
        montant_partie = round(mise, 2)
        msg = f"Félicitations! Tu as gagné {montant_partie:.2f}€!"
    elif mon_score == score_croupier:
        montant_partie = -round(mise / 2, 2)
        msg = f"Egalité. Ici, l'égalité te fais perdre {abs(montant_partie):.2f}€, la moitié de ta mise... "
    else:
        montant_partie = -round(mise, 2)
        msg = f"Dommage... Tu as perdu {abs(montant_partie):.2f}€."
    montant += montant_partie
    montant = round(montant, 2)
    return montant, msg, mon_score, score_croupier

if __name__ == "__main__":

    # Configuration initiale du jeu
    nom = input("Bienvenue au Casino! Quel est ton nom? ")
    while True:
        try:
            montant = float(input(f"\nSalut {nom}! Combien souhaites-tu apporter au Casino aujourd'hui? (€) "))
            if montant > 0:
                break
            else:
                print("Veuillez entrer un montant au moins égal à 1.")
        except ValueError:
            print("Veuillez entrer un nombre valide.")

    print(f"\nFélicitations {nom}! Tu démarres avec {montant:.2f}€.")
    montant_depart = montant
    mise = round(montant / 10, 2)
    print(f"Chaque partie coûtera {mise:.2f}€.\n")
    parties_jouees = 0
    historique = [montant_depart]

    # Simulation des 10 parties
    print(f"Tu rentres avec {montant:.2f}€. Il y aura 10 parties et chaque partie coûte {mise:.2f}€. Bonne chance!\n")
    while parties_jouees < 10:
        print(f"--- Partie Numéro {parties_jouees + 1} ---")
        montant, msg, mon_score, score_croupier = jouer_partie(montant, mise)
        historique.append(montant)
        print(f" Carte Croupier n° {score_croupier}")
        print(f" Carte {nom} n° {mon_score}")
        print(msg)
        parties_jouees += 1
        print(f"Tu dispose de {montant:.2f}€\n")

        # Vérification si le joueur est ruiné
        if montant <= 0:
            print("Tu es ruiné... Le Casino te remercie et te souhaites une agréable journée. A bientôt!\n")
            break

    gain_net = montant - montant_depart
    roi = core.calcul_roi(gain_net, montant_depart)
    if montant > 0:
        print(f"\nFin du jeu! Tu repars avec {montant:.2f}€. Ton gain net est de {gain_net:.2f}€.")
        print(f"Ton retour sur investissement est de {roi:.2f}% par rapport à ta mise initiale.\n")

    print(f"Au revoir {nom} !")

    # Génération du graphique
    chemin_actuel = os.path.dirname(os.path.abspath(__file__))
    chemin_docs = os.path.join(os.path.dirname(chemin_actuel), "docs")
    os.makedirs(chemin_docs, exist_ok=True)
    chemin_image = os.path.join(chemin_docs, "v1.png")
    plt.figure(figsize=(10, 5))
    plt.plot(historique, marker='o', color='blue')
    plt.axhline(y=montant_depart, color='red', linestyle='--', label='Mise initiale')
    plt.title("Évolution du capital - Simulation Casino")
    plt.xlabel("Parties jouées")
    plt.ylabel("Capital (€)")
    plt.legend()
    plt.savefig(chemin_image)
    print(f"\n✅ Graphique sauvegardé avec succès dans : {chemin_image}")
    #plt.show()