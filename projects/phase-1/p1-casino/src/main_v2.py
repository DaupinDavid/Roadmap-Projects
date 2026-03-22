import random
import numpy as np
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

def simulation_distribution(montant_depart, mise, n=1000):
    """
    Simule n parties simultanées avec Numpy et affiche le faisceau de courbes.

    Args:
        montant_depart (float): Capital de départ en euros
        mise (float): Montant misé par partie en euros
        n (int): Nombre de simulations simultanées (défaut : 1000)

    Returns:
        None
    """
    nb_parties = 10

    cartes_joueur   = np.random.randint(1, 11, size=(n, nb_parties))
    cartes_croupier = np.random.randint(1, 11, size=(n, nb_parties))

    gains = np.where(
        cartes_joueur > cartes_croupier, mise,
        np.where(cartes_joueur == cartes_croupier, -mise / 2, -mise)
    )

    historique = np.hstack([
        np.full((n, 1), montant_depart),
        montant_depart + np.cumsum(gains, axis=1)
    ])

    fig = plt.figure(figsize=(12, 6))
    plt.plot(historique.T, alpha=0.1, color='blue', linewidth=0.5)
    plt.axhline(y=montant_depart, color='red', linestyle='--',
                linewidth=2, label='Capital de départ')
    plt.title(f"Distribution de {n} simulations — {nb_parties} parties chacune")
    plt.xlabel("Parties jouées")
    plt.ylabel("Capital (€)")
    plt.legend()
    
    capitaux_finaux = historique[:, -1]
    gain_moyen_total = np.mean(capitaux_finaux) - montant_depart
    volume_total_jeu = nb_parties * mise
    roi_moyen = (gain_moyen_total / volume_total_jeu) * 100

    print(f"\n📊 {'Distribution':<25} — {n} joueurs sur {nb_parties} parties")
    print(f"{'Gain moyen total':<25} : {core.formater_montant(gain_moyen_total)}")
    print(f"{'ROI moyen / joueur':<25} : {roi_moyen:.2f} %")
    return fig

def simulation_loi_grands_nombres(montant_depart, mise, n=1_000_000):
    """
    Simule n parties en un seul vecteur Numpy.
    Démontre que le Casino gagne toujours sur le long terme.

    Args:
        montant_depart (float): Capital de départ en euros
        mise (float): Montant misé par partie en euros
        n (int): Nombre de parties simulées (défaut : 1 000 000)

    Returns:
        None
    """
    cartes_joueur   = np.random.randint(1, 11, size=n)
    cartes_croupier = np.random.randint(1, 11, size=n)

    gains = np.where(
        cartes_joueur > cartes_croupier, mise,
        np.where(cartes_joueur == cartes_croupier, -mise / 2, -mise)
    )

    historique = montant_depart + np.cumsum(gains)

    gain_net = historique[-1] - montant_depart

    fig = plt.figure(figsize=(12, 6))
    plt.plot(historique, color='blue', linewidth=0.5)
    plt.axhline(y=montant_depart, color='red', linestyle='--',
                linewidth=2, label='Capital de départ')
    plt.title(f"Loi des Grands Nombres — {n:,} parties simulées")
    plt.xlabel("Parties jouées")
    plt.ylabel("Capital (€)")
    plt.legend()

    volume_total_mise = n * mise
    roi_reel = (gain_net / volume_total_mise) * 100

    print(f"\n📊 {'Loi des Grands Nombres':<25} — {n:,} parties")
    print(f"{'Gain moyen / partie':<25} : {gain_net/n:.2f} €")
    print(f"{'Capital final':<25} : {core.formater_montant(historique[-1])}")
    print(f"{'ROI Réel (sur volume)':<25} : {roi_reel:.2f} %")
    return fig

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

    # Chemins de sauvegarde
    chemin_actuel = os.path.dirname(os.path.abspath(__file__))
    chemin_docs = os.path.join(os.path.dirname(chemin_actuel), "docs")
    os.makedirs(chemin_docs, exist_ok=True)

    # Graphique 1 — Partie Joueur
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(historique, marker='o', color='blue')
    plt.axhline(y=montant_depart, color='red', linestyle='--', label='Mise initiale')
    plt.title("Évolution du capital - Simulation Casino")
    plt.xlabel("Parties jouées")
    plt.ylabel("Capital (€)")
    plt.legend()
    chemin_g1 = os.path.join(chemin_docs, "v2_1-joueur.png")
    fig1.savefig(chemin_g1)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"\n✅ Graphique 1 sauvegardé : {chemin_g1}")

    # Graphique 2 — Distribution 1 000 simulations
    print("\n⏳ Simulation de 1 000 distributions en cours...")
    fig2 = simulation_distribution(montant_depart, mise, n=1000)
    chemin_g2 = os.path.join(chemin_docs, "v2_2-distribution.png")
    fig2.savefig(chemin_g2)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"✅ Graphique 2 sauvegardé : {chemin_g2}")

    # Graphique 3 — Loi des Grands Nombres
    print("\n⏳ Simulation de 1 000 000 parties en cours...")
    fig3 = simulation_loi_grands_nombres(montant_depart, mise, n=1_000_000)
    chemin_g3 = os.path.join(chemin_docs, "v2_3-loi_grands_nombres.png")
    fig3.savefig(chemin_g3)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"✅ Graphique 3 sauvegardé : {chemin_g3}")

    plt.close('all')
    print("\n✅ Trois graphiques générés dans docs/")