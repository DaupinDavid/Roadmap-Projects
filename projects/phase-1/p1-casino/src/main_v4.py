import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc
import os
import roadmap_core.base_script as core

# =================================================================
# WORKER MULTIPROCESSING 
# =================================================================

def worker_cellule_heatmap(montant_depart, mise_test, seuil, nb, n):
    """
    Worker autosuffisant pour une cellule de la heatmap.
    Logique inline — aucune dépendance à une fonction du module parent.
    Compatible Windows (spawn) et Linux/Mac (fork).

    Args:
        montant_depart (float): Capital de départ en euros
        mise_test (float): Mise testée pour cette cellule
        seuil (float): Seuil de ruine en euros
        nb (int): Nombre de parties pour cette cellule
        n (int): Nombre de joueurs simulés

    Returns:
        float: Taux de ruine de la cellule
    """
    import numpy as np  # Import local — garanti disponible dans le sous-process
    c1 = np.random.randint(1, 11, size=(n, nb))
    c2 = np.random.randint(1, 11, size=(n, nb))
    gains = np.where(c1 > c2, mise_test,
                     np.where(c1 == c2, -mise_test / 2, -mise_test))
    historique = np.hstack([
        np.full((n, 1), montant_depart),
        montant_depart + np.cumsum(gains, axis=1)
    ])
    alerte_seuil = historique <= seuil
    etat_final_ruine = np.maximum.accumulate(alerte_seuil, axis=1)
    historique = np.where(etat_final_ruine, seuil, historique)
    return float(np.mean(etat_final_ruine[:, -1]) * 100)


NB_PARTIES   = 10
N_SIMULATION = 1_000_000   # Graphique 2 — distribution ruine
N_COURBE     = 100_000     # Graphique 3 — courbe risque vs mise
N_HEATMAP    =  50_000     # Graphique 4 — heatmap (30 cellules)

# =================================================================
# FONCTIONS DE SIMULATION
# =================================================================

def simulation_risque_ruine(montant_depart, mise, seuil=0.0,
                             nb_parties=NB_PARTIES, n=N_SIMULATION):
    """
    Simule n joueurs sur nb_parties et calcule le taux de ruine.

    Args:
        montant_depart (float): Capital de départ en euros
        mise (float): Montant misé par partie en euros
        seuil (float): Seuil de ruine en euros (0 = ruine totale)
        nb_parties (int): Nombre de parties par joueur
        n (int): Nombre de joueurs simulés

    Returns:
        taux_ruine (float): % de joueurs ayant touché le seuil
        historique_moyen (np.ndarray): Capital moyen à chaque partie
        n_ruines (int): Nombre absolu de joueurs ruinés
        ruine_liste_morts (np.ndarray): Masque booléen des joueurs ruinés
        historique (np.ndarray): Capitaux complets (n, nb_parties+1)
    """
    c1 = np.random.randint(1, 11, size=(n, nb_parties))
    c2 = np.random.randint(1, 11, size=(n, nb_parties))
    gains = np.where(c1 > c2, mise, np.where(c1 == c2, -mise / 2, -mise))

    historique = np.hstack([
        np.full((n, 1), montant_depart),
        montant_depart + np.cumsum(gains, axis=1)
    ])

    alerte_seuil = historique <= seuil
    etat_final_ruine = np.maximum.accumulate(alerte_seuil, axis=1)
    historique = np.where(etat_final_ruine, seuil, historique)

    # Ruine = capital <= seuil à n'importe quel moment de la session
    ruine_liste_morts = etat_final_ruine[:, -1]
    n_ruines = int(np.sum(ruine_liste_morts))
    taux_ruine = n_ruines / n * 100
    historique_moyen = np.mean(historique, axis=0)

    return taux_ruine, historique_moyen, n_ruines, ruine_liste_morts, historique


def courbe_risque_vs_mise(montant_depart, seuil=0.0,
                           nb_parties=NB_PARTIES, n=N_COURBE):
    """
    Calcule le taux de ruine pour des mises allant de 5% à 50% du capital.

    Args:
        montant_depart (float): Capital de départ en euros
        seuil (float): Seuil de ruine en euros
        nb_parties (int): Nombre de parties
        n (int): Joueurs simulés par point de la courbe

    Returns:
        pourcentages (np.ndarray): % de mise testés (5 à 50)
        taux_ruines (np.ndarray): Taux de ruine correspondants
    """
    pourcentages = np.arange(5, 55, 5)
    taux_ruines  = []

    for pct in pourcentages:
        mise_test = round(montant_depart * pct / 100, 2)
        taux, _, _, _, _ = simulation_risque_ruine(
            montant_depart, mise_test, seuil, nb_parties, n
        )
        taux_ruines.append(taux)

    return pourcentages, np.array(taux_ruines)


def heatmap_risque(montant_depart, seuil=0.0, n=N_HEATMAP):
    """
    Calcule le risque de ruine sur une grille mise × nb_parties.
    Utilise Pool.starmap pour paralléliser les 30 cellules sur tous les cœurs.

    Args:
        montant_depart (float): Capital de départ en euros
        seuil (float): Seuil de ruine en euros
        n (int): Joueurs simulés par cellule

    Returns:
        matrice (np.ndarray): Taux de ruine (% mise × nb_parties)
        pct_mises (list): Pourcentages de mise testés
        nb_parties_list (list): Nombres de parties testés
    """
    pct_mises       = [5, 10, 15, 20, 25, 30]
    nb_parties_list = [5, 10, 20, 50, 100]

    # Construction des 30 missions (une par cellule)
    missions = [
        (montant_depart, round(montant_depart * pct / 100, 2), seuil, nb, n)
        for pct in pct_mises
        for nb in nb_parties_list
    ]

    n_coeurs = os.cpu_count() or 1
    with mp.Pool(processes=n_coeurs) as pool:
        resultats = pool.starmap(worker_cellule_heatmap, missions)

    matrice = np.array(resultats).reshape(len(pct_mises), len(nb_parties_list))
    return matrice, pct_mises, nb_parties_list

# =================================================================
# FONCTIONS GRAPHIQUES
# =================================================================

def graphique_joueur(historique_joueur, montant_depart, seuil, seuil_pct):
    """
    Évolution du capital du joueur sur ses parties.

    Args:
        historique_joueur (list): Capital à chaque partie
        montant_depart (float): Capital de départ en euros
        seuil (float): Seuil de ruine en euros
        seuil_pct (float): Pourcentage de seuil — affiché dans le titre

    Returns:
        fig: Figure matplotlib
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(historique_joueur, marker='o', color='blue')
    plt.axhline(y=montant_depart, color='red', linestyle='--', label='Capital départ')
    if seuil > 0:
        plt.axhline(y=seuil, color='orange', linestyle=':',
                    label=f'Seuil ruine ({seuil:.2f}€)')
    plt.title(f"Évolution du capital (Seuil {int(seuil_pct)}%) — Partie Joueur")
    plt.xlabel("Parties jouées")
    plt.ylabel("Capital (€)")
    plt.legend()
    return fig


def graphique_distribution_ruine(historique, ruine_liste_morts, montant_depart, seuil, taux_ruine, seuil_pct):
    """
    Histogramme des capitaux finaux — survivants (vert) vs ruinés (rouge).

    Args:
        historique (np.ndarray): Capitaux complets (n, nb_parties+1)
        ruine_liste_morts (np.ndarray): Masque booléen des joueurs ruinés
        montant_depart (float): Capital de départ
        seuil (float): Seuil de ruine en euros
        taux_ruine (float): Taux de ruine calculé
        seuil_pct (float): Pourcentage de seuil — affiché dans le titre

    Returns:
        fig: Figure matplotlib
    """
    capitaux_finaux = historique[:, -1]
    survivants      = capitaux_finaux[~ruine_liste_morts]
    ruines          = capitaux_finaux[ruine_liste_morts]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(survivants, bins=60, color='#2ECC71', alpha=0.75,
            label=f"Survivants ({len(survivants):,})")
    ax.hist(ruines,     bins=60, color='#E74C3C', alpha=0.75,
            label=f"Ruinés ({len(ruines):,})")
    ax.axvline(x=montant_depart, color='blue', linestyle='--',
               linewidth=2, label=f"Capital départ ({montant_depart:.2f}€)")
    if seuil > 0:
        ax.axvline(x=seuil, color='orange', linestyle=':',
                   linewidth=2, label=f"Seuil de ruine ({seuil:.2f}€)")
    ax.set_title(f"Distribution (Seuil {int(seuil_pct)}%) — Ruine réelle : {taux_ruine:.2f}%")
    ax.set_xlabel("Capital final (€)")
    ax.set_ylabel("Nombre de joueurs")
    ax.legend()
    return fig


def graphique_courbe_risque(pourcentages, taux_ruines, mise_joueur, montant_depart, seuil_pct):
    """
    Courbe risque de ruine en fonction du % de mise.
    Marque la mise du joueur sur la courbe.

    Args:
        pourcentages (np.ndarray): % de mise testés
        taux_ruines (np.ndarray): Taux de ruine correspondants
        mise_joueur (float): Mise utilisée par le joueur
        montant_depart (float): Capital de départ
        seuil_pct (float): Pourcentage de seuil — affiché dans le titre

    Returns:
        fig: Figure matplotlib
    """
    mise_pct = (mise_joueur / montant_depart) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pourcentages, taux_ruines, 'o-', color='#E74C3C',
            linewidth=2, markersize=7)
    ax.fill_between(pourcentages, taux_ruines, alpha=0.15, color='#E74C3C')
    ax.axvline(x=mise_pct, color='blue', linestyle='--', linewidth=2,
               label=f"Ta mise : {mise_pct:.0f}% ({mise_joueur:.2f}€)")
    ax.set_title(f"Risque vs Mise (Seuil {int(seuil_pct)}%) — Évolution du danger")
    ax.set_xlabel("Mise (% du capital de départ)")
    ax.set_ylabel("Taux de ruine (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def graphique_heatmap(matrice, pct_mises, nb_parties_list, seuil_pct):
    """
    Heatmap risque de ruine : axe X = nb_parties, axe Y = % mise.
    Gradient vert (sûr) → rouge (dangereux).

    Args:
        matrice (np.ndarray): Taux de ruine en %
        pct_mises (list): % de mise (axe Y)
        nb_parties_list (list): Nombres de parties (axe X)
        seuil_pct (float): Pourcentage de seuil — affiché dans le titre

    Returns:
        fig: Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrice, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Taux de ruine (%)')

    ax.set_xticks(range(len(nb_parties_list)))
    ax.set_xticklabels([str(n) for n in nb_parties_list])
    ax.set_yticks(range(len(pct_mises)))
    ax.set_yticklabels([f"{p}%" for p in pct_mises])
    ax.set_xlabel("Nombre de parties")
    ax.set_ylabel("Mise (% du capital)")
    ax.set_title(f"Heatmap (Seuil {int(seuil_pct)}%) — Risque par mise et durée")

    for i in range(len(pct_mises)):
        for j in range(len(nb_parties_list)):
            ax.text(j, i, f"{matrice[i, j]:.1f}%",
                    ha='center', va='center', fontsize=9,
                    color='white' if matrice[i, j] > 50 else 'black')
    return fig

# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":

    # --- Configuration initiale ---
    nom = input("Bienvenue au Casino! Quel est ton nom? ")
    while True:
        try:
            montant = float(input(
                f"\nSalut {nom}! Combien souhaites-tu apporter au Casino aujourd'hui? (€) "))
            if montant > 0:
                break
            else:
                print("Veuillez entrer un montant au moins égal à 1.")
        except ValueError:
            print("Veuillez entrer un nombre valide.")

    # --- Seuil de ruine configurable ---
    print("\n--- Configuration du Seuil de Ruine ---")
    print("0 = Ruine totale. 20 = Tu t'arrêtes si tu perds 20% de ton capital")
    while True:
        try:
            seuil_pct = float(input("Seuil de ruine en % de perte (0 à 99) : "))
            if 0 <= seuil_pct < 100:
                break
            else:
                print("Valeur entre 0 et 99.")
        except ValueError:
            print("Veuillez entrer un nombre valide.")

    seuil = round(montant * (1 - seuil_pct / 100), 2) if seuil_pct > 0 else 0.0
    if seuil_pct > 0:
        print(f"Seuil fixé à {seuil:.2f}€ (arrêt si perte > {seuil_pct:.0f}%)")
    else:
        print("Seuil : ruine totale (0€)")

    montant_depart = montant
    mise = round(montant / 10, 2)
    print(f"\nFélicitations {nom}! Tu démarres avec {montant:.2f}€.")
    print(f"Chaque partie coûtera {mise:.2f}€.\n")

    parties_jouees    = 0
    historique_joueur = [montant_depart]

    # --- 10 parties interactives ---
    print(f"Tu rentres avec {montant:.2f}€. "
          f"Il y aura {NB_PARTIES} parties. Bonne chance!\n")
    while parties_jouees < NB_PARTIES:
        print(f"--- Partie Numéro {parties_jouees + 1} ---")
        montant, msg, mon_score, score_croupier = core.jouer_partie(montant, mise)
        historique_joueur.append(montant)
        print(f" Carte Croupier n° {score_croupier}")
        print(f" Carte {nom} n° {mon_score}")
        print(msg)
        parties_jouees += 1
        print(f"Tu disposes de {montant:.2f}€\n")

        if montant <= seuil:
            print(f"Seuil de ruine atteint ({seuil:.2f}€). Le Casino te remercie!\n")
            break

    gain_net = montant - montant_depart
    roi = core.calcul_roi(gain_net, montant_depart)
    if montant > seuil:
        print(f"\nFin du jeu! Tu repars avec {montant:.2f}€.")
        print(f"Gain net : {gain_net:.2f}€ | ROI : {roi:.2f}%\n")
    print(f"Au revoir {nom} !")

    # --- Chemins de sauvegarde ---
    chemin_actuel = os.path.dirname(os.path.abspath(__file__))
    chemin_docs = os.path.join(os.path.dirname(chemin_actuel), "docs")
    os.makedirs(chemin_docs, exist_ok=True)
    nom_dossier  = f"v4_ruine_{int(seuil_pct)}prct"
    chemin_final = os.path.join(chemin_docs, nom_dossier)
    os.makedirs(chemin_final, exist_ok=True) 
    print(f"\n📂 Résultats rangés dans : {nom_dossier}")

    # --- Graphique 1 — Partie joueur ---
    fig1 = graphique_joueur(historique_joueur, montant_depart, seuil, seuil_pct)
    chemin_g1 = os.path.join(chemin_final, f"v4_1_joueur_{int(seuil_pct)}prct.png")
    fig1.savefig(chemin_g1)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"\n✅ Graphique 1 sauvegardé : {chemin_g1}")

    # --- Graphique 2 — Distribution ruine (1 000 000 joueurs) ---
    print(f"\n⏳ Simulation de {N_SIMULATION:,} joueurs en cours...")
    taux, hist_moy, n_ruines, ruine_liste_morts, historique_full = simulation_risque_ruine(
        montant_depart, mise, seuil, NB_PARTIES, N_SIMULATION
    )
    print(f"📊 Taux de ruine : {taux:.2f}% ({n_ruines:,} joueurs sur {N_SIMULATION:,})")

    fig2 = graphique_distribution_ruine(
        historique_full, ruine_liste_morts, montant_depart, seuil, taux, seuil_pct
    )
    chemin_g2 = os.path.join(chemin_final, f"v4_2_distrib_{int(seuil_pct)}prct.png")
    fig2.savefig(chemin_g2)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"✅ Graphique 2 sauvegardé : {chemin_g2}")

    # Libération RAM — le grand tableau n'est plus nécessaire
    del historique_full, ruine_liste_morts
    gc.collect()  # Force le Garbage Collector

    # --- Graphique 3 — Courbe risque vs mise ---
    print(f"\n⏳ Courbe risque vs mise ({N_COURBE:,} joueurs/point × 10 points)...")
    pourcentages, taux_ruines = courbe_risque_vs_mise(
        montant_depart, seuil, NB_PARTIES, N_COURBE
    )
    fig3 = graphique_courbe_risque(pourcentages, taux_ruines, mise, montant_depart, seuil_pct)
    chemin_g3 = os.path.join(chemin_final, f"v4_3_courbe_{int(seuil_pct)}prct.png")
    fig3.savefig(chemin_g3)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"✅ Graphique 3 sauvegardé : {chemin_g3}")

    # --- Graphique 4 — Heatmap ---
    print(f"\n⏳ Heatmap ({N_HEATMAP:,} joueurs/cellule × 30 cellules)...")
    matrice, pct_mises, nb_parties_list = heatmap_risque(montant_depart, seuil, N_HEATMAP)
    fig4 = graphique_heatmap(matrice, pct_mises, nb_parties_list, seuil_pct)
    chemin_g4 = os.path.join(chemin_final, f"v4_4_heatmap_{int(seuil_pct)}prct.png")
    fig4.savefig(chemin_g4)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"✅ Graphique 4 sauvegardé : {chemin_g4}")

    plt.close('all')
    print(f"\n{'='*55}")
    print(f"✅ V4 — Risque de Ruine terminée. 4 graphiques générés.")
    print(f"📊 Taux de ruine (ta session) : {taux:.2f}%")
    print(f"📊 Seuil utilisé : {seuil:.2f}€ ({seuil_pct:.0f}% de perte)")
    print(f"{'='*55}")