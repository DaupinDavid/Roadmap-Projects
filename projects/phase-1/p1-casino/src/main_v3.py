import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os
import multiprocessing as mp
import roadmap_core.base_script as core

# =================================================================
# FONCTIONS OUVRIÈRES (WORKERS) — EN HAUT DU FICHIER OBLIGATOIRE
# =================================================================

def worker_simulation(nb_parties, mise):
    """
    Fonction exécutée par chaque cœur CPU.
    Retourne le gain total du bloc — RAM minimale.

    Args:
        nb_parties (int): Nombre de parties à simuler sur ce cœur
        mise (float): Montant misé par partie en euros

    Returns:
        float: Gain total du bloc
    """
    c1 = np.random.randint(1, 11, size=nb_parties)
    c2 = np.random.randint(1, 11, size=nb_parties)
    gains = np.where(c1 > c2, mise, np.where(c1 == c2, -mise / 2, -mise))
    return np.sum(gains)

def worker_historique(nb_parties, mise):
    """
    Fonction exécutée par chaque cœur CPU.
    Retourne les gains individuels pour reconstruire la courbe.

    Args:
        nb_parties (int): Nombre de parties à simuler sur ce cœur
        mise (float): Montant misé par partie en euros

    Returns:
        np.ndarray: Gains de chaque partie du bloc
    """
    c1 = np.random.randint(1, 11, size=nb_parties)
    c2 = np.random.randint(1, 11, size=nb_parties)
    gains = np.where(c1 > c2, mise, np.where(c1 == c2, -mise / 2, -mise))
    return gains

# =================================================================
# FONCTIONS DE SIMULATION
# =================================================================

def simulation_distribution(montant_depart, mise, n=1000):
    """
    Simule n parties simultanées avec Numpy — faisceau de courbes.

    Args:
        montant_depart (float): Capital de départ en euros
        mise (float): Montant misé par partie en euros
        n (int): Nombre de simulations simultanées (défaut : 1000)

    Returns:
        fig: Figure matplotlib
    """
    nb_parties = 10
    c1 = np.random.randint(1, 11, size=(n, nb_parties))
    c2 = np.random.randint(1, 11, size=(n, nb_parties))
    gains = np.where(c1 > c2, mise, np.where(c1 == c2, -mise / 2, -mise))
    historique = np.hstack([
        np.full((n, 1), montant_depart),
        montant_depart + np.cumsum(gains, axis=1)
    ])

    capitaux_finaux = historique[:, -1]
    gain_moyen = np.mean(capitaux_finaux) - montant_depart
    volume_jeu = nb_parties * mise
    roi_moyen = (gain_moyen / volume_jeu) * 100

    fig = plt.figure(figsize=(12, 6))
    plt.plot(historique.T, alpha=0.1, color='blue', linewidth=0.5)
    plt.axhline(y=montant_depart, color='red', linestyle='--',
                linewidth=2, label='Capital de départ')
    plt.title(f"Distribution — {n} joueurs sur {nb_parties} parties")
    plt.xlabel("Parties jouées")
    plt.ylabel("Capital (€)")
    plt.legend()

    print(f"\n📊 {'Distribution':<25} — {n} joueurs sur {nb_parties} parties")
    print(f"{'Gain moyen total':<25} : {core.formater_montant(gain_moyen)}")
    print(f"{'ROI moyen / joueur':<25} : {roi_moyen:.2f} %")
    return fig

def run_numpy(montant_depart, mise, n):
    """
    Simule n parties sur 1 seul cœur — référence benchmark.

    Args:
        montant_depart (float): Capital de départ en euros
        mise (float): Montant misé par partie en euros
        n (int): Nombre de parties à simuler

    Returns:
        tuple: (historique, gain_net, roi_reel, temps_execution)
    """
    debut = time.time()
    gain_total = worker_simulation(n, mise)
    temps = time.time() - debut

    gain_net = gain_total
    volume = n * mise
    roi_reel = (gain_net / volume) * 100

    return gain_net, roi_reel, temps

def run_multiprocessing(montant_depart, mise, n):
    """
    Simule n parties sur tous les cœurs disponibles.
    Utilise starmap pour une syntaxe propre.

    Args:
        montant_depart (float): Capital de départ en euros
        mise (float): Montant misé par partie en euros
        n (int): Nombre de parties à simuler

    Returns:
        tuple: (historique, gain_net, roi_reel, temps_execution, n_coeurs)
    """
    n_coeurs = os.cpu_count() or 1
    parties_par_coeur = n // n_coeurs
    missions = [(parties_par_coeur, mise) for _ in range(n_coeurs)]

    debut = time.time()
    with mp.Pool(processes=n_coeurs) as pool:
        resultats = pool.starmap(worker_historique, missions)
    temps = time.time() - debut

    gains_totaux = np.concatenate(resultats)
    historique = montant_depart + np.cumsum(gains_totaux)
    gain_net = historique[-1] - montant_depart
    volume = n * mise
    roi_reel = (gain_net / volume) * 100

    return historique, gain_net, roi_reel, temps, n_coeurs

def graphique_benchmark(temps_numpy, temps_multi, n_coeurs, n_parties):
    """
    Génère le graphique de comparaison vitesse Numpy vs Multiprocessing.

    Args:
        temps_numpy (float): Temps Numpy en secondes
        temps_multi (float): Temps Multiprocessing en secondes
        n_coeurs (int): Nombre de cœurs utilisés
        n_parties (int): Nombre de parties simulées

    Returns:
        fig: Figure matplotlib
    """
    acceleration = round(temps_numpy / temps_multi, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Barres vitesse
    labels = [f"Numpy\n(1 cœur)", f"Multiprocessing\n({n_coeurs} cœurs)"]
    temps  = [temps_numpy, temps_multi]
    colors = ['#E74C3C', '#2ECC71']

    axes[0].bar(labels, temps, color=colors, width=0.4)
    axes[0].set_title("Comparaison vitesse d'exécution")
    axes[0].set_ylabel("Temps (secondes)")
    for i, t in enumerate(temps):
        axes[0].text(i, t + 0.02, f"{t:.4f}s",
                     ha='center', fontweight='bold', fontsize=11)

    # Accélération
    axes[1].bar(["Accélération"], [acceleration], color='#3498DB', width=0.3)
    axes[1].set_title("Facteur d'accélération")
    axes[1].set_ylabel("x fois plus rapide")
    axes[1].text(0, acceleration + 0.05, f"{acceleration}x",
                 ha='center', fontsize=16, fontweight='bold', color='#2C3E50')

    rouge = mpatches.Patch(color='#E74C3C',
                           label=f"Numpy : {temps_numpy:.4f}s")
    vert  = mpatches.Patch(color='#2ECC71',
                           label=f"Multiprocessing : {temps_multi:.4f}s")
    bleu  = mpatches.Patch(color='#3498DB',
                           label=f"Accélération : {acceleration}x")
    fig.legend(handles=[rouge, vert, bleu],
               loc='lower center', ncol=3, fontsize=10)

    plt.suptitle(
        f"Benchmark — Numpy vs Multiprocessing ({n_parties:,} parties)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig

# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":

    # Configuration initiale du jeu
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

    print(f"\nFélicitations {nom}! Tu démarres avec {montant:.2f}€.")
    montant_depart = montant
    mise = round(montant / 10, 2)
    print(f"Chaque partie coûtera {mise:.2f}€.\n")
    parties_jouees = 0
    historique_joueur = [montant_depart]

    # Simulation des 10 parties interactives
    print(f"Tu rentres avec {montant:.2f}€. "
          f"Il y aura 10 parties et chaque partie coûte {mise:.2f}€. Bonne chance!\n")
    while parties_jouees < 10:
        print(f"--- Partie Numéro {parties_jouees + 1} ---")
        montant, msg, mon_score, score_croupier = core.jouer_partie(montant, mise)
        historique_joueur.append(montant)
        print(f" Carte Croupier n° {score_croupier}")
        print(f" Carte {nom} n° {mon_score}")
        print(msg)
        parties_jouees += 1
        print(f"Tu dispose de {montant:.2f}€\n")

        if montant <= 0:
            print("Tu es ruiné... Le Casino te remercie. A bientôt!\n")
            break

    gain_net = montant - montant_depart
    roi = core.calcul_roi(gain_net, montant_depart)
    if montant > 0:
        print(f"\nFin du jeu! Tu repars avec {montant:.2f}€. "
              f"Ton gain net est de {gain_net:.2f}€.")
        print(f"Ton retour sur investissement est de {roi:.2f}%"
              f" par rapport à ta mise initiale.\n")

    print(f"Au revoir {nom} !")

    # Chemins de sauvegarde
    chemin_actuel  = os.path.dirname(os.path.abspath(__file__))
    chemin_docs = os.path.join(os.path.dirname(chemin_actuel), "docs")
    os.makedirs(chemin_docs, exist_ok=True)

    # Graphique 1 — Partie joueur
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(historique_joueur, marker='o', color='blue')
    plt.axhline(y=montant_depart, color='red', linestyle='--', label='Mise initiale')
    plt.title("Évolution du capital - Simulation Casino")
    plt.xlabel("Parties jouées")
    plt.ylabel("Capital (€)")
    plt.legend()
    chemin_g1 = os.path.join(chemin_docs, "v3_1_joueur.png")
    fig1.savefig(chemin_g1)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"\n✅ Graphique 1 sauvegardé : {chemin_g1}")

    # Graphique 2 — Distribution
    print("\n⏳ Simulation de 1 000 distributions en cours...")
    fig2 = simulation_distribution(montant_depart, mise, n=1000)
    chemin_g2 = os.path.join(chemin_docs, "v3_2_distribution.png")
    fig2.savefig(chemin_g2)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"✅ Graphique 2 sauvegardé : {chemin_g2}")

    # --- Graphique 3 — Numpy ---
    N_BENCHMARK = 500_000_000
    print(f"\n⏳ Simulation Numpy {N_BENCHMARK:,} parties (1 cœur)...")

    gn_numpy, roi_numpy, temps_numpy = run_numpy(montant_depart, mise, N_BENCHMARK)
    gains_ref = worker_historique(N_BENCHMARK, mise)
    hist_ref = montant_depart + np.cumsum(gains_ref)
    pas_ref = max(1, len(hist_ref) // 5000)
    fig3 = plt.figure(figsize=(12, 6))

    plt.plot(hist_ref[::pas_ref], color='blue', linewidth=0.5)
    plt.axhline(y=montant_depart, color='red', linestyle='--', label='Capital de départ')
    plt.title(f"Numpy Référence — {N_BENCHMARK:,} parties (1 cœur)")
    plt.xlabel(f"Parties jouées (Affichage 1/{pas_ref})")
    plt.ylabel("Capital (€)")
    plt.legend()

    chemin_g3 = os.path.join(chemin_docs, "v3_3_numpy.png")
    fig3.savefig(chemin_g3)
    plt.show(block=False)
    plt.pause(0.1)

    print(f"\n📊 {'Numpy (1 cœur)':<25} — {N_BENCHMARK:,} parties")
    print(f"{'Gain net':<25} : {core.formater_montant(gn_numpy)}")
    print(f"{'ROI réel':<25} : {roi_numpy:.4f} %")
    print(f"{'Temps':<25} : {temps_numpy:.4f}s")
    print(f"✅ Graphique 3 sauvegardé : {chemin_g3}")

    # Graphique 4 — Multiprocessing
    print(f"\n⏳ Simulation Multiprocessing {N_BENCHMARK:,} parties "
          f"({os.cpu_count()} cœurs)...")
    hist_multi, gn_multi, roi_multi, temps_multi, n_coeurs = \
        run_multiprocessing(montant_depart, mise, N_BENCHMARK)

    # Sous-échantillonnage — 1 point sur 1000 pour éviter le crash matplotlib
    pas = max(1, len(hist_multi) // 5000)
    fig4 = plt.figure(figsize=(12, 6))
    plt.plot(hist_multi[::pas], color='green', linewidth=0.5)
    plt.axhline(y=montant_depart, color='red', linestyle='--',
                linewidth=2, label='Capital de départ')
    plt.title(f"Multiprocessing — {N_BENCHMARK:,} parties ({n_coeurs} cœurs)")
    plt.xlabel(f"Parties jouées (1 point / {pas})")
    plt.ylabel("Capital (€)")
    plt.legend()
    chemin_g4 = os.path.join(chemin_docs, "v3_4_multiprocessing.png")
    fig4.savefig(chemin_g4)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"\n📊 {'Multiprocessing':<25} — {N_BENCHMARK:,} parties")
    print(f"{'Gain net':<25} : {core.formater_montant(gn_multi)}")
    print(f"{'ROI réel':<25} : {roi_multi:.4f} %")
    print(f"{'Temps':<25} : {temps_multi:.4f}s")
    print(f"✅ Graphique 4 sauvegardé : {chemin_g4}")

    # Graphique 5 — Benchmark
    print("\n⏳ Génération du benchmark...")
    fig5 = graphique_benchmark(temps_numpy, temps_multi, n_coeurs, N_BENCHMARK)
    acceleration = round(temps_numpy / temps_multi, 2)
    chemin_g5 = os.path.join(chemin_docs, "v3_5_benchmark.png")
    fig5.savefig(chemin_g5)
    plt.show(block=False)
    plt.pause(0.1)
    print(f"✅ Graphique 5 sauvegardé : {chemin_g5}")

    plt.close('all')
    print(f"\n{'='*55}")
    print(f"✅ Mission Niveau 80 terminée. 5 graphiques générés.")
    print(f"⏱  Numpy        : {temps_numpy:.4f}s")
    print(f"⏱  Multiprocess : {temps_multi:.4f}s")
    print(f"🚀 Accélération : {acceleration}x")
    print(f"{'='*55}")