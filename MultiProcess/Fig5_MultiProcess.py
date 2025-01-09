import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from tqdm import tqdm

from egttools.games import AbstractNPlayerGame
from egttools import sample_simplex, calculate_nb_states
from egttools.numerical import PairwiseComparisonNumerical
from egttools.analytical import PairwiseComparison
from egttools.utils import calculate_stationary_distribution
from egttools.plotting import draw_invasion_diagram

from multiprocessing import Pool, cpu_count 
from multiprocessing import freeze_support

from pgg_game import PGGWithCommitment


# # Parameters of the game:
strategy_labels = ["F=1","F=2","F=3","F=4","F=5" ,"C", "D", "FAKE", "FREE"]
nb_strategies = len(strategy_labels)

group_size = 5
Z = 100
c = 0.65
beta = 0.25
delta = 6.



# Plot Fig 5.a
nb_points = 10
eps_values_a_b = np.linspace(0., 1., nb_points)
r_values = np.linspace(2., 5., nb_points)
average_F = np.zeros((nb_points, nb_points), dtype = float)

def calcul_average(args):
    i, j, eps, r = args
    game = PGGWithCommitment(group_size, c, r, eps, delta)
    evolver = PairwiseComparison(Z, game)
    transition_matrix,_ = evolver.calculate_transition_and_fixation_matrix_sml(beta)
    stationary_distribution = calculate_stationary_distribution(transition_matrix.transpose())
    avg = sum(stationary_distribution[0:5] * np.arange(1, 6)) / sum(stationary_distribution[0:5])
    return i, j, avg

# Plot Fig 5.b
optimal_F = np.zeros((nb_points, nb_points), dtype = float)

def calcul_optimal(args):
    i, j, eps, r = args
    game = PGGWithCommitment(group_size, c, r, eps, delta)
    evolver = PairwiseComparison(Z, game)
    transition_matrix,_ = evolver.calculate_transition_and_fixation_matrix_sml(beta)
    stationary_distribution = calculate_stationary_distribution(transition_matrix.transpose())
    opti = np.argmax(stationary_distribution[0:5]) + 1
    return i, j, opti


# Plot Fig 5.c
r_c = 2.5
eps_values_c_d = np.linspace(0., 2., nb_points)
delta_values = np.linspace(0., 6., nb_points)
optimal_F_r25 = np.zeros((nb_points, nb_points), dtype = float)

def calcul_optimal_r25(args):
    i, j, eps, delta = args
    game = PGGWithCommitment(group_size, c, r_c, eps, delta)
    evolver = PairwiseComparison(Z, game)
    transition_matrix,_ = evolver.calculate_transition_and_fixation_matrix_sml(beta)
    stationary_distribution = calculate_stationary_distribution(transition_matrix.transpose())
    opti = np.argmax(stationary_distribution[0:5]) + 1
    print("For eps = ", eps, " and delta = ", delta, " the optimal F is ", opti)
    return i, j, opti

# Plot Fig 5.d
r_d = 4.0
optimal_F_r40 = np.zeros((nb_points, nb_points), dtype = float)

def calcul_optimal_r40(args):
    i, j, eps, delta = args
    game = PGGWithCommitment(group_size, c, r_d, eps, delta)
    evolver = PairwiseComparison(Z, game)
    transition_matrix,_ = evolver.calculate_transition_and_fixation_matrix_sml(beta)
    stationary_distribution = calculate_stationary_distribution(transition_matrix.transpose())
    opti = np.argmax(stationary_distribution[0:5]) + 1
    print("For eps = ", eps, " and delta = ", delta, " the optimal F is ", opti)
    return i, j, opti


# --- Bloc principal ---
if __name__ == '__main__':
    # Sécurise le démarrage des processus sur Windows
    freeze_support()


#    # Same for fig a and b
#    # --- Générer les arguments ---
#    args_list = [(i, j, eps_values_a_b[i], r_values[j]) 
#                 for i in range(nb_points) for j in range(nb_points)]
#     
#    # For Fig 5.a
#    # --- Exécuter avec Multiprocessing ---
#    with Pool(processes=cpu_count()) as pool:
#        results = list(tqdm(pool.imap(calcul_average, args_list), total=len(args_list)))
#    # --- Remplir la matrice avec les résultats ---
#    for i, j, avg in results:
#        average_F[i, j] = avg#
#
#    # For Fig 5.b
#    c = 0.65
#    with Pool(processes=cpu_count()) as pool:
#        results = list(tqdm(pool.imap(calcul_optimal, args_list), total=len(args_list)))
#    for i, j, opti in results:
#        optimal_F[i, j] = opti


    # Generate ars for fig c and d 
    args_list_c_d = [(i, j, eps_values_c_d[i], delta_values[j]) 
                 for i in range(nb_points) for j in range(nb_points)]
    
    # For Fig 5.c
    c = 1.4
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(calcul_optimal_r25, args_list_c_d), total=len(args_list_c_d)))
    for i, j, opti in results:
        optimal_F_r25[i, j] = opti

    # For Fig 5.d 
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(calcul_optimal_r40, args_list_c_d), total=len(args_list_c_d)))
    for i, j, opti in results:
        optimal_F_r40[i, j] = opti





#    # --- Vérifier les résultats ---
#    print("Plot Fig 5.a")
#    print("Matrice des moyennes F :")
#    print(average_F)
#    print("Min : ", np.min(average_F))
#    print("Max : ", np.max(average_F))
#   
#   # --- Affichage des résultats ---
#    plt.figure(figsize=(6, 6), dpi=150)
#    levels = np.linspace(3.0, 4.0, 11)
#    contour = plt.contourf(eps_values_a_b, r_values, average_F.T, levels=levels, cmap='viridis')
#    contour_lines = plt.contour(eps_values_a_b, r_values, average_F.T, levels=levels, colors='white', linewidths=0.5)
#    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")
#    plt.xlabel(r'Arrangement cost, $\epsilon$')
#    plt.ylabel('Multiplication factor, r')
#    plt.title('Average level of commitment (COMPF) - Fig 5.a')
#    plt.tight_layout()
#    plt.show()
#
#    print("Plot Fig 5.b")
#    print("c " , c)
#    print("Matrice des optimals F :")
#    print(optimal_F)
#    print("Min : ", np.min(optimal_F))
#    print("Max : ", np.max(optimal_F))
#
#    # Tracer la figure
#    plt.figure(figsize=(6, 6), dpi=150)
#
#    levels = np.linspace(1., 5.0, 5)
#    contour = plt.contourf(eps_values_a_b, r_values, optimal_F.T, levels=levels, cmap='inferno')
#    contour_lines = plt.contour(eps_values_a_b, r_values, optimal_F.T, levels=levels, colors='white', linewidths=0.5)
#    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")
#    plt.xlabel(r'Arrangement cost, $\epsilon$')
#    plt.ylabel('Multiplication factor, r')
#    plt.title('Optimal commitment threshold (F*) - Fig 5.b')
#    plt.colorbar(contour)
#    plt.tight_layout()
#    plt.show()



    print("\n\nPlot Fig 5.c")
    print("c " , c)
    print("Matrice des optimals r_25 F :")
    print(optimal_F_r25)
    print("Min : ", np.min(optimal_F_r25))
    print("Max : ", np.max(optimal_F_r25))


    print("\n\nPlot Fig 5.d")
    print("Matrice des optimal r_40 F :")
    print(optimal_F_r40)
    print("Min : ", np.min(optimal_F_r40))
    print("Max : ", np.max(optimal_F_r40))

    # Define the contour levels between 0 and 5
    levels = np.linspace(0., 5., 6)

    plt.figure(figsize=(10, 5), dpi=150)

    # First graph: Avoidance frequency of cooperation
    plt.subplot(1, 2, 1)
    contour = plt.contourf(eps_values_c_d, delta_values, optimal_F_r25.T, levels=levels, cmap='inferno')
    contour_lines = plt.contour(eps_values_c_d, delta_values, optimal_F_r25.T, levels=levels, colors='white', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")  
    plt.xlabel(r'Arrangement cost, $\epsilon$')
    plt.ylabel(r'Compensation cost, $\delta$')
    plt.title(r'$F^*$ for $r = 2.5$ - Fig 5.c')

    # Second graph: Avoidance frequency of commitment
    plt.subplot(1, 2, 2)
    contour = plt.contourf(eps_values_c_d, delta_values, optimal_F_r40.T, levels=levels, cmap='inferno')
    plt.colorbar(orientation='vertical')
    contour_lines = plt.contour(eps_values_c_d, delta_values, optimal_F_r40.T, levels=levels, colors='white', linewidths=0.5)
    plt.clabel(contour_lines, fontsize=8, fmt="%.2f") 
    plt.xlabel(r'Arrangement cost, $\epsilon$')
    plt.ylabel(r'Compensation cost, $\delta$')
    plt.title(r'$F^*$ for $r = 4.0$ - Fig 5.d')


    plt.tight_layout()
    plt.show()