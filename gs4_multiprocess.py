import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List

from egttools.games import AbstractNPlayerGame
from egttools import sample_simplex, calculate_nb_states
from egttools.analytical import PairwiseComparison
from egttools.utils import calculate_stationary_distribution
from egttools.plotting import draw_invasion_diagram


from multiprocessing import Pool, cpu_count 
from multiprocessing import freeze_support
from tqdm import tqdm

class PGGWithLongCommitment(AbstractNPlayerGame):
    def __init__(self, 
                 group_size: int,   # Number of participants in the PGG
                 c: float,          # Cost of cooperation
                 r: float,          # Enhancing factor (multiplier)
                 eps: float,        # Cost to propose a commitment
                 delta: float,      # Cost for not respecting the commitment
                 R: int             # Number of rounds
                 ):
        # Initialize superclass
        AbstractNPlayerGame.__init__(self, 20, group_size)  # Adjusted for additional strategies

        # Parameters and configurations
        self.nb_strategies_ = 20
        self.group_size_ = group_size
        self.strategies = [
            "COMP1_1", "COMP2_1", "COMP3_1", "COMP4_1", 
            "COMP1_2", "COMP2_2", "COMP3_2", "COMP4_2",
            "COMP1_3", "COMP2_3", "COMP3_3", "COMP4_3", 
            "COMP1_4", "COMP2_4", "COMP3_4", "COMP4_4", 
            "C", "D", "FAKE", "FREE"
        ]

        self.c = c
        self.r = r
        self.eps = eps
        self.delta = delta
        self.R = R  # Number of rounds
        self.nb_group_configurations_ = self.nb_group_configurations()  # Calculate number of possible group configurations
        self.calculate_payoffs()  # Calculate payoffs for each strategy in different group configurations

    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        # Initialize payoffs for each strategy in the group
        game_payoffs[:] = 0.
        COMP1_1, COMP2_1, COMP3_1, COMP4_1= 0, 1, 2, 3
        COMP1_2, COMP2_2, COMP3_2, COMP4_2= 4,5,6,7
        COMP1_3, COMP2_3, COMP3_3, COMP4_3= 8,9,10,11
        COMP1_4, COMP2_4, COMP3_4, COMP4_4= 12,13,14,15
        COOPERATOR, DEFECTOR, FAKER, FREE = 16,17,18,19

        # Calculate the number of each type of player in the group
        nb_commitment = sum(group_composition[:16])  # Number of commitment strategies

        if nb_commitment == 0:  # Classical PGG
            nb_contributors = group_composition[COOPERATOR]
        else:  # With commitments
            nb_contributors = nb_commitment + group_composition[COOPERATOR] + group_composition[FREE]

        nb_fake = group_composition[FAKER]  # Number of fake players
        nb_accept = nb_fake + nb_contributors

        # Calculate the total contribution and reward for the group
        total_contribution = self.c * nb_contributors
        total_reward = self.r * total_contribution
        individual_reward = total_reward / self.group_size_
        # Determine F (minimum commitment threshold) and F_prime
        F = int(0)
        comp = int(0)
        for i in group_composition[:16]:
            comp = comp%4
            comp += 1
            if(comp>F and i>0):
                F = comp
        j = int(0)
        F_prime = int(0)
        comp_prime= int(0)
        while j <=3:
            nbr = sum(group_composition[j*4:4+j*4])
            comp_prime +=1
            if(0<nbr):
                F_prime = comp_prime
            j+=1
        #F = next((i + 1 for i, count in enumerate(group_composition[:25]) if count > 0), 0)
        #F_prime = next((int(self.strategies[i].split('_')[-1]) for i in range(25) if group_composition[i] > 0 and '_' in self.strategies[i]), 0)
        if nb_commitment==0:
           for index, strategy_count in enumerate(group_composition):
            if strategy_count > 0:
                game_payoffs[index] += individual_reward
                if self.strategies[index] == "C":
                    game_payoffs[index] -=  self.c
                game_payoffs[index] = self.R*game_payoffs[index]
        elif F <= nb_accept: 
          for index, strategy_count in enumerate(group_composition):
            if strategy_count > 0:
                game_payoffs[index] += individual_reward
                if self.strategies[index] == "C":
                    game_payoffs[index] -=  self.c
                if(nb_commitment > 0):
                    if "COMP" in self.strategies[index]:
                        game_payoffs[index] -=  (self.c + (self.eps/nb_commitment))
                    elif self.strategies[index] == "FREE":
                        game_payoffs[index] -= self.c
                if F <= nb_contributors:
                    if "COMP" in self.strategies[index]:
                        game_payoffs[index] +=((nb_fake*self.delta)/nb_commitment)
                    elif self.strategies[index] == "FAKE":
                        game_payoffs[index] -= self.delta
                    game_payoffs[index] = self.R*game_payoffs[index]
                elif F_prime <= nb_contributors:
                    game_payoffs[index] = self.R*game_payoffs[index]
                else:
                    if "COMP" in self.strategies[index]:
                        game_payoffs[index] +=((nb_fake*self.delta)/nb_commitment)
                    elif self.strategies[index] == "FAKE":
                        game_payoffs[index] -= self.delta

    def calculate_payoffs(self) -> np.ndarray:
        """Calculate and store the payoffs for each strategy in the game."""

        # Initialize an array to store payoffs for each configuration
        payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)

        # Loop over all possible group configurations
        for i in range(self.nb_group_configurations_):
            # Generate a sample group composition
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            group_composition = np.array(group_composition, dtype=float)

            # Play the game with the given group composition
            self.play(group_composition, payoffs_container)

            # Update the payoff for each strategy based on this configuration
            for strategy_index, strategy_payoff in enumerate(payoffs_container):
                self.update_payoff(strategy_index, i, strategy_payoff)

            # Reset the payoff container for the next configuration
            payoffs_container[:] = 0

        return self.payoffs()

# Define parameters
group_size = 4
c = 0.65
eps = 0.25
delta = 20
Z = 100  # Population size
beta = 0.25  # Selection intensity
#R_values = np.logspace(2, 100 , 6)  # Example range for R
R_values = np.array([2, 5, 10, 20, 50, 100])
r_values = np.linspace(2.0, 5.0, 6)  # Example range for r
optimal_F_prime = np.zeros((len(r_values), len(R_values)))

def calculate_optimal_prime(args):
    i, j, R, r = args 
    optimal_F = np.zeros((5,), dtype = float)
    game = PGGWithLongCommitment(group_size, c, r, eps, delta, R)
    evolver = PairwiseComparison(Z, game)
    transition_matrix, _ = evolver.calculate_transition_and_fixation_matrix_sml(beta)
    stationary_distribution = calculate_stationary_distribution(transition_matrix.transpose())
    #print(stationary_distribution)
    comp_1 = stationary_distribution[:4]
    #print(comp_1)
    comp_2 = stationary_distribution[4:8]
    #print(comp_2)
    comp_3 = stationary_distribution[8:12]
    #print(comp_3)
    comp_4 = stationary_distribution[12:16]
    #print(comp_4)
    optimal_F = np.argmax([sum(comp_1) / len(comp_1), sum(comp_2) / len(comp_2),sum(comp_3) / len(comp_3),sum(comp_4) / len(comp_1)]) +1
    return i, j, optimal_F


if __name__ == '__main__':
    # Sécurise le démarrage des processus sur Windows
    freeze_support()

    # --- Générer les arguments ---
    args_list = [(i, j, R_values[i], r_values[j]) 
                 for i in range(len(R_values)) for j in range(len(r_values))]


    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(calculate_optimal_prime, args_list), total=len(args_list)))
    # --- Remplir la matrice avec les résultats ---
    for i, j, optimal in results:
        optimal_F_prime[i, j] = optimal

    print(optimal_F_prime)