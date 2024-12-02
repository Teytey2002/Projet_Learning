Eliott le Pallec
class PGGwithCommitment(AbstractNPlayerGame):
    def __init__(self,
                 group_size: int,  # number of participants in the PGG
                 c: float,          # cost of cooperation
                 r: float,          # enhancing factor (multiplier)
                 eps: float,          # cost for propose a commitment
                 F: int,            #nombre de gens qui acceptent le contrat pour pouvoir jouer le PGG
                 delta : float,     #cost to dont respect the commitment
                ):
        AbstractNPlayerGame.__init__(self, 5, group_size)
        self.nb_strategies_ = 5
        self.group_size_ = group_size

        self.strategies = ["COMP", "C", "D", "FAKE", "FREE"]  # Cooperate, Defect, Non-participate, Punish
        self.c = c
        self.r = r
        self.eps = eps
        self.delta = delta
        self.F = F

        self.nb_group_configurations_ = self.nb_group_configurations()

        # payoffs in different group configurations
        self.calculate_payoffs()


    def play(self,
             group_composition: Union[List[int], np.ndarray],
             game_payoffs: np.ndarray
            ) -> None:
        game_payoffs[:] = 0.
        nb_commitment = group_composition[0]
        if(nb_commitment==0){
            nb_contributors = group_composition[1]
        }
        else:
          nb_contributors = group_composition[0] + group_composition[1] +group_composition[4]  # number of contributors
        nb_fake = group_composition[3]  # number of fake
        nb_accept = nb_fake + nb_contributors
        
        total_contribution = self.c * (nb_contributors)
        total_reward = self.r * total_contribution
        individual_reward = total_reward / (self.group_size_)
        #je dois mettre une condition if pour voir quand le PGG n'est pas joué, sinon on lance le PGG
        #ici c'est dans le cas où on joue et que tout les comp veulent bien lancer le PGG
        if(nb_accept >= self.F or nb_commitment==0): #le nombre de gens qui acceptent le contrat doit être plus grand que F
        #ou le nombre de COMP est égal à 0 et on joue juste un PGG classique
          for index, strategy_count in enumerate(group_composition):
              game_payoffs[index] += individual_reward 
              if self.strategies[index] == "COMP":
                game_payoffs[index] -=  (c + (self.eps/nb_commitment)- ((nb_fake*self.delta)/nb_commitment))
              elif self.strategies[index] == "C":
                game_payoffs[index] -=  self.c
              elif self.strategies[index] == "FAKE":
                game_payoffs[index] -= self.delta
              elif self.strategies[index] == "FREE":
                if(nb_commitment > 0):
                  game_payoffs[index] -= self.c


    def calculate_payoffs(self) -> np.ndarray:
        payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)

        for i in range(self.nb_group_configurations_):
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            group_composition = np.array(group_composition, dtype=float)

            self.play(group_composition, payoffs_container)

            for strategy_index, strategy_payoff in enumerate(payoffs_container):
                self.update_payoff(strategy_index, i, strategy_payoff)


            payoffs_container[:] = 0

        return self.payoffs()