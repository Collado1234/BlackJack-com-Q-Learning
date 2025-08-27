# env.py
import abc
import random
from typing import List, Tuple, Any

Card = int
Hand = list[Card]   #mão do jogador ou do dealer
State = tuple[int, int, bool]  # (player_sum, dealer_upcard, usable_ace)

class AbstractGame(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> Any:
        pass

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        pass

    @abc.abstractmethod
    def get_valid_actions(self, state: Any) -> List[int]:
        pass

def draw_card() -> Card:
    card = random.randint(1, 13) # 1-10 são valores normais, 11-13 são figuras (valem 10)
    return min(card, 10)    
    




class BlackJackGame(AbstractGame):
    def __init__(self):
        self.state = None
        self.done = False

    def reset(self) -> Any:
        self.state = (0, 0)  # Exemplo de estado inicial
        self.done = False
        return self.state

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        if self.done:
            raise Exception("O jogo terminou. Por favor, resete o ambiente.")
        
        # Lógica simplificada para o exemplo
        if action == 1:  # Suponha que 1 é "hit"
            self.state = (self.state[0] + 10, self.state[1])  # Adiciona 10 ao jogador
        elif action == 0:  # Suponha que 0 é "stand"
            self.done = True
        
        reward = 0.0
        if self.state[0] > 21:
            reward = -1.0
            self.done = True
        elif self.done:
            reward = 1.0 if self.state[0] <= 21 else -1.0
        
        return self.state, reward, self.done, {}

    def get_valid_actions(self, state: Any) -> List[int]:
        if self.done:
            return []
        return [0, 1]  # Stand ou Hit