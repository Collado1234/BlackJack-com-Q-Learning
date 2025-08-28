from abc import ABC, abstractmethod

class GameInterface(ABC):
    @abstractmethod
    def reset(self):
        """Reinicia o estado do jogo e retorna o estado inicial do jogo.
        """
        pass

    @abstractmethod
    def step(self, action):
        """Executa uma ação no ambiente e retorna o novo estado, recompensa e se o jogo terminou."""
        pass

    @abstractmethod
    def get_state(self):
        """Retorna o estado atual do jogo."""
        pass
