import numpy as np
import random
from collections import defaultdict
import pickle 

class QLearningAgent:    
    def __init__(self,
                 actions: list,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.9995):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))


    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])
        
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        max_future_q = 0 if done else np.max(self.q_table[next_state])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

     # Função para salvar a tabela Q
    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Tabela Q salva em {filename}")

    # Função para carregar a tabela Q
    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, 'rb') as f:
                loaded_q_table = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
            print(f"Tabela Q carregada de {filename}")
            return True
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado. Iniciando novo treinamento.")
            return False