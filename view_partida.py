from Blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
import numpy as np
import random

# -----------------
# Configurações
# -----------------
SEED = 42
NUM_EPISODES = 100000  # irrelevante, já que não há aprendizado
ACTIONS = [0, 1] 

random.seed(SEED)

# Inicializa o ambiente e o agente "sem aprendizado"
env = BlackjackEnv(seed=SEED)
agent = QLearningAgent(
    actions=ACTIONS,
    alpha=0.0,          # Sem aprendizado
    gamma=0.0,          # Irrelevante sem aprendizado
    epsilon_start=1.0,  # Sempre aleatório
    epsilon_end=1.0,    # Mantém epsilon = 1
    epsilon_decay=1.0   # Não reduz exploração
)

print("\n--- Jogando 10 rodadas aleatórias (sem aprendizado) ---")
for i in range(10):
    print(f"\n--- Rodada {i+1} ---")
    state, _, _, message = env.reset() 
    player_hand, dealer_hand = env.get_hands()
    
    print(f"Carta visível do Dealer: {BlackjackEnv.format_card(dealer_hand[0])}")
    print(f"Sua mão: {[BlackjackEnv.format_card(card) for card in player_hand]}")
    
    done = False
    
    while not done:
        # Aqui o agente joga 100% aleatório
        action = agent.choose_action(state)
        action_text = "Parar" if action == 0 else "Pedir Carta"
        
        print(f"Agente decidiu: {action_text}")
        
        next_state, reward, done, message = env.step(action)
        
        # Se pediu carta, mostra a nova mão
        if action == 1 and not done:
            player_hand, _ = env.get_hands()
            print(f"Sua nova mão: {[BlackjackEnv.format_card(card) for card in player_hand]}")
        
        if done:
            print("\n--- Resultado Final ---")
            player_hand, dealer_hand = env.get_hands()
            print(f"Mão final do Jogador: {[BlackjackEnv.format_card(card) for card in player_hand]} (Total: {env._hand_value(player_hand)})")
            print(f"Mão final do Dealer: {[BlackjackEnv.format_card(card) for card in dealer_hand]} (Total: {env._hand_value(dealer_hand)})")
            print(message)
        
        state = next_state
        
    print(f"Recompensa final: {reward}")

# -----------------
# Simulação Avançada
# -----------------
print("\n--- Simulação Avançada (100 jogos aleatórios) ---")

wins = 0
losses = 0
draws = 0

for i in range(100):
    state, _, _, _ = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)  # sempre aleatório
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1

# Relatório final
total_games = wins + losses + draws
win_rate = (wins / total_games) * 100 if total_games > 0 else 0
print("\n--- Relatório Final ---")
print(f"Total de jogos: {total_games}")
print(f"Vitórias: {wins}")
print(f"Derrotas: {losses}")
print(f"Empates: {draws}")
print(f"Taxa de Vitória: {win_rate:.2f}%")
