from Blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
import numpy as np
import random
import os.path

# -----------------
# Configurações
# -----------------cls
SEED = 42
Q_TABLE_FILE = "q_table.pkl"
NUM_EPISODES = 1000000
ACTIONS = [0, 1] 

random.seed(SEED)

# Inicializa o ambiente e o agente
env = BlackjackEnv(seed=SEED)
agent = QLearningAgent(actions=ACTIONS)

# -----------------
# Lógica de Treinamento
# -----------------
# Tenta carregar a tabela Q salva
if not agent.load_q_table(Q_TABLE_FILE):
    print("Iniciando treinamento...")
    for episode in range(NUM_EPISODES):
        state, _, _, _ = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        
        if (episode + 1) % 100000 == 0:
            print(f"Episódio {episode + 1}/{NUM_EPISODES}, Epsilon: {agent.epsilon:.4f}")

    print("\nTreinamento concluído!")
    agent.save_q_table(Q_TABLE_FILE)

# -----------------
# Demonstração e Análise
# -----------------
print("\n--- Analisando o desempenho em 10 rodadas ---")
for i in range(10):
    print(f"\n--- Rodada {i+1} ---")
    state, _, _, message = env.reset() 
    player_hand, dealer_hand = env.get_hands()
    
    print(f"Carta visível do Dealer: {BlackjackEnv.format_card(dealer_hand[0])}")
    print(f"Sua mão: {[BlackjackEnv.format_card(card) for card in player_hand]}")
    
    done = False
    
    while not done:
        action = np.argmax(agent.q_table[state])
        action_text = "Parar" if action == 0 else "Pedir Carta"
        
        print(f"Agente decidiu: {action_text}")
        
        next_state, reward, done, message = env.step(action)
        
        # Se o agente pediu carta, mostra a nova mão
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
print("\n--- Simulação Avançada ---")
print("Rodando 100 jogos para avaliar o desempenho...")

# Variáveis para contabilizar os resultados
wins = 0
losses = 0
draws = 0

# Loop de simulação
for i in range(100):
    state, _, _, _ = env.reset()
    done = False
    
    while not done:
        # O agente sempre explora a melhor ação
        action = np.argmax(agent.q_table[state])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
    # Contabiliza o resultado do jogo
    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1

# Exibe o relatório final
total_games = wins + losses + draws
win_rate = (wins / total_games) * 100 if total_games > 0 else 0
print("\n--- Relatório Final ---")
print(f"Total de jogos: {total_games}")
print(f"Vitórias: {wins}")
print(f"Derrotas: {losses}")
print(f"Empates: {draws}")
print(f"Taxa de Vitória: {win_rate:.2f}%")