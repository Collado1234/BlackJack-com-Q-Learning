from Blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent

# -----------------
# Configurações
# -----------------
EPISODES = 1_000_000
ACTIONS = [0, 1]
SEED = 42

# Cria ambiente
env = BlackjackEnv(seed=SEED)

# =====================
# 1. Treinamento Q-Learning
# =====================
agent = QLearningAgent(
    actions=ACTIONS,
    alpha=0.20,
    gamma=0.90,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.9995
)

print("\n--- Iniciando treinamento Q-Learning ---")
for episode in range(EPISODES):
    state, _, _, _ = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state

    if (episode + 1) % 100000 == 0:
        print(f"Episódio {episode+1}/{EPISODES} concluído...")

# Salva a tabela Q
agent.save_q_table("q_table_treinada.pkl")
print(" Treinamento concluído e Q-table salva em 'q_table_treinada.pkl'")

# =====================
# 2. Carrega agente treinado
# =====================
trained_agent = QLearningAgent(actions=ACTIONS)
trained_agent.load_q_table("q_table_treinada.pkl")

# =====================
# 3. Define agente aleatório (cassino)
# =====================
random_agent = QLearningAgent(
    actions=ACTIONS,
    alpha=0.0,
    gamma=0.0,
    epsilon_start=1.0,
    epsilon_end=1.0,
    epsilon_decay=1.0
)

# =====================
# 4. Teste rápido (10 rodadas cada)
# =====================
def jogar_partidas(agent, nome, num_rodadas=10):
    print(f"\n--- Testando {nome} ({num_rodadas} rodadas) ---")
    for i in range(num_rodadas):
        print(f"\n--- Rodada {i+1} ---")
        state, _, _, message = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, message = env.step(action)
            state = next_state
        print(f"Resultado: {message}, Recompensa: {reward}")

jogar_partidas(trained_agent, "Agente TREINADO", 10)
jogar_partidas(random_agent, "Agente ALEATÓRIO (Cassino)", 10)

# =====================
# 5. Avaliação numérica (10.000 jogos cada)
# =====================
def avaliar_agente(agent, num_jogos=10000):
    wins = losses = draws = 0
    for _ in range(num_jogos):
        state, _, _, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    total = wins + losses + draws
    win_rate = wins / total * 100
    return wins, losses, draws, win_rate

print("\n--- Avaliação estatística ---")
w, l, d, rate = avaliar_agente(trained_agent, 10000)
print(f"Agente TREINADO -> Vitórias: {w}, Derrotas: {l}, Empates: {d}, Winrate: {rate:.2f}%")

w, l, d, rate = avaliar_agente(random_agent, 10000)
print(f"Agente ALEATÓRIO -> Vitórias: {w}, Derrotas: {l}, Empates: {d}, Winrate: {rate:.2f}%")
