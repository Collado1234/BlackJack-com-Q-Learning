import numpy as np
import random
from Blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent
import os

# Garante que a pasta Q_tables existe
Q_TABLE_DIR = "Q_tables"
os.makedirs(Q_TABLE_DIR, exist_ok=True)

def run_simulation(env, agent, num_games=100):
    """
    Roda uma simulação e retorna as estatísticas de vitória, derrota e empate. 
    """
    wins = 0
    losses = 0
    draws = 0
    
    for _ in range(num_games):
        state, _, _, _ = env.reset()
        done = False
        
        while not done:
            action = np.argmax(agent.q_table[state])
            next_state, reward, done, _ = env.step(action)
            state = next_state
            
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
            
    return wins, losses, draws

def train_and_evaluate(hyperparams, seed=42): #testar diferentes seeds
    """
    Treina e avalia um agente com base em um conjunto de hiperparâmetros.
    Retorna a taxa de vitória e o aproveitamento.
    """
    random.seed(seed)
    
    env = BlackjackEnv(seed=seed)
    agent = QLearningAgent(
        actions=[0, 1],
        alpha=hyperparams['alpha'],
        gamma=hyperparams['gamma'],
        epsilon_decay=hyperparams['epsilon_decay'],
    )
    
    # Cria um nome de arquivo único para esta configuração dentro da pasta Q_tables
    filename = os.path.join(
        Q_TABLE_DIR,
        f"q_table_a{hyperparams['alpha']}_g{hyperparams['gamma']}_ed{hyperparams['epsilon_decay']}_e{hyperparams['num_episodes']}.pkl"
    )

    if os.path.exists(filename):
        print(f"Carregando {filename}...")
        agent.load_q_table(filename)
    else:
        print(f"Treinando com: {hyperparams}...")
        for episode in range(hyperparams['num_episodes']):
            state, _, _, _ = env.reset()
            done = False
            
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
        
        agent.save_q_table(filename)
    
    wins, losses, draws = run_simulation(env, agent, num_games=1000)
    total_games = wins + losses + draws
    win_rate = (wins / total_games) * 100 if total_games > 0 else 0
    
    # Novo cálculo do aproveitamento
    points = (wins * 3) + (draws * 1)
    max_points = total_games * 3
    utilization_rate = (points / max_points) * 100 if max_points > 0 else 0
    
    return win_rate, wins, losses, draws, utilization_rate

if __name__ == "__main__":
    test_configs = [
    # Configuração base
    {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},

    # Mais episódios, menor decaimento
    {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.9999, 'num_episodes': 5000000},

    # Maior taxa de aprendizado
    {'alpha': 0.2, 'gamma': 0.9, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},

    # Menor taxa de aprendizado
    {'alpha': 0.05, 'gamma': 0.9, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},

    # Maior fator de desconto
    {'alpha': 0.1, 'gamma': 0.99, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},

    # Menor fator de desconto
    {'alpha': 0.1, 'gamma': 0.8, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},

    # Aprendizado mais agressivo + mais episódios
    {'alpha': 0.3, 'gamma': 0.9, 'epsilon_decay': 0.9997, 'num_episodes': 2000000},

    # Aprendizado lento + muitos episódios
    {'alpha': 0.01, 'gamma': 0.9, 'epsilon_decay': 0.9999, 'num_episodes': 5000000},

    # Exploração alta por mais tempo
    {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.99999, 'num_episodes': 5000000},

    # Exploração curta (epsilon cai rápido)
    {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.999, 'num_episodes': 500000},

    # Muito aprendizado curto
    {'alpha': 0.5, 'gamma': 0.9, 'epsilon_decay': 0.999, 'num_episodes': 200000},

    # Gamma alto com mais aprendizado
    {'alpha': 0.2, 'gamma': 0.99, 'epsilon_decay': 0.9997, 'num_episodes': 2000000},

    # Gamma baixo para estratégias mais imediatistas
    {'alpha': 0.1, 'gamma': 0.5, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},

    # Configuração “exploradora extrema”
    {'alpha': 0.05, 'gamma': 0.95, 'epsilon_decay': 0.99999, 'num_episodes': 3000000},
    
    # Jogo completamente aleatório (sem aprendizado)
    {'alpha': 0.0, 'gamma': 0.0, 'epsilon_decay': 1.0, 'num_episodes': 100000}
]


    results = []
    for config in test_configs:
        win_rate, wins, losses, draws, utilization_rate = train_and_evaluate(config)
        results.append({
            'Config': config,
            'Vitória %': f"{win_rate:.2f}%",
            'Aproveitamento': f"{utilization_rate:.2f}%",
            'Vitórias': wins,
            'Derrotas': losses,
            'Empates': draws
        })

    # Imprimir relatório em formato de tabela
    print("\n" + "="*110)
    print("                 Relatório de Desempenho do Q-Learning no Blackjack")
    print("="*110)
    
    header = f"{'ALPHA':<8} {'GAMMA':<8} {'EPS_DECAY':<12} {'EPISÓDIOS':<12} {'VITÓRIA %':<12} {'APROVEITAMENTO':<15} {'VITÓRIAS':<12} {'DERROTAS':<12} {'EMPATES':<12}"
    print(header)
    print("-" * 110)
    
    for res in results:
        config = res['Config']
        line = f"{config['alpha']:<8.2f} {config['gamma']:<8.2f} {config['epsilon_decay']:<12.4f} {config['num_episodes']:<12} {res['Vitória %']:<12} {res['Aproveitamento']:<15} {res['Vitórias']:<12} {res['Derrotas']:<12} {res['Empates']:<12}"
        print(line)
        
    print("="*110)
