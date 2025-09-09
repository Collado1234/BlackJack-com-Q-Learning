import numpy as np
import random
import os
import csv
from Blackjack_env import BlackjackEnv
from q_learning_agent import QLearningAgent

# Garante que a pasta Q_tables e o diretório de resultados existem
Q_TABLE_DIR = "Q_tables"
RESULTS_DIR = "results"
os.makedirs(Q_TABLE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_simulation(env, agent, num_games=1000, policy="q_learning"):
    wins = losses = draws = 0
    for _ in range(num_games):
        state, _, _, _ = env.reset()
        done = False
        while not done:
            if policy == "q_learning":
                if agent and agent.q_table and state in agent.q_table:
                    action = np.argmax(agent.q_table[state])
                else:
                    action = random.choice([0, 1])
            elif policy == "random":
                action = random.choice([0, 1])
            elif policy == "casino":
                player_sum, _, _ = state
                action = 1 if player_sum < 17 else 0
            else:
                raise ValueError("Política inválida!")
            next_state, reward, done, _ = env.step(action)
            state = next_state
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws

def avg_stats(results):
    if not results:
        return (0, 0, 0, 0)
    wins = np.mean([r[0] for r in results])
    losses = np.mean([r[1] for r in results])
    draws = np.mean([r[2] for r in results])
    total = wins + losses + draws
    win_rate = (wins / total) * 100 if total > 0 else 0
    return win_rate, wins, losses, draws

def train_and_evaluate(hyperparams, seed=42, num_eval_seeds=20, verbose=False):
    filename = os.path.join(
        Q_TABLE_DIR,
        f"q_table_a{hyperparams['alpha']}_g{hyperparams['gamma']}_ed{hyperparams['epsilon_decay']}_e{hyperparams['num_episodes']}.pkl"
    )

    env = BlackjackEnv(seed=seed)
    agent = QLearningAgent(
        actions=[0, 1],
        alpha=hyperparams['alpha'],
        gamma=hyperparams['gamma'],
        epsilon_decay=hyperparams['epsilon_decay'],
    )
    
    if os.path.exists(filename):
        if verbose:
            print(f"Carregando {filename}...")
        agent.load_q_table(filename)
    else:
        if verbose:
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

    results_q = []
    for s in range(num_eval_seeds):
        env_eval = BlackjackEnv(seed=s)
        results_q.append(run_simulation(env_eval, agent, policy="q_learning"))

    results_rand = []
    rand_agent = QLearningAgent(actions=[0,1], alpha=0, gamma=0)
    for s in range(num_eval_seeds):
        env_eval = BlackjackEnv(seed=s)
        results_rand.append(run_simulation(env_eval, rand_agent, policy="random"))

    return avg_stats(results_q), avg_stats(results_rand), None, hyperparams

def evaluate_casino_policy(num_eval_seeds=20):
    results_casino = []
    for s in range(num_eval_seeds):
        env_eval = BlackjackEnv(seed=s)
        results_casino.append(run_simulation(env_eval, None, policy="casino"))
    return avg_stats(results_casino)

def save_to_csv(results, filename="blackjack_results.csv"):
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)  # Usando o delimitador padrão (vírgula)
        writer.writerow(['ALPHA', 'GAMMA', 'EPS_DECAY', 'EPISODIOS', 'Q-LEARN WIN%', 'RANDOM WIN%', 'CASINO WIN%'])
        for conf, avg_q, avg_rand, avg_casino in results:
            writer.writerow([
                conf['alpha'],
                conf['gamma'],
                conf['epsilon_decay'],
                conf['num_episodes'],
                f"{avg_q[0]:.2f}",
                f"{avg_rand[0]:.2f}",
                f"{avg_casino[0]:.2f}"
            ])
    print(f"\nResultados salvos em '{filepath}'")

if __name__ == "__main__":
    test_configs = [
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.9999, 'num_episodes': 5000000},
        {'alpha': 0.2, 'gamma': 0.9, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},
        {'alpha': 0.05, 'gamma': 0.9, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},
        {'alpha': 0.1, 'gamma': 0.99, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},
        {'alpha': 0.1, 'gamma': 0.8, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},
        {'alpha': 0.3, 'gamma': 0.9, 'epsilon_decay': 0.9997, 'num_episodes': 2000000},
        {'alpha': 0.01, 'gamma': 0.9, 'epsilon_decay': 0.9999, 'num_episodes': 5000000},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.99999, 'num_episodes': 5000000},
        {'alpha': 0.1, 'gamma': 0.9, 'epsilon_decay': 0.999, 'num_episodes': 500000},
        {'alpha': 0.5, 'gamma': 0.9, 'epsilon_decay': 0.999, 'num_episodes': 200000},
        {'alpha': 0.2, 'gamma': 0.99, 'epsilon_decay': 0.9997, 'num_episodes': 2000000},
        {'alpha': 0.1, 'gamma': 0.5, 'epsilon_decay': 0.9995, 'num_episodes': 1000000},
        {'alpha': 0.05, 'gamma': 0.95, 'epsilon_decay': 0.99999, 'num_episodes': 3000000},
    ]

    print("Avaliando a política de cassino...")
    avg_casino = evaluate_casino_policy(num_eval_seeds=20)
    print(f"Taxa de vitória do Cassino (média de 20 seeds): {avg_casino[0]:.2f}%")
    
    results = []
    for config in test_configs:
        avg_q, avg_rand, _, conf = train_and_evaluate(config, verbose=True, num_eval_seeds=20)
        results.append((conf, avg_q, avg_rand, avg_casino))

    print("\n" + "="*150)
    print(" " * 30 + "Relatório de Desempenho no Blackjack (MÉDIA de 20 seeds) - Q-Learning vs Aleatório vs Estratégia Cassino")
    print("="*150)
    header = f"{'ALPHA':<8} {'GAMMA':<8} {'EPS_DECAY':<12} {'EPISÓDIOS':<12} | {'Q-LEARN WIN%':<15} {'RANDOM WIN%':<15} {'CASINO WIN%':<15}"
    print(header)
    print("-" * 150)

    for conf, avg_q, avg_rand, avg_casino in results:
        line = f"{conf['alpha']:<8.2f} {conf['gamma']:<8.2f} {conf['epsilon_decay']:<12.4f} {conf['num_episodes']:<12} | {avg_q[0]:<15.2f} {avg_rand[0]:<15.2f} {avg_casino[0]:<15.2f}"
        print(line)

    print("="*150)

    save_to_csv(results)