import matplotlib.pyplot as plt
from blackjack import Blackjack
from agents import QLearningAgent

def train(episodes=5000):
    env = Blackjack(seed=42)
    agent = QLearningAgent(actions=[0, 1])  # 0=parar, 1=pedir
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        # monitoramento a cada 500 episódios
        if (ep+1) % 500 == 0:
            avg = sum(rewards[-500:]) / 500
            print(f"Episódio {ep+1}: média de recompensa = {avg:.2f}")

    return rewards

if __name__ == "__main__":
    rewards = train(5000)

    # gráfico suavizado
    import numpy as np
    window = 200
    smooth = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(smooth)
    plt.title("Aprendizado do agente no Blackjack")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa média")
    plt.show()
