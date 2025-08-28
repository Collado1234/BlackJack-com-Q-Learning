from q_learning import QLearningAgent
from blackjack import Blackjack
import matplotlib.pyplot as plt

EPISODES = 50000
REPORT_EVERY = 1000

env = Blackjack()
agent = QLearningAgent(state_size=None, action_size=2)

rewards = []
avg_rewards = []

for episode in range(1, EPISODES+1):
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

    if episode % REPORT_EVERY == 0:
        avg_reward = sum(rewards[-REPORT_EVERY:]) / REPORT_EVERY
        avg_rewards.append(avg_reward)
        print(f"Episódio {episode}, Recompensa média: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")

# Plot do aprendizado
plt.plot(range(0, EPISODES, REPORT_EVERY), avg_rewards)
plt.xlabel("Episódios")
plt.ylabel("Recompensa média")
plt.title("Aprendizado do Agente Q-Learning no Blackjack")
plt.show()
