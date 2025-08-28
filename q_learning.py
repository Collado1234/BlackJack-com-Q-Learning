import random
from collections import defaultdict # defaultdict for easier state-action value management

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount factor
        self.epsilon = epsilon #exploration rate
        self.Q = defaultdict(lambda: {0: 0.0, 1: 0.0}) # actions 0/1

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.get_valid_actions(state))
        else:
            q_vals = self.Q[state]
            return max(q_vals, key=q_vals.get)
        
    def learn(self, num_episodes=10000):
        stats = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                best_next = max(self.Q[next_state].values(), default=0)
                self.Q[state][action] += self.alpha * (reward + self.gamma * best_next - self.Q[state][action])
                state = next_state
                total_reward += reward
            
            stats.append(total_reward)
        return stats
