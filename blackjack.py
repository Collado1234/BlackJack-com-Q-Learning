import random
from deck import Deck


class Blackjack:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.deck = Deck(self.rng)
        self.reset()
    
    def reset(self):
        self.deck.reset()
        self.player_hand = [self.deck.draw(), self.deck.draw()]
        self.dealer_hand = [self.deck.draw(), self.deck.draw()]
        return self._get_obs()
    
    def _get_obs(self):
        return (
            self._hand_value(self.player_hand),
            Deck.card_value(self.dealer_hand[0]), #carta virada do dealer
        )
    
    def _hand_value(self, hand):
        total = sum(Deck.card_value(card) for card in hand)
        if any(card.rank == 1 for card in hand) and total + 10 <= 21:
            total += 10
        return total
    
    def step(self, action):
        #action: 0 = parar, 1 = pedir carta
        if action == 1:
            self.player_hand.append(self.deck.draw())
            if self._hand_value(self.player_hand) > 21:
                return self._get_obs(), -1, True # jogador perde
        else: #parar - dealer joga
            while self._hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.deck.draw())
            return self._get_obs(), self._determine_winner(), True
        return self._get_obs(), 0, False
    
    def _determine_winner(self):
        player_val = self._hand_value(self.player_hand)
        dealer_val = self._hand_value(self.dealer_hand)
        if dealer_val > 21 or player_val > dealer_val:
            return 1
        elif player_val < dealer_val:
            return -1
        else:
            return 0