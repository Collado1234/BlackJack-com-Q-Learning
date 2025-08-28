import random 
from game_interface import GameInterface
from Deck import Deck, Card

class BlackjackEnv(GameInterface):
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.deck = Deck(self.rng)
        self.player_hand: list[Card] = []
        self.dealer_hand: list[Card] = []
        self.done = False        

    def reset(self):
        self.deck.reset()
        self.player_hand = [self.deck.draw(), self.deck.draw()]  # jogador pega duas cartas
        self.dealer_hand = [self.deck.draw(), self.deck.draw()]  # dealer pega duas cartas
        self.done = False
        return self.get_state()
    
    def step(self, action: int):
        #action: 0 = parar, 1 = pedir carta
        if self.done:
            print("O jogo terminou. Por favor, reinicie o ambiente.")
            return self.get_state(), 0, True

        if action == 1:  # pedir carta
            self.player_hand.append(self.deck.draw())
            if self._hand_value(self.player_hand) > 21:
                self.done = True
                return self.get_state(), -1, True # jogador perde
            return self.get_state(), 0, False # jogo continua

        if action == 0: 
            self.done = True
            player_val = self._hand_value(self.player_hand) 

            # Vez do dealer
            while self._hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.deck.draw())
            
            reward = self._determine_winner()
            return self.get_state(), reward, True
        
    def get_state(self):
        # Estado: (soma da mão do jogador, valor da carta visível do dealer, se há um Ás na mão do jogador)
        player_sum = self._hand_value(self.player_hand)  # cartas que o jogador tem
        dealer_upcard = Deck.card_value(self.dealer_hand[0]) # carta visível do dealer
        usable_ace = any(card.rank == 1 for card in self.player_hand) and player_sum + 10 <= 21 # Ás utilizável
        return (player_sum, dealer_upcard, usable_ace)
    
    def _hand_value(self, hand):
        total = sum(Deck.card_value(card) for card in hand)
        # Ás (1 0u 11)
        if any(card.rank == 1 for card in hand) and total + 10 <= 21: # se não estourar, conta como 11
            total += 10
        return total
    
    def _is_blackjack(self, hand):
        return len(hand) == 2 and self._hand_value(hand) == 21
    
    def _determine_winner(self):
        player_val = self._hand_value(self.player_hand)
        dealer_val = self._hand_value(self.dealer_hand)

        player_bj = self._is_blackjack(self.player_hand)
        dealer_bj = self._is_blackjack(self.dealer_hand)

        if player_val > 21:
            return -1  # Perdeu 
        if dealer_val > 21 or player_val > dealer_val:
            return 1   # Ganhou
        elif player_val < dealer_val:
            return -1 # Perdeu
        else:
            return 0 # Empate

        

    

