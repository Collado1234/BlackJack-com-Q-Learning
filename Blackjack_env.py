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
        self.player_hand = [self.deck.draw(), self.deck.draw()]
        self.dealer_hand = [self.deck.draw(), self.deck.draw()]
        self.done = False
        return self.get_state(), 0, False, "Novo jogo iniciado."

    def step(self, action: int):
        if self.done:
            return self.get_state(), 0, True, "O jogo já terminou."
        
        message = ""
        reward = 0
        
        if action == 1: # Pedir carta (Hit)
            self.player_hand.append(self.deck.draw())
            player_val = self._hand_value(self.player_hand)
            message = f"Jogador pediu uma carta. Sua mão agora tem valor {player_val}."
            if player_val > 21:
                reward = -1
                self.done = True
                message += " Jogador estourou!"
            return self.get_state(), reward, self.done, message

        elif action == 0: # Parar (Stick)
            self.done = True
            
            # Dealer joga
            while self._hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.deck.draw())
                message += f"Dealer pegou uma carta. A mão do dealer agora é {self._hand_value(self.dealer_hand)}.\n"
            
            reward, outcome = self._determine_winner()
            message += f"Jogador parou. {outcome}"
            
            return self.get_state(), reward, self.done, message            
    
    def get_state(self):
        player_sum = self._hand_value(self.player_hand)
        dealer_card_value = Deck.card_value(self.dealer_hand[0])
        usable_ace = any(card.rank == 1 for card in self.player_hand) and player_sum <= 11
        return (player_sum, dealer_card_value, usable_ace)

    def get_hands(self):
        # Retorna as cartas exatas do jogador e do dealer
        return self.player_hand, self.dealer_hand
    
    # Adicione uma função para formatar a carta de forma legível
    @staticmethod
    def format_card(card: Card):
        ranks = {1: 'Ás', 11: 'Valete', 12: 'Dama', 13: 'Rei'}
        rank_name = ranks.get(card.rank, str(card.rank))
        return f"{rank_name} de {card.suit}"

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
            
            if player_bj and not dealer_bj:
                return 1.5, "Jogador venceu com um BlackJack!"
            elif dealer_bj and not player_bj:
                return -1, "Dealer venceu com um BlackJack!"

            if player_val > 21:
                return -1, "Jogador estourou."
            if dealer_val > 21:
                return 1, "Dealer estourou. Jogador venceu!"
            elif player_val > dealer_val:
                return 1, "Jogador venceu!"
            elif player_val < dealer_val:
                return -1, "Dealer venceu."
            else:
                return 0, "Empate."

   

    

    

    