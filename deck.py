import random
from collections import namedtuple

Card = namedtuple("Card", ["rank", "suit"])

class Deck:
    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()
        self.reset()
        pass
    
    def reset(self):
        suits = ["♠", "♥", "♦", "♣"]
        ranks = list(range(1, 14))  # 1 = Ás, 11=Valete, 12=Dama, 13=Rei
        self.cards = [Card(rank, suit) for rank in ranks for suit in suits] # Cria o baralho
        self.rng.shuffle(self.cards)
        
    def draw(self):
        if not self.cards:
            self.reset()  # baralho acabou reembaralha
        return self.cards.pop()
    
    @staticmethod
    def card_value(card: Card) -> int: #O valor da carta (Ás=1 ou 11, figuras=10) mas cisso tá no blackjack_env        
        return min(card.rank, 10)