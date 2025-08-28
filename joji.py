import random
from collections import namedtuple

Card = namedtuple("Card", ["rank", "suit"])

class Deck:
    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        suits = ["♠", "♥", "♦", "♣"]
        ranks = list(range(1, 14))  # 1 = Ás, 11=Valete, 12=Dama, 13=Rei
        self.cards = [Card(rank, suit) for rank in ranks for suit in suits]
        self.rng.shuffle(self.cards)

    def draw(self):
        if not self.cards:
            self.reset()  # baralho acabou → reembaralha
        return self.cards.pop()

    @staticmethod
    def card_value(card: Card) -> int:
        # Ás = 1 (ou 11, mas isso é tratado no jogo, não no baralho)
        return min(card.rank, 10)
