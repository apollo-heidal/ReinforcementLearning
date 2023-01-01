## blackjack
# 2+ participants: dealer and player(s)
# player dealt two cards to themself and players
# one of dealers cards is hidden until the end
# player(s) hit or stay to receive more cards or end their hand, respectively
# tie goes to dealer

# S,A,R,S,A,R...T

import random
import sys
import numpy as np
import matplotlib.pyplot as plt


class DeckEmptyException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        print("The deck is empty!")



class Deck:
    def __init__(self, n=-1) -> None:
        # number of decks used can be passed as argument
        # defaults to infinite deck
        # aces are considered 11 throughout sim until player goes bust
        # losing triggers an action to see if the player can stay in the game by using ace as a 1

        cards = dict()
        # init deck
        for c in range(2,12):
            if n == -1: # psuedoinfinite deck
                cards[c] = sys.maxsize
            else:
                if c == 10:
                    cards[c] = 16*n
                else:
                    cards[c] = 4*n
        self.cards = cards

    def getDeck(self):
        return self.cards

    def getCard(self):
        # since the deck dictionary has keys 1-11
        # we must map the actual cards in a deck
        # to this range (otherwise the chance of getting a face card is way too low)
        if max(self.cards.values()) < 1:
            raise DeckEmptyException()

        while True:
            card = random.randint(1,13)
            if card in [10,11,12,13]:
                card = 10
            elif card == 1:
                card = 11

            if self.cards[card] < 1:
                continue
            else:
                break

        self.cards[card] -= 1
        return card

    


class Dealer:
    def __init__(self, deck: Deck) -> None:
        self.deck = deck
        self.hand = []
        self.is_hitting = True


    def dealCard(self):
        return self.deck.getCard()


    def turn(self):
        '''
        Simple aggro policy:
            hand sum > 17 -> stick
        '''
        if sum(self.hand) <= 30:
            self.hit()
        else:
            # check for usable ace
            if 11 in self.hand:
                self.hand[self.hand.index(11)] = 1
            else:
                self.is_hitting = False


    def hit(self):
        self.hand.append(self.dealCard())




class Player:
    def __init__(self, dealer: Dealer, learning_rate=0.1) -> None:
        self.dealer = dealer

        self.hand = []
        self.is_hitting = True

        self.init_policy = self.initPolicy()
        self.learned_policy = dict()
        self.learning_rate = learning_rate


    def hit(self):
        card = self.dealer.dealCard()
        self.hand.append(card)


    def stick(self):
        self.is_hitting = False

    
    def initPolicy(self):
        p = dict()
        # random confident policy
        for s in range(21):
            p[s] = 0.5 + (0.5 * random.random())
        return p

    
    def turn(self):
        if sum(self.hand) >= 21:
            self.is_hitting = False
            return

        # lookup the state tuple in the policy table
        # use init_policy if the player has a new hand
        hashable_hand = tuple(self.hand)
        if self.learned_policy.__contains__(hashable_hand):
            # learned policy
            # explore when confidence is 50% +/-10%
            if self.learned_policy[hashable_hand] >= 0.6:
                self.hit()
            elif self.learned_policy[hashable_hand] < 0.4:
                self.stick()
            else:
                if np.random.choice([True, False]):
                    self.hit()
                else:
                    self.stick()
        else:
            # init policy
            s = sum(self.hand)
            if self.init_policy[s] < 0.5:
                self.stick()
            else:
                self.hit()


    def updatePolicy(self, win: bool):
        '''After each game, update this policy.
        Iterate backwards across the player's hand as such:
        - hash hand
        - raise or lower the confidence of sticking for that hashed state
        - this method should heavily weight reinforcement of later actions
        take into account how far over the player went; going from 12 to 22 should be reinforced more than going from 20 to 30
        also, include inverse weights for hands that are far from 21; something like hand = [2,2] should lose very little confidence even if it was in a losing hand

        distance factor: sum(hand) / 30
        30 is the highest possible hand (after using aces) and will maximize the negative policy update
        conversely, 2 ( / 30 ) will scale down the policy update for both winning and losing because this state is 

        '''

        while len(self.hand) >= 2: # player is dealt 2 cards at start so learning should not be done on a hand of 1 card
            hashable_hand = tuple(self.hand)
            # init policy at 50%
            if not self.learned_policy.__contains__(hashable_hand):
                self.learned_policy[hashable_hand] = 0.5

            # correction_factor increases the update more for wins
            # and lowers the update more for losses
            # "it penalizes incorrect confidence harder" 
            # this makes confidence = 1 | 0 policy limits

            # distance factor raises penalty for going over more
            # and lowers penalty for small sum hands that ended up losing
            distance_factor = (sum(self.hand) - 21) / 21
        
            # win update 
            if win:
                confidence_correction = 1 - self.learned_policy[hashable_hand]
                policy_update = self.learning_rate * confidence_correction * distance_factor
                self.learned_policy[hashable_hand] += policy_update
            else:
                confidence_correction = 0 - self.learned_policy[hashable_hand]
                policy_update = self.learning_rate * confidence_correction * distance_factor
                self.learned_policy[hashable_hand] += policy_update

            self.hand.pop()



            
class Game:
    def __init__(self, n_decks=-1) -> None:
        self.deck = Deck(n_decks)
        self.dealer = Dealer(self.deck)
        self.player = Player(self.dealer)

        # analysis
        self.player_hands = []
        self.dealer_hands = []
        self.wins = { True: 0, False: 0 }
    

    def simulate(self, n_games: np.uint):
        for g in range(n_games):
            if g % 25 == 0:
                if g == 0:
                    continue
                win_rate = self.wins[True] / sum(self.wins.values())
                print(f"After {g} games the player has won {win_rate * 100}% of games.")
            
            self.play()

        # helper plots
        plt.figure(figsize=(30,15))
        plt.plot(self.player_hands, label="player")
        plt.plot(self.dealer_hands, label="dealer")
        plt.legend()
        plt.show()


    def play(self):
        player_wins = None

        # setup: give dealer and player two cards
        for _ in range(2):
            self.dealer.hit()
            self.player.hit()

        # player hits
        while self.player.is_hitting:
            self.player.turn()

        # dealer hits
        while self.dealer.is_hitting:
            self.dealer.turn()

        player_sum = sum(self.player.hand)
        dealer_sum = sum(self.dealer.hand)

        # game decision
        if player_sum <= 21 and dealer_sum <= 21:
            if player_sum > dealer_sum:
                player_wins = True
            else:
                player_wins = False
        elif dealer_sum > 21:
            player_wins = True
        else:
            player_wins = False

        # analysis
        self.player_hands.append(player_sum)
        self.dealer_hands.append(dealer_sum)
        self.wins[player_wins] += 1

        # update policy
        self.player.updatePolicy(win=player_wins)

        # reset hands
        self.player.hand = []
        self.dealer.hand = []

    

if __name__ == "__main__":
    game = Game()
    game.simulate(n_games = 500)