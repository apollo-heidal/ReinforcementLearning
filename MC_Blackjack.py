import numpy as np
from numpy.lib.shape_base import dsplit


class BlackjackGame:
    def __init__(self) -> None:
        # states 2-21 are considered for state values and returns arrays
        self.state_values = dict((x, 0) for x in range(2, 22))

        # returns are a dict of k=state, value=[total return for state, n_occurences of state]
        self.returns = dict((x, [0, 0]) for x in range(2, 22))

        self.player_policy = {}
        # parameters: player sum (s), dealer's visible card (d), usable ace (a)
        # stored in dictionary where key is (s,d,a) and value is T|F
        for s in range(2, 22):
            for d in range(2, 12):
                for a in [True, False]:
                    if s < 20:
                        self.player_policy[(s, d, a)] = True
                    else:
                        self.player_policy[(s, d, a)] = False

    def playGame(self):
        # trace is an ordered list which captures the players states, actions, and rewards (SARSA...)
        game_trace = []

        player_hand = list(np.random.randint(low=2, high=12, size=(2,)))
        # use one ace if the starting hand is [ace, ace]
        if 11 in player_hand:
            player_hand[player_hand.index(11)] = 1

        dealer_hand = list(np.random.randint(low=2, high=12, size=(2,)))
        dealer_visible_card = dealer_hand[0]

        # players turn
        while True:
            game_trace.append(sum(player_hand))

            # returns T|F from policy
            player_hit = self.player_policy[(
                sum(player_hand), dealer_visible_card, 11 in player_hand)]
            game_trace.append(player_hit)
            if player_hit:
                player_hand.append(np.random.randint(low=2, high=12))

                if sum(player_hand) > 21:
                    # check for usable ace
                    if 11 in player_hand:
                        player_hand[player_hand.index(11)] = 1
                    else:
                        # player goes bust and dealer need not play
                        game_trace.append(-1)
                        return game_trace
            else:
                # player sticks
                break

        # dealer's turn
        while True:
            dealer_hit = sum(dealer_hand) < 17
            if dealer_hit:
                dealer_hand.append(np.random.randint(low=2, high=12))
            else:
                if 11 in dealer_hand:
                    # usable ace
                    dealer_hand[dealer_hand.index(11)] = 1
                    continue
                # dealer sticks
                break

        # decide game
        p_sum = sum(player_hand)
        d_sum = sum(dealer_hand)

        # if the player went bust, the function has already returned
        if d_sum > 21:
            # dealer went bust, so player wins
            game_trace.append(1)
            return game_trace

        if p_sum <= 21 and d_sum <= 21:
            # neither went bust
            if p_sum > d_sum:
                # player wins
                game_trace.append(1)
            elif d_sum > p_sum:
                # dealer wins
                game_trace.append(-1)
            else:
                # draw
                game_trace.append(0)

        return game_trace

    def simulate(self):
        count = 0
        while True:
            game_trace = self.playGame()
            if count % 10**5 == 0:
                self.prettyPrintValues(count)
            count += 1

            # there is only one reward per episode with is w/l/d = 1/-1/0 so that is also the return
            ret = game_trace.pop()
            for i in range(len(game_trace)//2):
                s = game_trace[i*2]
                a = game_trace[i*2+1]

                new_sum = self.returns[s][0] + ret
                n_visits = self.returns[s][1] + 1

                # update returns dict
                self.returns[s] = [new_sum, n_visits]

                # update state value
                self.state_values[s] = new_sum / n_visits

    def prettyPrintValues(self, count):
        print(f"{count}:")
        for k,v in self.state_values.items():
            if k == 2:
                continue
            print(f"{k} = {round((v+1)/2, 3)}", end="\t")
            histo = int((v+1) * 20) * "*"
            print(histo)
        print()



if __name__ == "__main__":
    game = BlackjackGame()
    game.simulate()
