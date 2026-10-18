"""Malmuth-Harville ICM."""


def icm(stacks, payouts):
    """Expected prize of each player. Busted players (stack 0) take the lowest remaining places."""
    n = len(stacks)
    eq = [0.0] * n
    alive = [i for i in range(n) if stacks[i] > 0]
    dead = [i for i in range(n) if stacks[i] <= 0]
    if dead:
        prize = lambda k: payouts[k] if k < len(payouts) else 0.0
        share = sum(prize(k) for k in range(len(alive), n)) / len(dead)
        for i in dead:
            eq[i] = share
    places = min(len(payouts), len(alive))

    def rec(remaining, total, place, prob):
        if place == places:
            return
        for i in remaining:
            p = prob * stacks[i] / total
            eq[i] += p * payouts[place]
            if place + 1 < places:
                rec([j for j in remaining if j != i], total - stacks[i], place + 1, p)

    rec(alive, sum(stacks[i] for i in alive), 0, 1.0)
    return eq
