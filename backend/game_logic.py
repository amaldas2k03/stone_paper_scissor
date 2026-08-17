"""
game_logic.py
-------------
Pure Rock-Paper-Scissors rules: choosing the computer's move and deciding a
round's outcome. Kept free of any web/framework code so it can be unit-tested in
isolation.
"""

from __future__ import annotations

import random
from typing import Dict

MOVES = ("rock", "paper", "scissors")

# What each move beats.
_BEATS: Dict[str, str] = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


def random_move(rng: random.Random | None = None) -> str:
    """Return a uniformly random computer move."""
    chooser = rng or random
    return chooser.choice(MOVES)


def decide(player_move: str, computer_move: str) -> str:
    """
    Decide the round from the player's perspective.

    Returns one of "win", "lose", "tie".

    Raises:
        ValueError: if either move is not a valid RPS move.
    """
    if player_move not in MOVES:
        raise ValueError(f"invalid player move: {player_move!r}")
    if computer_move not in MOVES:
        raise ValueError(f"invalid computer move: {computer_move!r}")

    if player_move == computer_move:
        return "tie"
    return "win" if _BEATS[player_move] == computer_move else "lose"


def play_round(player_move: str, rng: random.Random | None = None) -> Dict[str, str]:
    """
    Play one round: generate a computer move and resolve it.

    Returns a dict with player_move, computer_move and result. Assumes
    player_move has already been validated as a real gesture ('rock', 'paper' or
    'scissors') by the caller.
    """
    computer_move = random_move(rng)
    return {
        "player_move": player_move,
        "computer_move": computer_move,
        "result": decide(player_move, computer_move),
    }
