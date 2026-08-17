"""Unit tests for the pure game logic (no camera/model needed)."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from game_logic import MOVES, decide, play_round, random_move  # noqa: E402


@pytest.mark.parametrize(
    "player,computer,expected",
    [
        ("rock", "scissors", "win"),
        ("rock", "paper", "lose"),
        ("rock", "rock", "tie"),
        ("paper", "rock", "win"),
        ("paper", "scissors", "lose"),
        ("paper", "paper", "tie"),
        ("scissors", "paper", "win"),
        ("scissors", "rock", "lose"),
        ("scissors", "scissors", "tie"),
    ],
)
def test_decide(player, computer, expected):
    assert decide(player, computer) == expected


def test_decide_rejects_invalid_move():
    with pytest.raises(ValueError):
        decide("lizard", "rock")


def test_random_move_in_moves():
    rng = random.Random(42)
    for _ in range(50):
        assert random_move(rng) in MOVES


def test_play_round_shape():
    rng = random.Random(0)
    out = play_round("rock", rng)
    assert set(out) == {"player_move", "computer_move", "result"}
    assert out["player_move"] == "rock"
    assert out["computer_move"] in MOVES
    assert out["result"] in {"win", "lose", "tie"}
