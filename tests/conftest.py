"""Gemeinsame Fixtures: die Mini-Instanzen des Handbeispiels, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def hand_train():
    """Vier Trainingsinstanzen, Kapazität 100, je acht Gegenstände. Von Hand: Best Fit 4, 3, 4, 4; First Fit 4, 3, 5, 4;
    Worst Fit 5, 4, 5, 3; S1(0,10) 4, 3, 4, 4 Behälter."""
    return np.array([
        [25, 35, 65, 35, 50, 70, 30, 40],
        [10, 55, 15, 35, 15, 60, 20, 55],
        [45, 65, 15, 75, 15, 75, 25, 55],
        [30, 35, 70, 50, 25, 15, 30, 25],
    ], dtype=np.int64)


@pytest.fixture
def hand_test():
    """Zwei Testinstanzen. Von Hand: Best Fit 4, 4; First Fit 5, 4; Worst Fit 5, 3; S1(0,10) 5, 4 Behälter."""
    return np.array([[55, 65, 70, 20, 30, 65, 30, 40], [30, 45, 35, 10, 60, 45, 35, 25]], dtype=np.int64)


@pytest.fixture
def hand_matrix():
    """Behälterzahlen der vier Regeln (BF, FF, WF, S1) auf den vier Trainingsinstanzen, von Hand."""
    return np.array([[4, 3, 4, 4], [4, 3, 5, 4], [5, 4, 5, 3], [4, 3, 4, 4]], dtype=np.int64)


@pytest.fixture
def hand_matrix_test():
    """Behälterzahlen der vier Regeln auf den zwei Testinstanzen, von Hand."""
    return np.array([[4, 4], [5, 4], [5, 3], [5, 4]], dtype=np.int64)
