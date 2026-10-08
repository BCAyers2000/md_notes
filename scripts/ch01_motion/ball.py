"""The ball of Chapter 1: thrown straight up at V0 from the origin.

Shared by the Chapter 1 figure scripts so that every figure and every
quoted number describe the same throw.
"""

from scipy.constants import g as G  # standard gravity, 9.80665 m/s²

V0 = 12.0  # launch speed, m/s
T_TOP = V0 / G  # time to the top, s
T_LAND = 2 * V0 / G  # time to return to the hand, s
H_MAX = V0**2 / (2 * G)  # greatest height, m
