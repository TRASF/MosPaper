"""
Semi-Supervised Learning (SSL) algorithms for mosquito wingbeat classification.

This module contains implementations of SSL algorithms like:
- FixMatch: A simple and effective SSL algorithm with consistency regularization
- FlexMatch: An enhanced version of FixMatch with flexible thresholding
"""

# Import SSL implementations
from . import fixmatch
from . import flexmatch