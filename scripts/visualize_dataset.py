#!/usr/bin/env python3
"""Visualize dataset demos."""

import sys
sys.argv = [sys.argv[0], "visualize"] + sys.argv[1:]

from diffusion_manipulation.cli import main
main()
