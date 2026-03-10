#!/usr/bin/env python3
"""Evaluate trained diffusion policy."""

import sys
sys.argv = [sys.argv[0], "evaluate"] + sys.argv[1:]

from diffusion_manipulation.cli import main
main()
