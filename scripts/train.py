#!/usr/bin/env python3
"""Train diffusion policy."""

import sys
sys.argv = [sys.argv[0], "train"] + sys.argv[1:]

from diffusion_manipulation.cli import main
main()
