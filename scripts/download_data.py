#!/usr/bin/env python3
"""Download robomimic datasets."""

import sys
sys.argv = [sys.argv[0], "download"] + sys.argv[1:]

from diffusion_manipulation.cli import main
main()
