#!/usr/bin/env python3
"""Q1 census runner. The rule of record tools/t22b_cross_request.py is used
UNMODIFIED (byte-identical; all thresholds, floors, verdict logic and gated
k={8,16} untouched). Sole delta, applied at import time before main(): the
REPORTED-k set K_ALL is restricted to (8,16,32) — dropping k=64, which the
rule's own docstring classifies as 'reported, not gated' and which Stage A++
measured as a deterministic Monte-Carlo abort (t22b_run.log FATAL at decode
k=64 L0). No other name in the module is rebound."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import t22b_cross_request as rule
assert rule.K_ALL == (8, 16, 32, 64)
rule.K_ALL = (8, 16, 32)
rule.main()
