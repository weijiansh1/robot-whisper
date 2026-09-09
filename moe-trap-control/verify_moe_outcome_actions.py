#!/usr/bin/env python3
"""Physically re-execute every new outcome-model suffix without policy queries."""

import verify_mode_dispatched_actions as verifier
from moe_outcome_models import ARMS, PROTOCOL

verifier.ARMS,verifier.PROTOCOL = ARMS,PROTOCOL
verifier.__file__ = __file__

if __name__ == "__main__":
    verifier.main()
