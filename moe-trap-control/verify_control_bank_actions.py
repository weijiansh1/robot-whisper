#!/usr/bin/env python3
"""Apply the unchanged full MuJoCo action verifier to the frozen controller bank."""

import verify_mode_dispatched_actions as verifier
from control_bank import ARMS, PROTOCOL


if __name__ == "__main__":
    verifier.ARMS, verifier.PROTOCOL = ARMS, PROTOCOL
    # Child workers must enter this adapter and inherit the same frozen arm list.
    verifier.__file__ = __file__
    verifier.main()
