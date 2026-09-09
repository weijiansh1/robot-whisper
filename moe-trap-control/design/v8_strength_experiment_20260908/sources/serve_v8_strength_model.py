#!/usr/bin/env python3
"""Select the strength policy in the existing isolated IPC server lifecycle."""

import serve_v8_feature_model as server
from v8_strength_control import StrengthPolicy

# Multiprocessing imports this entry point before invoking replica targets.
# The binding is local to these new processes; the old servers are untouched.
server.V8FeaturePolicy = StrengthPolicy

if __name__ == "__main__":
    raise SystemExit(server.main())
