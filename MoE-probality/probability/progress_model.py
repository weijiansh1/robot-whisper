"""MoE-only streaming prediction of a fixed-horizon physical milestone."""

from .progress_labels import HORIZON
from .pure_moe import MoEWindowMonitor


SCHEMA = "himoe.progress.readout.v1"


class MoEProgressMonitor:
    """Use after inference, with the current action chunk held fixed.

    Calibration covers active prefixes with a full five-chunk observation window
    available under the recorded policy. Deadline-tail use is unvalidated.
    """

    def __init__(self, bundle):
        if (bundle.get("schema") != SCHEMA or bundle.get("horizon_chunks") != HORIZON
                or bundle.get("replan_steps") != 10):
            raise ValueError("not a five-chunk physical-progress model")
        self.reader = MoEWindowMonitor(bundle)

    def update(self, hb_router_probs):
        result = self.reader.update(hb_router_probs)
        return dict(ready=result["ready"], progress_probability=result["success_probability"],
                    horizon_chunks=HORIZON, natural_escape_probability=None)
