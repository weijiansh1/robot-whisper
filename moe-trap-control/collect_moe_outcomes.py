#!/usr/bin/env python3
"""Choose an audited controller from an actual causal MoE entry probe."""

import json
from pathlib import Path

import numpy as np

import collect_control_bank as bank
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest
from moe_outcome_models import (PROTOCOL, SETTINGS, MODELS, FeatureHistory, choose, decode_arm)

original_choice = bank.choose_operator
bank.PROTOCOL, bank.SETTINGS, bank.decode_arm = PROTOCOL, SETTINGS, decode_arm


class OutcomeSession(bank.ControlBankSession):
    def entry_probe(self, obs, live, trace, main, q, steps):
        probe = super().entry_probe(obs,live,trace,main,q,steps)
        path = Path(self.task["outcome_model_path"])
        if digest(path) != self.task["outcome_model_sha256"]:
            raise ValueError("Frozen outcome model changed")
        fitted = json.loads(path.read_text())
        if self.args.main_id in fitted["training_main_ids"]:
            raise ValueError("Online parent leaked into model fit")
        history = FeatureHistory()
        for row in main[:q]:
            history.update(row[PROBS_KEY],int(row["action_steps_before"]))
        feature = history.probe(probe[PROBS_KEY],steps)
        np.testing.assert_allclose(feature,self.task["entry_feature"],rtol=0,atol=1e-8)
        candidates = {}
        for method in MODELS+("clock_knn",):
            operator,scores = choose(fitted["model"],method,feature)
            if operator != self.task["selector_choices"][method]["operator"]:
                raise ValueError("Live model selection differs from frozen pre-treatment choice")
            np.testing.assert_allclose(scores,self.task["selector_choices"][method]["scores"],rtol=0,atol=1e-6)
            candidates[method] = dict(operator=operator,scores=scores.tolist())
        family,repeat = decode_arm(self.args.control["arm"])
        donor = None
        if family in MODELS+("clock_knn",):
            operator = candidates[family]["operator"]
        elif family.startswith("shuffle_"):
            selected = self.task["shuffled_choices"][self.args.control["arm"]]
            operator,donor = selected["operator"],selected["donor_main_id"]
        else:
            operator = original_choice(family,probe["component_scores"],probe["selector_thresholds"],self.args.main_id,repeat)
        self.selection = dict(protocol=PROTOCOL,arm=self.args.control["arm"],operator=operator,
            feature=feature.tolist(),models=candidates,donor_main_id=donor,
            model_sha256=self.task["outcome_model_sha256"],query=q,action_steps_before=steps,
            input_sha256=probe["input_sha256"].item().decode(),actual_probe=True)
        atomic_json(self.args.output/"selection.json",self.selection)
        return probe

    def branches(self):
        bank.choose_operator = lambda *_:self.selection["operator"]
        try:
            super().branches()
        finally:
            bank.choose_operator = original_choice
        self.report.update(outcome_model_sha256=self.task["outcome_model_sha256"],
            selection_sha256=digest(self.args.output/"selection.json"))
        self.save()


if __name__ == "__main__":
    import collect_native_long_modes as worker
    worker.NativeModesSession = OutcomeSession
    worker.main()
