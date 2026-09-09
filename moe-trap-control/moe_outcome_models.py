"""Causal MoE observations and four frozen estimators of intervention response."""

import copy
import hashlib
import json

import numpy as np

from control_bank import FAMILIES as BANK_FAMILIES, SETTINGS as BANK_SETTINGS
from mode_control import thresholds_at
from v8_closed_loop import limits, score_status
from v82_closed_loop import V82Monitor

PROTOCOL = "moe_control.moe_outcome_models.v1"
OPERATORS = BANK_FAMILIES[:-2]
MODELS = ("density", "markov", "knn_uplift", "ridge_uplift")
FAMILIES = ("resample", "side_minus", "moe_switch", "clock_knn")+MODELS+tuple("shuffle_"+m for m in MODELS)
ARMS = ("native",)+tuple(m+"_r"+str(r) for m in FAMILIES for r in (0,1))
MODEL_SETTINGS = dict(feature_dimensions=77, components=8, clusters=8, response_steps=32,
    minimum_query=8, ridge=10., neighbors=5, covariance_shrinkage=.5, transition_prior=2.,
    observation_clip=8., latent_clip=6., minimum_gain=.0001, seed=20260909,
    selection="MoE prefix and elapsed steps only; no task ID, object pose, outcome, or future frames",
    training="all previous 150 mains and paired fixed-operator outcomes; task-isolated cross-validation",
    reference_weight="each main has total weight one; no chunk-wise train/test split",
    projection="weighted standardized covariance PCA; eight whitened coordinates",
    shuffle="deterministic derangement of pre-treatment choices, exact operator counts per repeat",
    missing="zero-filled normalized components with five explicit validity indicators")
FEATURE_NAMES = tuple(prefix+"_"+letter for prefix in ("score","delta","mean4","valid") for letter in "FAPIC")+tuple(
    prefix+"_layer"+str(layer) for prefix in ("entropy","denoise_flow","query_mobility") for layer in range(8))+tuple(
    "expert_load_"+str(expert) for expert in range(32))+("elapsed_fraction",)
SETTINGS = dict(BANK_SETTINGS, outcome_models=MODEL_SETTINGS,
    policy_weights_training=False, selector_fitting=True,
    evaluation="100 fresh original Long initializations 15..24; all usable first v8.2 alarms")


def decode_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unknown outcome-model arm")
    return ("native",-1) if arm == "native" else (arm[:-3],int(arm[-1]))


class FeatureHistory:
    """Consume only executed policy queries; a counterfactual probe uses a copy."""

    def __init__(self):
        self.monitor = V82Monitor()
        self.thresholds, self.margins = limits()
        self.scores = []
        self.previous_root = None

    def update(self, probability, steps):
        status = self.monitor.update(probability)
        q = self.monitor.v7.query
        scores = score_status(status)
        valid = np.isfinite(scores)
        normalized = np.where(valid,(scores-thresholds_at(q,self.thresholds))/self.margins,0.)
        normalized = np.clip(normalized,-10.,10.)
        delta = normalized-self.scores[-1] if self.scores else np.zeros(5)
        self.scores.append(normalized)
        mean = np.mean(self.scores[-4:],axis=0)
        p = np.maximum(np.asarray(probability,dtype=np.float64)[:,:,1:,:],0.)
        p /= p.sum(-1,keepdims=True)
        root = np.sqrt(p)
        entropy = -(p*np.log(np.maximum(p,1e-12))).sum(-1).mean(axis=(1,2))/np.log(32.)
        flow = np.linalg.norm(np.diff(root,axis=1),axis=-1).mean(axis=(1,2))/np.sqrt(2.)
        mobility = (np.linalg.norm(root-self.previous_root,axis=-1).mean(axis=(1,2))/np.sqrt(2.)
                    if self.previous_root is not None else np.zeros(8))
        self.previous_root = root.copy()
        result = np.r_[normalized,delta,mean,valid.astype(float),entropy,flow,mobility,
                       p.mean(axis=(0,1,2)),float(steps)/520.]
        if result.shape != (77,) or not np.isfinite(result).all():
            raise ValueError("Invalid causal MoE features")
        return result

    def probe(self, probability, steps):
        return copy.deepcopy(self).update(probability,steps)


def project(model, values):
    x = np.asarray(values,float)
    standardized = np.clip((x-np.asarray(model["mean"]))/np.asarray(model["scale"]),-8.,8.)
    return np.clip(standardized @ np.asarray(model["projection"]),-6.,6.)


def cluster(model, z):
    values = np.atleast_2d(z)
    return np.square(values[:,None,:]-np.asarray(model["centers"])[None,:,:]).sum(-1).argmin(1)


def density_value(model, z):
    values = np.atleast_2d(z)
    likelihood = []
    for label in (0,1):
        state = model["gaussians"][label]
        delta = values-np.asarray(state["mean"])
        likelihood.append(-.5*(np.einsum("ni,ij,nj->n",delta,np.asarray(state["precision"]),delta)+state["logdet"]))
    return np.asarray(likelihood[1])-likelihood[0]


def choose(model, method, feature):
    if method not in MODELS+("clock_knn",):
        raise ValueError("Unknown fitted model")
    z = project(model,feature)
    design = np.r_[1.,z]
    if method == "density":
        predicted = np.asarray(model["response_coefficients"]) @ design
        scores = density_value(model,np.clip(predicted,-6.,6.))
    elif method == "markov":
        state = int(cluster(model,z)[0])
        scores = np.asarray(model["operator_transitions"])[:,state,:] @ np.asarray(model["committor"])
    elif method in ("knn_uplift","clock_knn"):
        distance = (np.square(np.asarray(model["entry_latent"])-z).sum(1) if method == "knn_uplift"
                    else np.square(np.asarray(model["entry_clock"])-np.asarray(feature)[-1]))
        nearest = np.argsort(distance,kind="stable")[:min(5,len(distance))]
        weights = np.exp(-distance[nearest]/max(float(distance[nearest[-1]]),1e-6))
        scores = weights @ np.asarray(model["uplifts"])[nearest]/(weights.sum()+1.)
    else:
        scores = np.asarray(model["uplift_coefficients"]) @ design
    scores = np.round(scores-scores[0],8)
    selected = int(np.argmax(scores)) if np.max(scores) > .0001 else 0
    return OPERATORS[selected], scores


def derangement(main_ids, method, repeat):
    ordered = sorted(main_ids)
    if len(ordered) < 2:
        return dict(zip(ordered,ordered))
    payload = json.dumps([PROTOCOL,method,repeat,"shuffle"],separators=(",",":"))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16],"little"))
    for _ in range(100):
        order = rng.permutation(len(ordered))
        if np.all(order != np.arange(len(ordered))):
            return {key:ordered[int(j)] for key,j in zip(ordered,order)}
    return dict(zip(ordered,ordered[1:]+ordered[:1]))
