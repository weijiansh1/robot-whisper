"""Shared definitions for the discrete-physical-event / MoE-routing study.

sim_state (47) layout, verified against formal/worker0/sim_layout.json:
    [0]      time
    [1:8]    robot0_joint1..7            <- 7 ARM JOINT ANGLES (present!)
    [8:10]   gripper0_finger_joint1,2
    [10:17]  moka_pot_1 free joint (pos 3, quat 4, wxyz)
    [17:24]  moka_pot_2 free joint (pos 3, quat 4)
    [24]     flat_stove_1_button
    [25:32]  arm qvel (7)
    [32:34]  finger qvel (2)
    [34:40]  moka_pot_1 qvel (6)
    [40:46]  moka_pot_2 qvel (6)
    [46]     stove button qvel
"""
import json
import numpy as np

RUN = "/home/jovyan/work/himoe-vla/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828"
HUB = ("/home/jovyan/work/himoe-vla/VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
       "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32")
OUT = "/home/jovyan/work/himoe-vla/analysis_events"

S_ARM = slice(1, 8)
S_FING = slice(8, 10)
S_P1 = slice(10, 13)
S_Q1 = slice(13, 17)
S_P2 = slice(17, 20)
S_Q2 = slice(20, 24)
S_BTN = 24
S_ARMV = slice(25, 32)

# Panda nominal joint limits (rad)
Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
Q_RANGE = Q_HI - Q_LO

TABLE_Z = 0.9661          # verified resting z for both moka pots
DENOISE = 9

# thresholds reused from analysis_dryrun_v10/physical_label_definitions.json
TH = dict(drop_m=0.035, goal_enter_m=0.055, goal_exit_m=0.10,
          lift_enter_m=0.035, lift_exit_m=0.012)


def goal_refs():
    d = json.load(open(f"{RUN}/analysis_dryrun_v10/physical_label_definitions.json"))
    return np.asarray(d["success_terminal_references"]["moka_pot_1_joint0"], dtype=np.float64)


G = goal_refs()                       # (82,3) pooled acceptable terminal positions
G_CENT = G.mean(0)
# two stove slots
from scipy.cluster.vq import kmeans2 as _km
_C, _ = _km(G, 2, seed=0, minit="++")
G_SLOT = _C[np.argsort(_C[:, 1])]     # (2,3)


def dgoal(p):
    """min distance from each row of p (n,3) to the goal reference set."""
    return np.sqrt(((p[:, None, :] - G[None]) ** 2).sum(-1)).min(1)


def quat_z(q):
    """object body z-axis in world frame, quat given as (w,x,y,z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)


def tilt_deg(q):
    zax = quat_z(q)
    n = np.linalg.norm(zax, axis=1) + 1e-9
    return np.degrees(np.arccos(np.clip(zax[:, 2] / n, -1, 1)))


def phys_block(ss, ps, ac):
    """Derived per-chunk physical quantities. ss (K,47), ps (K,8), ac (K,10,7)."""
    K = ss.shape[0]
    eef = ps[:, :3].astype(np.float64)
    aa = ps[:, 3:6].astype(np.float64)
    fing = ps[:, 6:8].astype(np.float64)
    q = ss[:, S_ARM].astype(np.float64)
    qv = ss[:, S_ARMV].astype(np.float64)
    p1 = ss[:, S_P1].astype(np.float64)
    p2 = ss[:, S_P2].astype(np.float64)
    q1 = ss[:, S_Q1].astype(np.float64)
    q2 = ss[:, S_Q2].astype(np.float64)
    btn = ss[:, S_BTN].astype(np.float64)
    width = np.abs(fing[:, 0] - fing[:, 1])
    d = dict(
        eef=eef, aa=aa, fing=fing, width=width, q=q, qv=qv,
        p1=p1, p2=p2, quat1=q1, quat2=q2, btn=btn,
        tilt1=tilt_deg(q1), tilt2=tilt_deg(q2),
        dg1=dgoal(p1), dg2=dgoal(p2),
        z1=p1[:, 2], z2=p2[:, 2],
        lift1=p1[:, 2] - TABLE_Z, lift2=p2[:, 2] - TABLE_Z,
        de1=np.linalg.norm(eef - p1, axis=1), de2=np.linalg.norm(eef - p2, axis=1),
        d12=np.linalg.norm(p1 - p2, axis=1),
        deg=dgoal(eef),
        ac=ac.astype(np.float64), K=K,
    )
    # joint-limit proximity (normalised, 0 = at a limit, 0.5 = mid range)
    nrm = np.minimum(q - Q_LO, Q_HI - q) / Q_RANGE
    d["jlim_min"] = nrm.min(1)
    d["jlim_nrm"] = nrm
    return d
