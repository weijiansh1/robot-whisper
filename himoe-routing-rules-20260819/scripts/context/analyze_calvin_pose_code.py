#!/usr/bin/env python3
"""Is the routing a pose code on CALVIN too -- and of the joints or the tool?

On LIBERO the state token's routing was ~96% a nonlinear function of the robot's
own 8 numbers, on five tasks and three checkpoints.  CALVIN-D is a different
simulator, a different action space (`calvin_d_joint`, the other AS signature),
and its `robot_obs` is 15-dim:

    0-2   tcp position          3-5   tcp orientation (euler)
    6     gripper opening       7-13  the seven arm joint angles
    14    gripper action

LIBERO's 8-dim state could not tell "the routing encodes the arm" from "the
routing encodes the end effector", because it only ever contained the end
effector.  Here both are present, and the checkpoint was trained in joint space,
so the question is sharp: does it encode the joints, the tool, or both?

Caveat stated up front: forward kinematics makes the two highly collinear, so a
clean attribution is not always available; the honest reading is the *gap*
between what each explains on its own.

Grouping unit for leave-one-out is the language instruction (33 of them), which
is the closest thing here to LIBERO's scenes and is a strictly harder test, since
the prompt is itself a model input.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import zarr
from scipy.stats import rankdata
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).resolve().parent
CAP = (HERE / "VLA_MUI_HUB/cache/HiMoE-VLA/calvin_d/task_D_D/routes-v1/server")
HB_LAYER = [2, 3, 4, 5, 12, 13, 14, 15]
NAME = ([f"tcp {a}" for a in "xyz"] + [f"tcp r{i}" for i in (1, 2, 3)]
        + ["夹爪开度"] + [f"关节{i}" for i in range(1, 8)] + ["夹爪动作"])
EEF = list(range(0, 7))            # tcp pose + gripper width
JNT = list(range(7, 14))           # the seven joint angles


def loso_auc(X, y, g):
    num = den = 0.0
    for h in np.unique(g):
        tr, te = g != h, g == h
        if not (0 < y[tr].sum() < tr.sum()) or not (0 < y[te].sum() < te.sum()):
            continue
        sc = StandardScaler().fit(X[tr])
        f = LogisticRegression(C=1.0, max_iter=1500).fit(sc.transform(X[tr]), y[tr])
        v = f.decision_function(sc.transform(X[te]))
        n1 = int(y[te].sum())
        r = rankdata(v)
        num += r[y[te]].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * int((~y[te]).sum())
    return num / den if den else float("nan")


def loso_pred(make, X, Y, g):
    pred = np.zeros_like(Y)
    for h in np.unique(g):
        tr, te = g != h, g == h
        sc = StandardScaler().fit(X[tr])
        v = make(sc.transform(X[tr]), Y[tr], sc.transform(X[te]))
        pred[te] = v.reshape(pred[te].shape)
    return pred


LIN = lambda A, Y, B: Ridge(alpha=10.0).fit(A, Y).predict(B)


def RBF(A, Y, B):
    ny = Nystroem(gamma=0.3, n_components=500, random_state=0).fit(A)
    return Ridge(alpha=1.0).fit(ny.transform(A), Y).predict(ny.transform(B))


def frac(Y, p):
    return 1 - ((Y - p) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def main() -> int:
    z = zarr.open(str(CAP / "routes.zarr"), mode="r")
    n = z["hb_expert_ids"].shape[0]
    ids = np.asarray(z["hb_expert_ids"][:, :, 0, 0, :]).astype(np.int64)
    hot = np.zeros((n, 8, 32), np.float32)
    np.put_along_axis(hot, ids, 1.0, -1)
    prob = np.asarray(z["hb_router_probs"][:, :, 0, 0, :],
                      np.float32).reshape(n, -1)
    X = np.load(CAP / "state.npy")
    prompts = np.array(json.loads((CAP / "prompts.json").read_text()))
    g = np.unique(prompts, return_inverse=True)[1]
    print("CALVIN-D，%d 个控制步，%d 条不同指令做留一单元" % (n, len(np.unique(g))))
    print("robot_obs 各维取值范围（确认布局）:")
    for j in (0, 3, 6, 7, 13, 14):
        print("  %-10s [%+.3f, %+.3f]" % (NAME[j], X[:, j].min(), X[:, j].max()))

    print("\n=== 1. 正向：本体感受 → 选中哪些专家（每层中位 |AUC-0.5|）===")
    print("层    全部 15 维   末端 7 维   关节角 7 维    LIBERO 对照(8维)")
    lib = [0.461, 0.437, 0.412, 0.438, 0.360, 0.370, 0.359, 0.309]
    for i, L in enumerate(HB_LAYER):
        keep = [e for e in range(32) if 0.05 < (hot[:, i, e] > 0).mean() < 0.95]
        row = []
        for cols in (list(range(15)), EEF, JNT):
            row.append(np.median([abs(loso_auc(X[:, cols], hot[:, i, e] > 0, g) - .5)
                                  for e in keep]))
        print("  %2d    %.3f       %.3f      %.3f          %.3f"
              % (L, row[0], row[1], row[2], lib[i]))

    print("\n=== 2. 反向：路由 → 本体感受（留一指令 R²）===")
    P = loso_pred(LIN, hot.reshape(n, -1), X, g)
    r2 = 1 - ((X - P) ** 2).sum(0) / ((X - X.mean(0)) ** 2).sum(0)
    print("  维          R²        维          R²")
    for a in range(8):
        b = a + 8
        s = "  %-10s %6.3f" % (NAME[a], r2[a])
        if b < 15:
            s += "    %-10s %6.3f" % (NAME[b], r2[b])
        print(s)
    print("  末端 7 维平均 %.3f     关节角 7 维平均 %.3f"
          % (r2[EEF].mean(), r2[JNT].mean()))

    print("\n=== 3. 预算：路由方差被解释掉多少 ===")
    print("  预测子                线性      非线性    (LIBERO 5 任务: 62-77% / 86-95%)")
    for nm, cols in (("全部 15 维", list(range(15))), ("末端 7 维", EEF),
                     ("关节角 7 维", JNT)):
        a = frac(prob, loso_pred(LIN, X[:, cols], prob, g))
        b = frac(prob, loso_pred(RBF, X[:, cols], prob, g))
        print("  %-18s   %5.1f%%     %5.1f%%" % (nm, 100 * a, 100 * b))
    # control
    rng = np.random.default_rng(0)
    Xs = X.copy()
    for h in np.unique(g):
        m = np.flatnonzero(g == h)
        Xs[m] = X[rng.permutation(m)]
    print("  %-18s   %5.1f%%     %5.1f%%   <- 打乱配对的对照"
          % ("全部 15 维（打乱）", 100 * frac(prob, loso_pred(LIN, Xs, prob, g)),
             100 * frac(prob, loso_pred(RBF, Xs, prob, g))))

    print("\n=== 4. 关节角和末端的共线性（说明能不能归因）===")
    r = 1 - ((X[:, EEF] - loso_pred(LIN, X[:, JNT], X[:, EEF], g)) ** 2).sum() \
        / ((X[:, EEF] - X[:, EEF].mean(0)) ** 2).sum()
    r2b = 1 - ((X[:, JNT] - loso_pred(LIN, X[:, EEF], X[:, JNT], g)) ** 2).sum() \
        / ((X[:, JNT] - X[:, JNT].mean(0)) ** 2).sum()
    print("  用关节角线性预测末端 R² %.3f ；用末端预测关节角 R² %.3f" % (r, r2b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
