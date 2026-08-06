"""Faithful Python port of the landmark + measurement logic in MATLAB ``Avatar.m``.

The method names mirror the MATLAB ones so the two can be diffed side by side.
Call order in :meth:`MatlabAvatar.run` reproduces the ``steps == 3`` branch of the
MATLAB constructor exactly.

Known MATLAB bugs are reproduced (and flagged ``MATLAB BUG``) so that outputs
agree with the reference.  Each one also has a corrected counterpart exposed
under a ``*_fixed`` key, so callers can choose fidelity or correctness.
"""

from __future__ import annotations

import numpy as np

from .matlab_ops import (
    constrained_flood_fill,
    find_minmax,
    fix_orientation,
    get_circumference,
    get_faces,
    get_v_on_line,
    rotate_person,
    signed_volume,
    sosmooth3,
    triangle_area_sum,
)


def _kmeans2_1d(x: np.ndarray) -> np.ndarray:
    """Exact, deterministic 1-D 2-means; returns labels in {0, 1}.

    MATLAB's ``kmeans`` uses random initialisation, which makes ``adjustCrotch``
    non-deterministic in principle.  For 1-D data the globally optimal 2-means
    partition is always a single split of the sorted values, so we solve it
    exactly by scanning every split.  This is deterministic and never worse than
    what MATLAB's Lloyd iterations converge to.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 2:
        return np.zeros(n, dtype=int)

    order = np.argsort(x, kind="stable")
    xs = x[order]
    csum = np.concatenate([[0.0], np.cumsum(xs)])
    csq = np.concatenate([[0.0], np.cumsum(xs ** 2)])

    def sse(i: int, j: int) -> float:
        """Sum of squared error of xs[i:j]."""
        m = j - i
        if m <= 0:
            return 0.0
        s = csum[j] - csum[i]
        q = csq[j] - csq[i]
        return q - s * s / m

    best, best_k = np.inf, 1
    for k in range(1, n):
        val = sse(0, k) + sse(k, n)
        if val < best:
            best, best_k = val, k

    labels_sorted = np.zeros(n, dtype=int)
    labels_sorted[best_k:] = 1
    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted
    return labels


class MatlabAvatar:
    """Port of the MATLAB ``Avatar`` class (landmark detection branch)."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, orient: bool = True):
        self.f = np.asarray(faces, dtype=np.int64)
        v = np.asarray(vertices, dtype=float)
        self.v = fix_orientation(v, self.f) if orient else v.copy()

        self.landmarks: dict[str, np.ndarray] = {}
        # Every measurement slice actually used, recorded for inspection/plotting.
        self.slices: dict[str, dict] = {}
        self.measurements: dict[str, float] = {}
        self.segments: dict[str, np.ndarray] = {}
        self.notes: list[str] = []

    def _record_slice(self, name, indices, u, w, girth, plane, frame=""):
        """Store the slice used for one measurement (inspection only, no effect).

        ``u``/``w`` are the two coordinates the girth was actually computed from
        (possibly in a rotated frame); ``points3d`` are the same vertices in the
        original oriented frame, for drawing on the body.
        """
        indices = np.asarray(indices, dtype=int)
        u = np.asarray(u, dtype=float)
        w = np.asarray(w, dtype=float)
        try:
            from .matlab_ops import get_circumference
            _, hull = get_circumference(u, w)
        except Exception:
            hull = np.arange(len(u))
        self.slices[name] = {
            "indices": indices,
            "points3d": self.v[indices] if indices.size else np.zeros((0, 3)),
            "uw": np.column_stack([u, w]) if u.size else np.zeros((0, 2)),
            "hull": np.asarray(hull, dtype=int),
            "girth": float(girth),
            "plane": float(plane),
            "frame": frame,
            "n_points": int(indices.size),
        }

    # -- convenience wrappers -------------------------------------------------
    def _von(self, z_values, keep_idx, v_slice=None):
        return get_v_on_line(
            self.v if v_slice is None else v_slice,
            self.f,
            z_values,
            keep_idx,
            self.v,
        )

    @property
    def _all_idx(self) -> np.ndarray:
        return np.arange(len(self.v), dtype=np.int64)

    # ======================================================================
    # Landmarks
    # ======================================================================
    def getLegsMin(self):
        """Lowest point of each foot.  Sides split on the sign of x (not crotch)."""
        v1, v2, v3 = self.v[:, 0], self.v[:, 1], self.v[:, 2]
        m_I = int(np.argmin(v3))
        leg1 = np.array([v1[m_I], v2[m_I], v3[m_I]])

        sel = (v1 < 0) if leg1[0] > 0 else (v1 > 0)
        opposite = self.v[sel]
        I = int(np.argmin(opposite[:, 2]))
        leg2 = opposite[I].copy()

        if leg1[0] < leg2[0]:
            l_leg, r_leg = leg2, leg1
        else:
            l_leg, r_leg = leg1, leg2
        self.l_foot, self.r_foot = l_leg, r_leg
        return l_leg, r_leg

    def findMaxMin(self, left, right, num):
        """Maximum-of-the-minima sweep across vertical planes between the feet."""
        v1r, v3r = rotate_person(self.v[:, 0], self.v[:, 2], np.pi / 2)
        _, right3 = rotate_person(right[0], right[2], np.pi / 2)
        _, left3 = rotate_person(left[0], left[2], np.pi / 2)

        partition = np.linspace(left3, right3, num)
        rotated = np.column_stack([v1r, self.v[:, 1], v3r])
        pts_list, idx_list = self._von(partition, self._all_idx, v_slice=rotated)

        min_val = np.full(num, np.nan)
        min_pos = np.zeros(num, dtype=int)
        for i in range(num):
            idx = idx_list[i]
            if idx.size == 0:
                min_val[i] = np.inf   # never selected by the argmin below
                continue
            j = int(np.argmax(v1r[idx]))
            min_val[i] = v1r[idx][j]
            min_pos[i] = j

        max_index = int(np.argmin(min_val))
        p = pts_list[max_index][min_pos[max_index]]
        return float(p[0]), float(p[2])

    def getMissingY(self, x: float, z: float) -> float:
        """Median y of vertices matching (x, z) to 4 decimals."""
        rv1 = np.round(self.v[:, 0], 4)
        rv3 = np.round(self.v[:, 2], 4)
        sel = (rv1 == np.round(x, 4)) & (rv3 == np.round(z, 4))
        if not np.any(sel):
            return float("nan")
        return float(np.median(self.v[sel, 1]))

    def getCrotch(self):
        k9_1, k9_3 = self.findMaxMin(self.r_foot, self.l_foot, 50)
        k9_2 = self.getMissingY(k9_1, k9_3)
        self.crotch = np.array([k9_1, k9_2, k9_3])
        return self.crotch

    # ----------------------------------------------------------------------
    def getArmpitsAlt(self):
        """Ray-march up the outside of the body to the top of the underarm arch."""
        v1 = self.v[:, 0]
        v3 = self.v[:, 2]
        maxv1, minv1 = v1.max(), v1.min()
        maxv3, minv3 = v3.max(), v3.min()
        step = (maxv1 - minv1) * 0.1

        notop = v3 < maxv3 - (maxv3 - minv3) * 0.15
        nobottom = v3 > (maxv3 - minv3) * 0.35 + minv3
        pz = (maxv3 - minv3) * 0.35 + minv3

        side = v1 > (maxv1 - minv1) * 0.5 * 0.10
        faces_l, _ = get_faces(self.f, np.flatnonzero(side & notop & nobottom))
        left_armpit = self.getOneArmpit(faces_l, np.array([maxv1, pz]), step, "left")

        side = v1 < -(maxv1 - minv1) * 0.5 * 0.10
        faces_r, _ = get_faces(self.f, np.flatnonzero(side & notop & nobottom))
        right_armpit = self.getOneArmpit(faces_r, np.array([minv1, pz]), step, "right")

        self.r_armpit, self.l_armpit = right_armpit, left_armpit
        return right_armpit, left_armpit

    def getOneArmpit(self, faces, p, step, side):
        n = 20
        multiplier = 1.0
        angle = np.linspace(0, 7 * np.pi / 8, n)
        if side == "left":
            angle = angle[::-1]

        x = self.v[faces, 0]
        z = self.v[faces, 2]
        face_area = 0.5 * (
            -z[:, 1] * x[:, 2]
            + z[:, 0] * (-x[:, 1] + x[:, 2])
            + x[:, 0] * (z[:, 1] - z[:, 2])
            + x[:, 1] * z[:, 2]
        )
        face_area = np.where(face_area == 0, np.finfo(float).eps, face_area)

        def hits(q) -> bool:
            s = (
                z[:, 0] * x[:, 2] - x[:, 0] * z[:, 2]
                + (z[:, 2] - z[:, 0]) * q[0]
                + (x[:, 0] - x[:, 2]) * q[1]
            ) / (2 * face_area)
            t = (
                x[:, 0] * z[:, 1] - z[:, 0] * x[:, 1]
                + (z[:, 0] - z[:, 1]) * q[0]
                + (x[:, 1] - x[:, 0]) * q[1]
            ) / (2 * face_area)
            return bool(np.any((s > 0) & (t > 0) & (1 - s - t > 0)))

        p = np.array(p, dtype=float)

        # Walk inwards until we first touch the body, then back off one quarter step.
        guard = 0
        while guard < 10000:
            guard += 1
            if hits(p):
                p[0] += -step / 4 if side == "right" else step / 4
                break
            p[0] += step / 4 if side == "right" else -step / 4

        # Climb the arch.
        armpit_found = False
        outer_guard = 0
        while not armpit_found and outer_guard < 10000:
            outer_guard += 1
            i = 1                      # MATLAB 1-based angle cursor
            intersect_at = 0
            next_found = False
            inner_guard = 0
            while not next_found and inner_guard < 10000:
                inner_guard += 1
                a = angle[i - 1]
                nextp = p + np.array([np.cos(a), np.sin(a)]) * step * multiplier
                if hits(nextp):
                    intersect_at = i
                    i += 1
                else:
                    if intersect_at == 0:
                        if i > 1:
                            i -= 1
                        else:
                            multiplier *= 0.8
                    else:
                        next_found = True
                        multiplier *= 0.9
                        if 0 < i <= n:
                            a_prev = angle[i - 2]
                            p = p + np.array([np.cos(a_prev), np.sin(a_prev)]) * step * multiplier
                if i == n - 1:
                    multiplier *= 0.5
                    intersect_at = 0
                    i = 1
                if multiplier <= 0.5 ** 5:
                    next_found = True
                    armpit_found = True

        # MATLAB returns y as NaN here by design; it is filled in later only by
        # the (unused) getArmpits variant.
        return np.array([p[0], np.nan, p[1]])

    # ----------------------------------------------------------------------
    def armSearch(self, side: str) -> np.ndarray:
        """Constrained flood fill from the fingertip inwards to the armpit plane."""
        if side == "r":
            hand_idx = int(np.argmin(self.v[:, 0]))
            limit = self.r_armpit[0]
            keep = lambda idx: self.v[idx, 0] < limit
        else:
            hand_idx = int(np.argmax(self.v[:, 0]))
            limit = self.l_armpit[0]
            keep = lambda idx: self.v[idx, 0] > limit

        seed = np.flatnonzero(np.any(self.f == hand_idx, axis=1))
        return constrained_flood_fill(self.f, seed, keep)

    def trunkSearch(self) -> np.ndarray:
        z_span = self.lShoulder[2] - self.crotch[2]
        z_mid = (self.crotch[2] + self.lShoulder[2]) / 2
        z_max, z_min = z_mid + 0.1 * z_span, z_mid - 0.1 * z_span
        x_span = self.lShoulder[0] - self.rShoulder[0]
        x_mid = (self.rShoulder[0] + self.lShoulder[0]) / 2
        x_max, x_min = x_mid + 0.1 * x_span, x_mid - 0.1 * x_span

        cand = np.flatnonzero(
            (self.v[:, 2] < z_max)
            & (self.v[:, 2] > z_min)
            & (self.v[:, 0] < x_max)
            & (self.v[:, 0] > x_min)
        )
        if cand.size == 0:
            return np.empty(0, dtype=np.int64)
        one_idx = int(cand[0])

        shoulder_z = max(self.lShoulder[2], self.rShoulder[2])
        keep = lambda idx: (self.v[idx, 2] < shoulder_z) & (self.v[idx, 2] > self.crotch[2])

        seed = np.flatnonzero(np.any(self.f == one_idx, axis=1))
        v_idx = constrained_flood_fill(self.f, seed, keep)

        drop = np.concatenate([self.rArmIdx, self.lArmIdx, self.legIdx])
        return np.setdiff1d(v_idx, drop)

    def getLegs(self) -> np.ndarray:
        """Everything below the crotch-to-hip lines in the (x, z) plane.

        Purely geometric -- no connectivity.  This is what keeps the two legs
        separable on a coarse mesh where they touch and a flood fill would merge
        them into one component.
        """
        r1 = np.array([self.crotch[0], self.crotch[2]])
        v1, v3 = self.v[:, 0], self.v[:, 2]

        h1 = np.array([self.r_hip[0], self.r_hip[2]])
        slope_r = (r1[1] - h1[1]) / (r1[0] - h1[0])
        inter_r = r1[1] - slope_r * r1[0]
        neg_r = np.flatnonzero(v3 - slope_r * v1 - inter_r <= 0)

        h1 = np.array([self.l_hip[0], self.l_hip[2]])
        slope_l = (r1[1] - h1[1]) / (r1[0] - h1[0])
        inter_l = r1[1] - slope_l * r1[0]
        neg_l = np.flatnonzero(v3 - slope_l * v1 - inter_l <= 0)

        leg_idx = np.union1d(neg_l, neg_r)
        arm_idx = np.concatenate([self.lArmIdx, self.rArmIdx])
        return np.setdiff1d(leg_idx, arm_idx)

    def getHead(self) -> np.ndarray:
        max_shoulder_z = max(self.rShoulder[2], self.lShoulder[2])
        head_idx = np.flatnonzero(self.v[:, 2] > max_shoulder_z)
        arm_idx = np.concatenate([self.lArmIdx, self.rArmIdx])
        return np.setdiff1d(head_idx, arm_idx)

    def getShoulders(self):
        l_arm = self.v[self.lArmIdx]
        self.lShoulder = l_arm[int(np.argmax(l_arm[:, 2]))].copy()
        r_arm = self.v[self.rArmIdx]
        self.rShoulder = r_arm[int(np.argmax(r_arm[:, 2]))].copy()
        return self.lShoulder, self.rShoulder

    # ----------------------------------------------------------------------
    def adjustCrotch(self) -> np.ndarray:
        """Raise the crotch z to where the inter-leg notch stops being distinct."""
        k9_adj = self.crotch.copy()
        N = 20
        z_points = np.linspace(self.crotch[2], min(self.r_armpit[2], self.l_armpit[2]), N)
        delta_v2 = np.zeros(N)
        cnd_vector = np.ones(N)

        arm_idx = np.concatenate([self.lArmIdx, self.rArmIdx])
        no_arm_idx = np.setdiff1d(self._all_idx, arm_idx)

        for i in range(N):
            pts, _ = self._von(z_points[i], no_arm_idx)
            if len(pts) < 3:
                continue
            v1, v2 = pts[:, 0], pts[:, 1]
            _, k = get_circumference(v1, v2)
            v1_cnvh, v2_cnvh = v1[k], v2[k]
            mean_v2 = v2.mean()

            sel = v2 < mean_v2
            v2_bot, v1_bot = v2[sel], v1[sel]
            selh = v2_cnvh < mean_v2
            v2h_bot, v1h_bot = v2_cnvh[selh], v1_cnvh[selh]
            if v1_bot.size == 0:
                continue

            qtr_l = v1_bot.mean() + (v1_bot.max() - v1_bot.mean()) / 2
            qtr_s = v1_bot.mean() + (v1_bot.min() - v1_bot.mean()) / 2
            mid_sel = (v1_bot > qtr_s) & (v1_bot < qtr_l)
            v1_bot, v2_bot = v1_bot[mid_sel], v2_bot[mid_sel]
            if v1_bot.size == 0:
                continue

            idx_v2_bot = int(np.argmax(v2_bot))
            mx_v_bot_1 = v1_bot[idx_v2_bot]
            idx_mx_cvh = (v1h_bot == mx_v_bot_1)

            if i == 0:
                continue
            if np.sum(idx_mx_cvh) > 0:
                cnd_vector[i] = 0
                continue

            mid_v1 = (v1_bot.max() + v1_bot.min()) / 2
            r_idx = v1h_bot > mid_v1
            l_idx = v1h_bot < mid_v1

            v2h_botR = v2h_bot[r_idx].min() if np.any(r_idx) else None
            v1h_botR = v1h_bot[r_idx][int(np.argmin(v2h_bot[r_idx]))] if np.any(r_idx) else None
            v2h_botL = v2h_bot[l_idx].min() if np.any(l_idx) else None
            v1h_botL = v1h_bot[l_idx][int(np.argmin(v2h_bot[l_idx]))] if np.any(l_idx) else None

            if v2h_botR is None and v2h_botL is None:
                continue
            if v2h_botR is None:
                mid_v2h, mid_v1h = v2h_botL, v1h_botL
            elif v2h_botL is None:
                mid_v2h, mid_v1h = v2h_botR, v1h_botR
            else:
                mid_v2h = (v2h_botR + v2h_botL) / 2
                mid_v1h = (v1h_botR + v1h_botL) / 2

            idx_mid = int(np.argmin(np.abs(v1_bot - mid_v1h)))
            mid_v2_bot = v2_bot[idx_mid]
            org_idx_mid = idx_mid
            org_v2_bot = v2_bot.copy()

            # Trim from the tail, then the head, while the middle stays the lower.
            work = v2_bot.copy()
            while work.size and mid_v2_bot <= work[-1]:
                work = work[:-1]
            if work.size:
                while work.size and mid_v2_bot <= work[0]:
                    work = work[1:]
                    idx_mid -= 1

            if work.size == 0:
                delta_v2[i] = 0
                continue

            mid_max_v2_bot = work.max()
            match = v1_bot[org_v2_bot == mid_max_v2_bot]
            x_mid_max = match[0] if match.size else mid_v1h
            x_v1_bot = v1_bot[org_idx_mid]

            if abs(x_mid_max) > abs(x_v1_bot):
                delta_v2[i] = mid_v2_bot - mid_v2h
            else:
                delta_v2[i] = mid_max_v2_bot - mid_v2h

        s1 = _kmeans2_1d(delta_v2)

        i = 0                                   # MATLAB i = 1
        cnd = cnd_vector[i]
        while i < N - 1 and cnd:
            i += 1
            cnd = cnd_vector[i] * (1.0 if s1[i] != s1[0] else 0.0)
        i -= 1
        k9_adj[2] = z_points[max(i, 0)]
        self.crotch = k9_adj
        return k9_adj

    def getCollar(self):
        self.collar = (self.lShoulder + self.rShoulder) / 2
        return self.collar

    # ======================================================================
    # Circumferences
    # ======================================================================
    def getWrist(self):
        r_hand = self.v[int(np.argmin(self.v[:, 0]))]
        l_hand = self.v[int(np.argmax(self.v[:, 0]))]

        hyp = np.linalg.norm(
            np.array([self.lShoulder[0], self.lShoulder[2]]) - np.array([l_hand[0], l_hand[2]])
        )
        theta_l = np.arccos(np.clip((self.lShoulder[2] - l_hand[2]) / hyp, -1, 1))
        hyp = np.linalg.norm(
            np.array([self.rShoulder[0], self.rShoulder[2]]) - np.array([r_hand[0], r_hand[2]])
        )
        theta_r = np.arccos(np.clip((self.rShoulder[2] - r_hand[2]) / hyp, -1, 1))

        out = {}
        for side, theta, arm_idx in (
            ("r", theta_r, self.rArmIdx),
            ("l", -theta_l, self.lArmIdx),
        ):
            x, z = rotate_person(self.v[:, 0], self.v[:, 2], theta)
            y = self.v[:, 1]
            rot = np.column_stack([x, y, z])

            z_start = z[arm_idx].min()
            z_end = (2 * z_start + z[arm_idx].max()) / 3
            z_start = (3 * z_start + z_end) / 4

            n = 20
            z_values = np.linspace(z_start, z_end, n)
            pts_list, idx_list = self._von(z_values, arm_idx, v_slice=rot)

            maj_axis = np.zeros(n)
            for i in range(n):
                idx = idx_list[i]
                if idx.size < 2:
                    maj_axis[i] = np.inf
                    continue
                pts2 = np.column_stack([x[idx], y[idx]])
                d = np.linalg.norm(pts2[:, None, :] - pts2[None, :, :], axis=2)
                maj_axis[i] = d.max()

            i = int(np.argmin(maj_axis))
            idx = idx_list[i]
            girth, b = get_circumference(x[idx], y[idx])
            points = pts_list[i][b]
            wrist = points.mean(axis=0)
            self._record_slice(f"{side}Wrist", idx, x[idx], y[idx], girth,
                               z_values[i], "arm-aligned")
            out[side] = (wrist, girth, points)

        (self.r_wrist, r_girth, _), (self.l_wrist, l_girth, _) = out["r"], out["l"]
        self.measurements["rWristGirth"] = r_girth
        self.measurements["lWristGirth"] = l_girth
        return self.r_wrist, self.l_wrist

    def slice_n_dice(self, n1, n2, z_start, z_end, keep_idx):
        z_points, j = None, 1
        for n in (n1, n2):
            z_points = np.linspace(z_start, z_end, n + 1)
            dist = np.zeros(n)
            for i in range(n):
                pts, _ = self._von((z_points[i] + z_points[i + 1]) / 2, keep_idx)
                dist[i] = pts[:, 1].max() if pts.size else -np.inf
            if n == n2:
                dist = sosmooth3(dist, 7)

            j = 1                              # MATLAB j = 1, incremented first
            cnd = True
            while cnd:
                j += 1
                cnd1 = j <= n - 1
                cnd2 = dist[j - 2] < dist[j - 1]     # dist(j-1) < dist(j), 1-based
                cnd = bool(cnd1 and cnd2)
            z_start = z_points[j - 2]
            z_end = z_points[j - 1]

        zc = (z_points[j - 2] + z_points[j - 1]) / 2
        pts, idx = self._von(zc, keep_idx)
        self._hip_idx = idx
        return pts[:, 0], pts[:, 1], pts[:, 2]

    def getHip(self):
        arm_idx = np.concatenate([self.lArmIdx, self.rArmIdx])
        keep_idx = np.setdiff1d(self._all_idx, arm_idx)

        z_start = self.crotch[2]
        z_end = (self.r_armpit[2] + self.crotch[2]) / 2
        x, y, z = self.slice_n_dice(3, 10, z_start, z_end, keep_idx)

        x_min_idx = int(np.argmin(x))
        x_max_idx = int(np.argmax(x))
        back = y > 0
        if np.any(back):
            back_x, back_y, back_z = x[back], y[back], z[back]
            k = int(np.argmin(np.abs(np.abs(back_x) - abs(self.crotch[0]))))
            back_point = np.array([back_x[k], back_y[k], back_z[k]])
        else:
            back_point = np.array([x[x_min_idx], y[x_min_idx], z[x_min_idx]])

        mn_z = float(z.mean())
        self.r_hip = np.array([x[x_min_idx], y[x_min_idx], mn_z])
        self.l_hip = np.array([x[x_max_idx], y[x_max_idx], mn_z])
        self.lowerBack = back_point

        circ, _ = get_circumference(x, y)
        self._record_slice("hip", self._hip_idx, x, y, circ, mn_z)
        self.measurements["hipGirth"] = circ
        return circ

    def getWaist(self):
        z_mid = np.mean([self.r_armpit[2], self.r_hip[2]])
        pts, idx = self._von(z_mid, self.trunkIdx)
        circ, _ = get_circumference(pts[:, 0], pts[:, 1])
        self._record_slice("waist", idx, pts[:, 0], pts[:, 1], circ, z_mid)
        self.measurements["waistGirth"] = circ
        return circ

    def getChestCircumference(self):
        z_value = np.median([self.r_armpit[2], self.l_armpit[2]])
        pts, idx = self._von(z_value, self.trunkIdx)
        circ, _ = get_circumference(pts[:, 0], pts[:, 1])
        self._record_slice("chest", idx, pts[:, 0], pts[:, 1], circ, z_value)
        self.measurements["chestGirth"] = circ
        return circ

    def getThighGirth(self):
        v1 = self.v[:, 0]
        r_leg = self.legIdx[v1[self.legIdx] < self.crotch[0]]
        l_leg = self.legIdx[v1[self.legIdx] >= self.crotch[0]]

        z = 0.75 * (self.r_hip[2] - self.r_ankle[2]) + self.r_ankle[2]
        pts, idx = self._von(z, r_leg)
        circ, b = get_circumference(pts[:, 0], pts[:, 1])
        self._record_slice("rThigh", idx, pts[:, 0], pts[:, 1], circ, z)
        self.measurements["rThighGirth"] = circ
        self.rThighPoints = pts[b] if len(pts) else np.zeros((0, 3))

        z = 0.75 * (self.l_hip[2] - self.l_ankle[2]) + self.l_ankle[2]
        pts, idx = self._von(z, l_leg)
        circ, b = get_circumference(pts[:, 0], pts[:, 1])
        self._record_slice("lThigh", idx, pts[:, 0], pts[:, 1], circ, z)
        self.measurements["lThighGirth"] = circ
        self.lThighPoints = pts[b] if len(pts) else np.zeros((0, 3))

    def getAnkleGirth(self):
        n = 20
        for side in ("r", "l"):
            if side == "r":
                z_start = self.v[self.v[:, 0] < 0, 2].min()
                leg_idx = np.intersect1d(np.flatnonzero(self.v[:, 0] < self.crotch[0]), self.legIdx)
            else:
                z_start = self.v[self.v[:, 0] > 0, 2].min()
                leg_idx = np.intersect1d(np.flatnonzero(self.v[:, 0] > self.crotch[0]), self.legIdx)
            z_end = (3 * z_start + self.crotch[2]) / 4
            z_start = (7 * z_start + z_end) / 8

            z_values = np.linspace(z_start, z_end, n)
            pts_list, _ = self._von(z_values, leg_idx)
            circ = np.array([get_circumference(p[:, 0], p[:, 1])[0] if len(p) else np.inf
                             for p in pts_list])
            i = int(np.argmin(circ))
            value, b = get_circumference(pts_list[i][:, 0], pts_list[i][:, 1])
            points = pts_list[i][b]
            ankle = points.mean(axis=0)
            _, aidx = self._von(z_values[i], leg_idx)
            self._record_slice(f"{side}Ankle", aidx, pts_list[i][:, 0],
                               pts_list[i][:, 1], value, z_values[i])
            # MATLAB BUG: getAnkleGirth returns [lAnkle, rAnkle, lAnkleGirth,
            # rAnkleGirth, ...] but the constructor assigns output slots 3 and 4
            # to [self.r_ankle_girth, self.l_ankle_girth] -- so the two girths
            # land on the wrong sides.  The ankle *landmarks* are not swapped.
            # Reproduced here; corrected values kept under the *_fixed keys.
            if side == "r":
                self.r_ankle = ankle
                self.measurements["lAnkleGirth"] = value
                self.measurements["rAnkleGirth_fixed"] = value
            else:
                self.l_ankle = ankle
                self.measurements["rAnkleGirth"] = value
                self.measurements["lAnkleGirth_fixed"] = value

    def _calf_girth(self, z_start, z_end, leg_val):
        diff = z_end - z_start
        z_start = 0.15 * diff + z_start
        z_end = z_end - 0.5 * diff

        if leg_val == 0:      # left leg lives at x > 0
            p1 = np.array([self.l_ankle[0], self.l_ankle[2]])
            p2 = np.array([(self.crotch[0] + self.l_hip[0]) / 2,
                           (self.crotch[2] + self.l_hip[2]) / 2])
            sign = -1.0
            idx = np.flatnonzero(self.v[:, 0] > 0)
        else:
            p1 = np.array([self.r_ankle[0], self.r_ankle[2]])
            p2 = np.array([(self.crotch[0] + self.r_hip[0]) / 2,
                           (self.crotch[2] + self.r_hip[2]) / 2])
            sign = 1.0
            idx = np.flatnonzero(self.v[:, 0] < 0)

        hyp = np.linalg.norm(p1 - p2)
        theta = np.arccos(np.clip((p2[1] - p1[1]) / hyp, -1, 1))
        angle = sign * theta

        x, z = rotate_person(self.v[:, 0], self.v[:, 2], angle)
        y = self.v[:, 1]
        rot = np.column_stack([x, y, z])

        n = 20
        intervals = np.linspace(z_start, z_end, n + 1)
        z_points = (intervals[:n] + intervals[1:]) / 2

        circ = np.zeros(n)
        _, vidx = self._von(z_points[0], idx, v_slice=rot)
        circ[0] = get_circumference(x[vidx], y[vidx])[0]
        i = 1                                    # MATLAB 1-based loop cursor
        for i in range(1, n + 1):
            if i == n:
                break
            _, vidx = self._von(z_points[i], idx, v_slice=rot)
            circ[i] = get_circumference(x[vidx], y[vidx])[0]
            if circ[i - 1] > circ[i]:
                break

        if i < 2 or i > n - 1:
            return None, 0.0, angle       # caller falls back to calfGirthOther

        m = 20
        z_points2 = np.linspace(intervals[i - 1], intervals[i], m)
        pts_list, idx_list = self._von(z_points2, idx, v_slice=rot)
        circ2 = np.array([get_circumference(x[ix], y[ix])[0] for ix in idx_list])
        g = int(np.argmax(circ2))
        value, b = get_circumference(x[idx_list[g]], y[idx_list[g]])
        name = "lCalf" if leg_val == 0 else "rCalf"
        self._record_slice(name, idx_list[g], x[idx_list[g]], y[idx_list[g]],
                           value, z_points2[g], "shank-aligned")
        return value, z_points2[g], angle

    def _calf_girth_other(self, z_value, angle, leg_val):
        x, z = rotate_person(self.v[:, 0], self.v[:, 2], angle)
        y = self.v[:, 1]
        rot = np.column_stack([x, y, z])
        if leg_val == 0:
            idx = np.flatnonzero(self.v[:, 0] > 0)
            z_start, z_end = self.l_ankle[2], self.crotch[2]
        else:
            idx = np.flatnonzero(self.v[:, 0] < 0)
            z_start, z_end = self.r_ankle[2], self.crotch[2]
        if z_value == 0:
            z_value = 0.25 * (z_end - z_start) + z_start
        _, vidx = self._von(z_value, idx, v_slice=rot)
        return get_circumference(x[vidx], y[vidx])[0]

    def getCalf(self):
        l_val, l_z, l_angle = self._calf_girth(self.l_ankle[2], self.crotch[2], 0)
        r_val, r_z, r_angle = self._calf_girth(self.r_ankle[2], self.crotch[2], 1)

        if l_val is None and r_val is None:
            l_val = self._calf_girth_other(0, l_angle, 0)
            r_val = self._calf_girth_other(0, r_angle, 1)
        elif l_val is None:
            l_val = self._calf_girth_other(r_z, l_angle, 0)
        elif r_val is None:
            r_val = self._calf_girth_other(l_z, r_angle, 1)

        self.measurements["lCalfGirth"] = l_val
        self.measurements["rCalfGirth"] = r_val

    def getArmGirth(self):
        # MATLAB BUG: B and the right-arm terms reference armMaxL(2) where
        # armMaxR(2) is meant.  Reproduced; see *_fixed note in run().
        A = np.array([0.0, self.r_wrist[2] - self.armMaxR[1]])
        B = np.array([self.r_wrist[0] - self.armMaxR[0], self.r_wrist[2] - self.armMaxL[1]])
        C = np.array([0.0, self.l_wrist[2] - self.armMaxL[1]])
        D = np.array([self.l_wrist[0] - self.armMaxL[0], self.l_wrist[2] - self.armMaxL[1]])

        r_theta = np.arccos(np.clip(A.dot(B) / (np.linalg.norm(A) * np.linalg.norm(B)), -1, 1))
        l_theta = np.arccos(np.clip(C.dot(D) / (np.linalg.norm(C) * np.linalg.norm(D)), -1, 1))

        specs = [
            ("rForearmGirth", self.r_wrist, self.armMaxR, 3, r_theta, self.rArmIdx),
            ("lForearmGirth", self.l_wrist, self.armMaxL, 3, -l_theta, self.lArmIdx),
            ("rBicepGirth", self.r_wrist, self.armMaxR, 1, r_theta, self.rArmIdx),
            ("lBicepGirth", self.l_wrist, self.armMaxL, 1, -l_theta, self.lArmIdx),
        ]
        for name, wrist, arm_max, w, theta, arm_idx in specs:
            target = np.array([(w * wrist[0] + (4 - w) * arm_max[0]) / 4,
                               (w * wrist[2] + (4 - w) * arm_max[1]) / 4])
            _, rot_z = rotate_person(target[0], target[1], theta)
            x, z = rotate_person(self.v[:, 0], self.v[:, 2], theta)
            rot = np.column_stack([x, self.v[:, 1], z])
            _, vidx = self._von(rot_z, arm_idx, v_slice=rot)
            value = get_circumference(rot[vidx, 0], rot[vidx, 1])[0]
            self._record_slice(name.replace("Girth", ""), vidx,
                               rot[vidx, 0], rot[vidx, 1], value, rot_z,
                               "arm-aligned")
            self.measurements[name] = value

    # ======================================================================
    # Lengths
    # ======================================================================
    def getArmLength(self):
        self.armMaxR = np.array([(self.r_armpit[0] + self.rShoulder[0]) / 2,
                                 (self.r_armpit[2] + self.rShoulder[2]) / 2])
        self.armMaxL = np.array([(self.l_armpit[0] + self.lShoulder[0]) / 2,
                                 (self.l_armpit[2] + self.lShoulder[2]) / 2])
        self.measurements["rArmLength"] = float(np.hypot(
            self.armMaxR[0] - self.r_wrist[0], self.armMaxR[1] - self.r_wrist[2]))
        self.measurements["lArmLength"] = float(np.hypot(
            self.armMaxL[0] - self.l_wrist[0], self.armMaxL[1] - self.l_wrist[2]))

    def getCollarScalpLength(self):
        head_i = int(np.argmax(self.v[:, 2]))
        head_z = self.v[head_i, 2]
        head_x = self.v[head_i, 0]
        # MATLAB BUG: collar(1,2) is the Y component, but it is compared against
        # a Z coordinate.  On this scan that inflates the result to ~97% of
        # stature.  Faithful value kept; corrected value exposed alongside.
        self.measurements["collarScalpLength"] = float(
            np.hypot(head_x - self.collar[0], head_z - self.collar[1]))
        self.measurements["collarScalpLength_fixed"] = float(
            np.hypot(head_x - self.collar[0], head_z - self.collar[2]))

    def getTrunkLength(self):
        # MATLAB BUG: same Y-for-Z confusion as getCollarScalpLength.
        self.measurements["trunkLength"] = float(
            np.hypot(self.crotch[0] - self.collar[0], self.crotch[2] - self.collar[1]))
        self.measurements["trunkLength_fixed"] = float(
            np.hypot(self.crotch[0] - self.collar[0], self.crotch[2] - self.collar[2]))

    def getLegLength(self):
        V = self.v
        # MATLAB QUIRK: here "right" is x > 0, the opposite of every other
        # routine in the file (r_hip, r_foot and friends all use x < 0).
        r_new = V[V[:, 0] > 0]
        l_new = V[V[:, 0] < 0]
        r_min_id = int(np.argmin(r_new[:, 2]))
        l_min_id = int(np.argmin(l_new[:, 2]))
        r_min_z, l_min_z = r_new[r_min_id, 2], l_new[l_min_id, 2]
        r_min_x, l_min_x = r_new[r_min_id, 0], l_new[l_min_id, 0]

        c_min_x = (r_min_x - l_min_x) / 2
        delta_x = r_min_x - c_min_x
        l_delta_z = self.l_hip[2] - l_min_z
        r_delta_z = self.r_hip[2] - r_min_z
        self.measurements["lLegLength"] = float(np.hypot(delta_x, l_delta_z))
        self.measurements["rLegLength"] = float(np.hypot(delta_x, r_delta_z))

    def getCrotchHeight(self):
        self.measurements["crotchHeight"] = float(self.crotch[2] - self.v[:, 2].min())

    def getNoseTip(self):
        """Most anterior point in the 30-60% band between shoulder and scalp."""
        above = self.v[self.v[:, 2] > max(self.rShoulder[2], self.lShoulder[2])]
        if above.size == 0:
            self.nose_tip = np.array([np.nan] * 3)
            return self.nose_tip
        val_low = above[:, 2].min()
        dist = above[:, 2].max() - val_low
        tmp = above[above[:, 2] > val_low + dist * 0.3]
        tmp = tmp[tmp[:, 2] < val_low + 0.6 * dist]
        if tmp.size == 0:
            self.nose_tip = np.array([np.nan] * 3)
            return self.nose_tip
        i = int(np.argmin(tmp[:, 1]))
        self.nose_tip = np.array([tmp[i, 0], tmp[i, 1], tmp[i, 2]])
        return self.nose_tip

    # ======================================================================
    # Areas and volume
    # ======================================================================
    def getSurfaceArea(self):
        self.measurements["SA_total"] = triangle_area_sum(self.v, self.f)
        self.measurements["VOL_total"] = signed_volume(self.v, self.f)

        def partial(indices, isleg=0):
            if isleg == 1:
                indices = indices[self.v[indices, 0] > self.crotch[0]]
            elif isleg == 2:
                indices = indices[self.v[indices, 0] <= self.crotch[0]]
            faces, _ = get_faces(self.f, indices)
            return triangle_area_sum(self.v, faces)

        self.measurements["SA_trunk"] = partial(self.trunkIdx)
        self.measurements["SA_lleg"] = partial(self.legIdx, 1)
        self.measurements["SA_rleg"] = partial(self.legIdx, 2)
        self.measurements["SA_legs"] = partial(self.legIdx, 3)
        self.measurements["SA_head"] = partial(self.headIdx)
        self.measurements["SA_rArm"] = partial(self.rArmIdx)
        self.measurements["SA_lArm"] = partial(self.lArmIdx)

    # ======================================================================
    # Driver -- mirrors the MATLAB constructor's steps==3 branch, in order
    # ======================================================================
    def run(self) -> "MatlabAvatar":
        self.getLegsMin()
        self.getCrotch()
        self.getArmpitsAlt()

        self.rArmIdx = self.armSearch("r")
        self.lArmIdx = self.armSearch("l")

        self.getShoulders()
        self.adjustCrotch()
        self.getCollar()
        self.getWrist()
        self.getHip()

        self.legIdx = self.getLegs()
        self.headIdx = self.getHead()
        self.trunkIdx = self.trunkSearch()

        self.getWaist()
        self.getArmLength()
        self.getCollarScalpLength()
        self.getTrunkLength()
        self.getAnkleGirth()
        self.getLegLength()
        self.getCalf()
        self.getThighGirth()
        self.getCrotchHeight()
        self.getArmGirth()
        self.getChestCircumference()
        self.getNoseTip()
        self.getSurfaceArea()

        self.measurements["height"] = float(self.v[:, 2].max() - self.v[:, 2].min())

        self.landmarks = {
            "r_wrist": self.r_wrist, "l_wrist": self.l_wrist,
            "r_armpit": self.r_armpit, "l_armpit": self.l_armpit,
            "r_hip": self.r_hip, "l_hip": self.l_hip,
            "r_foot": self.r_foot, "l_foot": self.l_foot,
            "crotch": self.crotch, "l_ankle": self.l_ankle, "r_ankle": self.r_ankle,
            "lShoulder": self.lShoulder, "rShoulder": self.rShoulder,
            "collar": self.collar, "lowerBack": self.lowerBack,
            "nose_tip": self.nose_tip,
        }
        self.segments = {
            "left_arm": self.lArmIdx, "right_arm": self.rArmIdx,
            "legs": self.legIdx, "head": self.headIdx, "trunk": self.trunkIdx,
            "left_leg": self.legIdx[self.v[self.legIdx, 0] >= self.crotch[0]],
            "right_leg": self.legIdx[self.v[self.legIdx, 0] < self.crotch[0]],
        }
        return self
