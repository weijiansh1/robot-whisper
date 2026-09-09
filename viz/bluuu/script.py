from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import *


ROOT = Path(__file__).resolve().parents[2]
REPLAY = json.loads((ROOT / "rerun-2026-08-27/replay.json").read_text())
EARLY = json.loads(
    (
        ROOT
        / "VLA_MUI_HUB/moe-token-dynamics/results/early_seedblock_summary.json"
    ).read_text()
)
FAILURE_MODES = json.loads(
    (ROOT / "rerun-2026-08-27/failure_modes.json").read_text()
)

BG = "#0D1117"
SUCCESS = "#69D18B"
STASIS = "#FF6B6B"
EVENT = "#FFB84D"
MOE = "#58C4DD"
CURRENT = "#FFE66D"
STRUCTURE = "#7D8590"
WHITE_SOFT = "#E6EDF3"
FONT = "Liberation Mono"

TITLE_SIZE = 42
HEADING_SIZE = 30
BODY_SIZE = 24
LABEL_SIZE = 19


def make_text(
    value: str,
    size: int = BODY_SIZE,
    color: str = WHITE_SOFT,
    weight: str = NORMAL,
    max_width: float | None = None,
) -> Text:
    text = Text(value, font=FONT, font_size=size, color=color, weight=weight)
    if max_width is not None and text.width > max_width:
        text.set_width(max_width)
    return text


def get_rollout(episode: int) -> dict:
    return next(row for row in REPLAY["rollouts"] if row["episode"] == episode)


def active_experts(episode: int, chunk: int) -> list[int]:
    rollout = get_rollout(episode)
    return [expert for expert, row in enumerate(rollout["route"]) if row[chunk]]


def expert_grid(active: list[int], color: str, center: np.ndarray) -> VGroup:
    cells = VGroup()
    active_set = set(active)
    for index in range(32):
        selected = index in active_set
        cell = Square(
            side_length=0.34,
            stroke_color=color if selected else STRUCTURE,
            stroke_width=1.4 if selected else 0.6,
            fill_color=color if selected else STRUCTURE,
            fill_opacity=0.9 if selected else 0.08,
        )
        cells.add(cell)
    cells.arrange_in_grid(rows=4, cols=8, buff=0.11)
    cells.move_to(center)
    return cells


def smooth_path(points: list[np.ndarray], color: str, width: float = 5) -> VMobject:
    path = VMobject(stroke_color=color, stroke_width=width)
    path.set_points_smoothly(points)
    return path


class StoryScene(Scene):
    def setup(self) -> None:
        self.camera.background_color = BG

    def title(self, value: str) -> Text:
        title = make_text(value, TITLE_SIZE, WHITE_SOFT, BOLD, 12.5)
        title.to_edge(UP, buff=0.55)
        return title

    def clean_exit(self) -> None:
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)
        self.wait(0.3)


class Scene1_StateBranches(StoryScene):
    def construct(self) -> None:
        title = self.title("Same state. Different futures.")
        subtitle = make_text(
            "abstract policy-environment state space",
            LABEL_SIZE,
            STRUCTURE,
        ).to_edge(DOWN, buff=0.55)
        self.add_subcaption(
            "The same initial state can lead to success, stasis, or an event-like failure.",
            duration=4,
        )
        self.play(Write(title), FadeIn(subtitle), run_time=1.6)
        self.wait(1.2)

        start = LEFT * 5.3 + DOWN * 0.1
        start_ring = Circle(radius=0.27, color=WHITE_SOFT, stroke_width=3).move_to(start)
        start_dot = Dot(start, radius=0.08, color=WHITE_SOFT)
        start_label = make_text("same snapshot", LABEL_SIZE, WHITE_SOFT).next_to(
            start_ring, DOWN, buff=0.35
        )
        self.add_subcaption("All three trajectories begin at one snapshot.", duration=3)
        self.play(GrowFromCenter(start_ring), FadeIn(start_dot), FadeIn(start_label), run_time=1.2)
        self.wait(1.0)

        goal = RIGHT * 5.0 + UP * 1.45
        goal_ring = Circle(radius=0.42, color=SUCCESS, stroke_width=4).move_to(goal)
        goal_ring.add(Circle(radius=0.17, color=SUCCESS, fill_opacity=0.9).move_to(goal))
        basin_center = RIGHT * 1.9 + DOWN * 1.0
        basin = VGroup(
            *[
                Circle(radius=radius, color=STASIS, stroke_opacity=opacity)
                .stretch(1.6, 0)
                .move_to(basin_center)
                for radius, opacity in [(0.45, 0.55), (0.8, 0.28), (1.15, 0.12)]
            ]
        )

        success_path = CubicBezier(
            start,
            LEFT * 2.6 + UP * 1.8,
            RIGHT * 2.2 + UP * 1.7,
            goal,
        ).set_stroke(SUCCESS, width=5)

        prefix = [
            start,
            LEFT * 3.6 + DOWN * 0.6,
            LEFT * 1.6 + DOWN * 0.75,
            RIGHT * 0.4 + DOWN * 0.95,
        ]
        theta = np.linspace(np.pi, 3.8 * np.pi, 26)
        radius = np.linspace(1.2, 0.12, len(theta))
        spiral = [
            basin_center
            + np.array([1.2 * r * np.cos(a), 0.55 * r * np.sin(a), 0.0])
            for a, r in zip(theta, radius)
        ]
        stasis_path = smooth_path(prefix + spiral, STASIS)

        event_break = RIGHT * 2.9 + UP * 0.65
        event_path_a = CubicBezier(
            start,
            LEFT * 2.5 + UP * 0.4,
            RIGHT * 0.5 + UP * 0.85,
            event_break,
        ).set_stroke(EVENT, width=5)
        event_path_b = Line(
            event_break,
            RIGHT * 4.65 + DOWN * 1.9,
            color=EVENT,
            stroke_width=5,
        )

        success_dot = Dot(start, radius=0.1, color=SUCCESS)
        stasis_dot = Dot(start, radius=0.1, color=STASIS)
        event_dot = Dot(start, radius=0.1, color=EVENT)

        self.add_subcaption(
            "Success keeps making progress. Stasis loses progress. Event failures break abruptly.",
            duration=7,
        )
        self.play(
            Create(goal_ring),
            FadeIn(basin),
            Create(success_path),
            Create(stasis_path),
            Create(event_path_a),
            Create(event_path_b),
            run_time=2.2,
        )
        self.play(
            MoveAlongPath(success_dot, success_path),
            MoveAlongPath(stasis_dot, stasis_path),
            Succession(
                MoveAlongPath(event_dot, event_path_a),
                MoveAlongPath(event_dot, event_path_b),
            ),
            run_time=4.2,
            rate_func=linear,
        )
        self.wait(1.2)

        n_fail = sum(task["n_fail"] for task in FAILURE_MODES.values())
        n_stasis = sum(
            sum(task["stasis"].values()) for task in FAILURE_MODES.values()
        )
        counts = Group(
            make_text(f"{2048 - n_fail} success", LABEL_SIZE, SUCCESS, BOLD),
            make_text(f"{n_stasis} stasis", LABEL_SIZE, STASIS, BOLD),
            make_text(f"{n_fail - n_stasis} event-like", LABEL_SIZE, EVENT, BOLD),
        ).arrange(RIGHT, buff=0.9)
        counts.move_to(DOWN * 2.75)
        self.add_subcaption(
            "Across four tasks, stasis accounts for 275 of 307 failed rollouts.",
            duration=4,
        )
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.15) for item in counts], lag_ratio=0.2), run_time=1.7)
        self.wait(1.4)

        success_group = Group(success_path, success_dot, goal_ring)
        event_group = Group(event_path_a, event_path_b, event_dot)
        focus = SurroundingRectangle(basin, color=STASIS, buff=0.22, corner_radius=0.16)
        focus_label = make_text("strongest evidence: STASIS", BODY_SIZE, STASIS, BOLD)
        focus_label.next_to(basin, UP, buff=0.55)
        self.add_subcaption(
            "The strongest measured MoE signal is specific to stopping progress.",
            duration=4,
        )
        self.play(
            success_group.animate.set_opacity(0.16),
            event_group.animate.set_opacity(0.16),
            start_ring.animate.set_opacity(0.25),
            start_dot.animate.set_opacity(0.25),
            Create(focus),
            FadeIn(focus_label),
            run_time=1.8,
        )
        self.wait(2.0)
        self.clean_exit()


class Scene2_CurrentChunkLens(StoryScene):
    CHUNKS = [0, 7, 12, 15, 20]

    def construct(self) -> None:
        title = self.title("One lens. One current chunk.")
        self.add_subcaption(
            "Each readout uses only the current chunk, although the environment has already changed.",
            duration=5,
        )
        self.play(Write(title), run_time=1.5)
        self.wait(1.0)

        success_line = NumberLine(
            x_range=[0, 20, 5],
            length=5.25,
            include_numbers=False,
            include_ticks=True,
            color=SUCCESS,
            stroke_opacity=0.55,
        ).move_to(LEFT * 3.2 + UP * 1.0)
        stasis_line = NumberLine(
            x_range=[0, 20, 5],
            length=5.25,
            include_numbers=False,
            include_ticks=True,
            color=STASIS,
            stroke_opacity=0.55,
        ).move_to(LEFT * 3.2 + DOWN * 0.45)
        success_label = make_text("ep22 success", LABEL_SIZE, SUCCESS, BOLD).next_to(
            success_line, LEFT, buff=0.3
        )
        stasis_label = make_text("ep05 stasis", LABEL_SIZE, STASIS, BOLD).next_to(
            stasis_line, LEFT, buff=0.3
        )

        lens_x = success_line.n2p(0)[0]
        lens = RoundedRectangle(
            width=0.42,
            height=2.05,
            corner_radius=0.12,
            color=CURRENT,
            stroke_width=3,
            fill_color=CURRENT,
            fill_opacity=0.08,
        ).move_to(np.array([lens_x, 0.27, 0.0]))
        k_label = make_text("k = 0", LABEL_SIZE, CURRENT, BOLD).next_to(
            lens, UP, buff=0.25
        )

        grid_success = expert_grid(active_experts(22, 0), SUCCESS, RIGHT * 3.65 + UP * 0.9)
        grid_stasis = expert_grid(active_experts(5, 0), STASIS, RIGHT * 3.65 + DOWN * 0.95)
        grid_title = make_text("current HB expert set", LABEL_SIZE, MOE, BOLD).move_to(
            RIGHT * 3.65 + UP * 2.15
        )
        grid_success_label = make_text("success", LABEL_SIZE, SUCCESS).next_to(
            grid_success, LEFT, buff=0.3
        )
        grid_stasis_label = make_text("stasis", LABEL_SIZE, STASIS).next_to(
            grid_stasis, LEFT, buff=0.3
        )

        self.play(
            Create(success_line),
            Create(stasis_line),
            FadeIn(success_label),
            FadeIn(stasis_label),
            Create(lens),
            FadeIn(k_label),
            FadeIn(grid_title),
            FadeIn(grid_success),
            FadeIn(grid_stasis),
            FadeIn(grid_success_label),
            FadeIn(grid_stasis_label),
            run_time=2.0,
        )
        self.wait(1.2)

        action_cells = VGroup(
            *[
                RoundedRectangle(
                    width=0.38,
                    height=0.34,
                    corner_radius=0.06,
                    stroke_color=STRUCTURE,
                    stroke_width=1,
                    fill_color=STRUCTURE,
                    fill_opacity=0.1,
                )
                for _ in range(10)
            ]
        ).arrange(RIGHT, buff=0.09)
        action_cells.move_to(LEFT * 3.25 + DOWN * 2.05)
        action_label = make_text("one chunk emits 10 actions", LABEL_SIZE, STRUCTURE)
        action_label.next_to(action_cells, DOWN, buff=0.28)
        self.add_subcaption("One selected chunk commits ten open-loop action steps.", duration=3)
        self.play(FadeIn(action_cells), FadeIn(action_label), run_time=0.9)
        self.play(
            LaggedStart(
                *[
                    cell.animate.set_fill(CURRENT, opacity=0.72).set_stroke(
                        CURRENT, width=1.5
                    )
                    for cell in action_cells
                ],
                lag_ratio=0.12,
            ),
            run_time=1.8,
        )
        self.wait(0.8)
        self.play(action_cells.animate.set_opacity(0.28), run_time=0.6)

        readout = make_text("no held-out readout at k0", LABEL_SIZE, STRUCTURE)
        readout.move_to(RIGHT * 3.6 + DOWN * 2.35)
        self.play(FadeIn(readout), run_time=0.6)

        for chunk in self.CHUNKS[1:]:
            target_x = success_line.n2p(chunk)[0]
            next_lens = lens.copy().move_to(np.array([target_x, 0.27, 0.0]))
            next_label = make_text(f"k = {chunk}", LABEL_SIZE, CURRENT, BOLD)
            next_label.next_to(next_lens, UP, buff=0.25)
            next_success = expert_grid(
                active_experts(22, chunk), SUCCESS, RIGHT * 3.65 + UP * 0.9
            )
            next_stasis = expert_grid(
                active_experts(5, chunk), STASIS, RIGHT * 3.65 + DOWN * 0.95
            )

            self.add_subcaption(
                f"Replace the lens with chunk {chunk}; no previous router state is retained.",
                duration=2,
            )
            self.play(
                Transform(lens, next_lens),
                Transform(k_label, next_label),
                Transform(grid_success, next_success),
                Transform(grid_stasis, next_stasis),
                run_time=1.35,
            )
            self.wait(0.7)

            if chunk == 7:
                value = EARLY["results"]["hidden_identity"]["7"]["auc"]
                next_readout = make_text(
                    f"current hidden AUC {value:.3f}  |  no clue",
                    LABEL_SIZE,
                    STRUCTURE,
                    BOLD,
                    5.8,
                ).move_to(RIGHT * 3.55 + DOWN * 2.35)
                self.add_subcaption("At k7, current hidden state is at chance.", duration=3)
                self.play(ReplacementTransform(readout, next_readout), run_time=0.9)
                readout = next_readout
                self.wait(1.0)
            elif chunk == 12:
                hidden = EARLY["results"]["hidden_identity"]["12"]
                routing = EARLY["results"]["route_current"]["12"]
                next_readout = Group(
                    make_text(
                        f"current hidden AUC {hidden['auc']:.3f}  |  weak clue",
                        LABEL_SIZE,
                        CURRENT,
                        BOLD,
                    ),
                    make_text(
                        f"maxT p={hidden['familywise_maxT_p']:.3f}   routing={routing['auc']:.3f} (n.s.)",
                        LABEL_SIZE,
                        STRUCTURE,
                    ),
                ).arrange(DOWN, buff=0.16)
                next_readout.move_to(RIGHT * 3.55 + DOWN * 2.3)
                self.add_subcaption(
                    "At k12, current pre-expert hidden state has a weak corrected signal; current routing does not.",
                    duration=5,
                )
                self.play(FadeOut(readout), FadeIn(next_readout), run_time=1.1)
                readout = next_readout
                self.play(Circumscribe(readout, color=CURRENT), run_time=1.0)
                self.wait(1.6)
            elif chunk == 15:
                onset = DashedLine(
                    stasis_line.n2p(15) + DOWN * 0.45,
                    stasis_line.n2p(15) + UP * 0.45,
                    color=STASIS,
                    dash_length=0.08,
                )
                onset_label = make_text("earliest onset", LABEL_SIZE, STASIS, BOLD)
                onset_label.next_to(onset, UP, buff=0.28).shift(LEFT * 0.18)
                self.add_subcaption("The earliest measured stasis onset is k15.", duration=3)
                self.play(Create(onset), FadeIn(onset_label), run_time=1.0)
                self.wait(1.2)
            elif chunk == 20:
                median = DashedLine(
                    stasis_line.n2p(19) + DOWN * 0.45,
                    stasis_line.n2p(19) + UP * 0.45,
                    color=STASIS,
                    dash_length=0.08,
                )
                median_label = make_text("median k19", LABEL_SIZE, STASIS)
                median_label.next_to(median, DOWN, buff=0.32).shift(RIGHT * 0.12)
                self.add_subcaption("The median stasis onset is k19.", duration=3)
                self.play(Create(median), FadeIn(median_label), run_time=1.0)
                self.wait(1.4)

        boundary = make_text(
            "weak hidden precursor  !=  routed-expert cause",
            BODY_SIZE,
            WHITE_SOFT,
            BOLD,
            11.5,
        ).to_edge(DOWN, buff=0.55)
        self.add_subcaption(
            "This is a hidden-state association, not evidence that routed experts cause the failure.",
            duration=4,
        )
        self.play(
            FadeOut(action_cells),
            FadeOut(action_label),
            FadeOut(readout),
            FadeIn(boundary),
            run_time=1.2,
        )
        self.wait(2.2)
        self.clean_exit()


class Scene3_EvidenceCurve(StoryScene):
    def construct(self) -> None:
        title = self.title("Precursor, then detector")
        self.add_subcaption(
            "Now separate the current-only early result from the later history-based monitor.",
            duration=4,
        )
        self.play(Write(title), run_time=1.5)
        self.wait(1.0)

        axes = Axes(
            x_range=[0, 36, 6],
            y_range=[0, 1.0, 0.2],
            x_length=10.6,
            y_length=4.65,
            axis_config={
                "color": STRUCTURE,
                "stroke_opacity": 0.35,
                "include_numbers": False,
            },
            tips=False,
        ).shift(DOWN * 0.25)
        x_numbers = Group(
            *[
                make_text(str(value), 18, STRUCTURE).next_to(
                    axes.c2p(value, 0), DOWN, buff=0.12
                )
                for value in range(0, 37, 6)
            ]
        )
        y_numbers = Group(
            *[
                make_text(f"{value:.1f}", 18, STRUCTURE).next_to(
                    axes.c2p(0, value), LEFT, buff=0.12
                )
                for value in np.arange(0.2, 1.01, 0.2)
            ]
        )
        x_label = make_text("closed-loop chunk k", LABEL_SIZE, STRUCTURE)
        x_label.next_to(axes, DOWN, buff=0.45)
        y_label = make_text("held-out AUC", LABEL_SIZE, STRUCTURE).rotate(PI / 2)
        y_label.next_to(axes, LEFT, buff=0.45)
        chance = DashedLine(
            axes.c2p(0, 0.5),
            axes.c2p(36, 0.5),
            color=STRUCTURE,
            dash_length=0.12,
            stroke_opacity=0.55,
        )
        chance_label = make_text("chance", LABEL_SIZE, STRUCTURE).next_to(
            chance, RIGHT, buff=0.2
        )
        self.play(
            Create(axes),
            FadeIn(x_numbers),
            FadeIn(y_numbers),
            FadeIn(x_label),
            FadeIn(y_label),
            Create(chance),
            FadeIn(chance_label),
            run_time=1.8,
        )
        self.wait(1.0)

        hidden_values = [
            EARLY["results"]["hidden_identity"][str(k)]["auc"] for k in (7, 12)
        ]
        route_values = [
            EARLY["results"]["route_current"][str(k)]["auc"] for k in (7, 12)
        ]
        hidden_points = [axes.c2p(k, value) for k, value in zip((7, 12), hidden_values)]
        route_points = [axes.c2p(k, value) for k, value in zip((7, 12), route_values)]
        hidden_line = DashedLine(
            hidden_points[0], hidden_points[1], color=CURRENT, dash_length=0.14, stroke_width=4
        )
        route_line = DashedLine(
            route_points[0], route_points[1], color=MOE, dash_length=0.1, stroke_width=3
        )
        hidden_dots = VGroup(*[Dot(point, radius=0.09, color=CURRENT) for point in hidden_points])
        route_dots = VGroup(*[Dot(point, radius=0.075, color=MOE) for point in route_points])
        hidden_label = make_text("current hidden", LABEL_SIZE, CURRENT, BOLD)
        hidden_label.next_to(hidden_points[-1], UP, buff=0.24)
        route_label = make_text("current routing (n.s.)", LABEL_SIZE, MOE)
        route_label.next_to(route_points[-1], DOWN, buff=0.24)

        self.add_subcaption(
            "Only two current-only horizons were tested: k7 and k12.",
            duration=4,
        )
        self.play(
            Create(hidden_line),
            Create(route_line),
            LaggedStart(*[GrowFromCenter(dot) for dot in hidden_dots], lag_ratio=0.25),
            LaggedStart(*[GrowFromCenter(dot) for dot in route_dots], lag_ratio=0.25),
            FadeIn(hidden_label),
            FadeIn(route_label),
            run_time=2.0,
        )
        self.wait(1.5)

        onset_left = axes.c2p(15, 0)[0]
        onset_right = axes.c2p(30, 0)[0]
        onset_band = Rectangle(
            width=onset_right - onset_left,
            height=axes.y_length,
            stroke_width=0,
            fill_color=STASIS,
            fill_opacity=0.08,
        ).move_to(
            np.array(
                [
                    (onset_left + onset_right) / 2,
                    axes.c2p(0, 0.5)[1],
                    0,
                ]
            )
        )
        onset_band.set_z_index(-1)
        median_line = DashedLine(
            axes.c2p(19, 0),
            axes.c2p(19, 1),
            color=STASIS,
            dash_length=0.12,
            stroke_opacity=0.75,
        )
        onset_label = make_text("stasis onsets: k15...k30", LABEL_SIZE, STASIS, BOLD)
        onset_label.move_to(axes.c2p(23, 0.16))
        median_label = make_text("median 19", LABEL_SIZE, STASIS)
        median_label.move_to(axes.c2p(20.7, 0.31))
        self.add_subcaption(
            "The observed stasis onsets begin at k15, with a median at k19.",
            duration=4,
        )
        self.play(FadeIn(onset_band), Create(median_line), FadeIn(onset_label), FadeIn(median_label), run_time=1.8)
        self.wait(1.5)

        history_x = REPLAY["horizons"]
        history_y = REPLAY["auc"]["moe"]
        history_path = VMobject(stroke_color=MOE, stroke_width=4)
        history_path.set_points_smoothly(
            [axes.c2p(k, value) for k, value in zip(history_x, history_y)]
        )
        history_dash = DashedVMobject(history_path, num_dashes=28, dashed_ratio=0.62)
        history_dots = VGroup(
            *[
                Dot(axes.c2p(k, value), radius=0.075, color=MOE)
                for k, value in zip(history_x, history_y)
            ]
        )
        history_label = make_text("7-chunk-history monitor", LABEL_SIZE, MOE, BOLD)
        history_label.move_to(axes.c2p(27.8, 0.69))
        late_value = history_y[-1]
        late_label = make_text(f"k34  AUC {late_value:.4f}", LABEL_SIZE, MOE, BOLD)
        late_label.next_to(axes.c2p(34, late_value), LEFT, buff=0.28).shift(UP * 0.22)

        self.add_subcaption(
            "A different seven-chunk-history monitor becomes strong only after stasis has formed.",
            duration=6,
        )
        self.play(Create(history_dash), run_time=2.8)
        self.play(
            LaggedStart(*[GrowFromCenter(dot) for dot in history_dots], lag_ratio=0.15),
            FadeIn(history_label),
            run_time=1.3,
        )
        self.play(FadeIn(late_label), Circumscribe(history_dots[-1], color=MOE), run_time=1.2)
        self.wait(2.0)

        chart = Group(
            axes,
            x_numbers,
            y_numbers,
            x_label,
            y_label,
            chance,
            chance_label,
            hidden_line,
            route_line,
            hidden_dots,
            route_dots,
            hidden_label,
            route_label,
            onset_band,
            median_line,
            onset_label,
            median_label,
            history_dash,
            history_dots,
            history_label,
            late_label,
        )
        verdict_labels = Group(
            make_text("EARLY", HEADING_SIZE, CURRENT, BOLD),
            make_text("LATE", HEADING_SIZE, MOE, BOLD),
            make_text("NOT SHOWN", HEADING_SIZE, STASIS, BOLD),
        )
        verdict_details = Group(
            make_text("weak pre-expert hidden precursor", BODY_SIZE, WHITE_SOFT),
            make_text("strong no-progress detector", BODY_SIZE, WHITE_SOFT),
            make_text("a routed-expert cause", BODY_SIZE, WHITE_SOFT),
        )
        for row, (label, detail) in enumerate(zip(verdict_labels, verdict_details)):
            y = 1.0 - row
            label.move_to(LEFT * 2.8 + UP * y)
            detail.move_to(RIGHT * 1.15 + UP * y)
        verdict = Group(verdict_labels, verdict_details)
        final_line = make_text(
            "association  !=  causation",
            BODY_SIZE,
            STRUCTURE,
            BOLD,
        ).to_edge(DOWN, buff=0.65)

        self.add_subcaption(
            "The early clue and the late detector are different claims. Neither proves a routed-expert cause.",
            duration=6,
        )
        self.play(FadeOut(chart), FadeOut(title), run_time=1.0)
        self.play(
            LaggedStart(*[FadeIn(item, shift=UP * 0.12) for item in verdict], lag_ratio=0.12),
            run_time=2.2,
        )
        self.play(FadeIn(final_line), run_time=1.0)
        self.wait(3.0)
        self.clean_exit()
