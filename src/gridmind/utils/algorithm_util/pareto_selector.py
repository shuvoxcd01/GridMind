from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Sequence, Tuple, TypeVar

from gridmind.algorithms.evolutionary_rl.neuroevolution.neuro_agent import NeuroAgent
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


T = TypeVar("T")


@dataclass
class ParetoConfig:
    """
    Configuration for Pareto dominance.

    Attributes
    ----------
    minimize :
        A Boolean mask indicating which objectives are to be minimized.
        If None, all objectives are assumed to be minimized.
    """

    minimize: Sequence[bool] | None = None


class ParetoSelector(Generic[T]):
    """
    Utility for non-dominated sorting and layered Pareto-front selection.

    Parameters
    ----------
    objective_fn :
        A function mapping an individual to its objective values, e.g.
        objective_fn(individual) -> Sequence[float]
    config :
        ParetoConfig describing whether each objective is minimized or maximized.

    Notes
    -----
    - Non-dominated sorting uses the standard definition:
      A dominates B if A is no worse in all objectives and strictly better in at least one.
    - Maximization is handled by internally flipping the sign of those objectives.
    """

    def __init__(
        self,
        objective_fn: Callable[[T], Sequence[float]],
        config: ParetoConfig | None = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.config = config or ParetoConfig()


    def non_dominated_sort(self, population: Sequence[T]) -> List[List[T]]:
        """
        Perform non-dominated sorting on the given population.

        Returns
        -------
        fronts :
            A list of Pareto fronts. fronts[0] is the first (best) front,
            fronts[1] is the second, and so on.
        """
        if not population:
            return []

        objectives = [tuple(self.objective_fn(ind)) for ind in population]
        n = len(population)

        # If minimize mask not provided, assume all minimize
        num_obj = len(objectives[0])
        minimize_mask = self._get_minimize_mask(num_obj)

        # Pre-transform objectives so everything becomes a minimization
        transformed = [
            self._transform_for_minimize(values, minimize_mask)
            for values in objectives
        ]

        # For each solution i:
        #   S[i] = set of solutions dominated by i
        #   domination_count[i] = number of solutions that dominate i
        S: List[List[int]] = [[] for _ in range(n)]
        domination_count: List[int] = [0] * n

        for i in range(n):
            for j in range(i + 1, n):
                di = transformed[i]
                dj = transformed[j]

                if self._dominates(di, dj):
                    S[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(dj, di):
                    S[j].append(i)
                    domination_count[i] += 1

        # First front: those not dominated by anyone
        fronts: List[List[int]] = []
        current_front: List[int] = [i for i in range(n) if domination_count[i] == 0]

        while current_front:
            fronts.append(current_front)
            next_front: List[int] = []

            for p in current_front:
                for q in S[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)

            current_front = next_front

        # Convert index-based fronts to individual-based fronts
        return [[population[i] for i in front] for front in fronts]

    def select(self, population: Sequence[T], num_selection: int) -> List[T]:
        """
        Select up to k solutions using layered Pareto-front selection.

        Strategy
        --------
        1. Compute Pareto fronts F0, F1, F2, ...
        2. Concatenate fronts in order until k solutions are chosen.
        3. If the last front overflows k, truncate it.

        This matches your rule:
        - Take all non-dominated solutions.
        - If fewer than k, take those that become non-dominated when you
          remove the current non-dominated set, and so on.
        """
        if num_selection <= 0 or not population:
            return []

        fronts = self.non_dominated_sort(population)

        selected: List[T] = []
        for front in fronts:
            remaining = num_selection - len(selected)
            if remaining <= 0:
                break

            if len(front) <= remaining:
                selected.extend(front)
            else:
                distances = self._compute_crowding_distances(front)
                order = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
                selected.extend([front[i] for i in order[:remaining]])
                break

        return selected


    def _get_minimize_mask(self, num_obj: int) -> Tuple[bool, ...]:
        if self.config.minimize is None:
            return tuple(True for _ in range(num_obj))

        if len(self.config.minimize) != num_obj:
            raise ValueError(
                f"Length of minimize mask ({len(self.config.minimize)}) "
                f"does not match number of objectives ({num_obj})."
            )

        return tuple(self.config.minimize)

    @staticmethod
    def _transform_for_minimize(
        values: Sequence[float],
        minimize_mask: Sequence[bool],
    ) -> Tuple[float, ...]:
        """
        Transform objectives so that all of them are effectively minimized.
        For objectives to be maximized, we negate the value.
        """
        transformed = []
        for v, minimize in zip(values, minimize_mask):
            transformed.append(v if minimize else -v)
        return tuple(transformed)

    @staticmethod
    def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
        """
        Check if a dominates b in a minimization setting.
        """
        assert len(a) == len(b)
        not_worse_in_all = True
        strictly_better_in_at_least_one = False

        for va, vb in zip(a, b):
            if va > vb:  # worse in at least one objective
                not_worse_in_all = False
                break
            if va < vb:  # strictly better in at least one
                strictly_better_in_at_least_one = True

        return not_worse_in_all and strictly_better_in_at_least_one


    def _compute_crowding_distances(self, front: Sequence[T]) -> List[float]:
        """
        Compute NSGA-II crowding distances for individuals in a single front.

        Returns a list of distances aligned with the input `front` order.
        Boundary points per objective receive infinite distance.
        """
        n = len(front)
        if n == 0:
            return []
        if n == 1:
            return [float("inf")]

        objs = [tuple(self.objective_fn(ind)) for ind in front]
        m = len(objs[0])

        distances: List[float] = [0.0] * n

        for j in range(m):
            indices = list(range(n))
            indices.sort(key=lambda i: objs[i][j])

            min_val = objs[indices[0]][j]
            max_val = objs[indices[-1]][j]

            distances[indices[0]] = float("inf")
            distances[indices[-1]] = float("inf")

            denom = max_val - min_val
            if denom == 0:
                continue

            for t in range(1, n - 1):
                i_prev = indices[t - 1]
                i_next = indices[t + 1]
                i_curr = indices[t]
                increment = (objs[i_next][j] - objs[i_prev][j]) / denom
                if distances[i_curr] != float("inf"):
                    distances[i_curr] += increment

        return distances


    def render(self, population: Sequence[T], selected: Sequence[T] | None = None, mode: str = "text") -> None | np.ndarray:
        """
        Render Pareto fronts in the specified mode.

        Parameters
        ----------
        population :
            The population to render.
        mode :
            The rendering mode. Supported modes:
            - "human": draws Pareto fronts using matplotlib (requires 2D objectives).
            - "rgb_array": returns an RGB image array of the plot (requires 2D objectives).
        """

        if mode == "human":
            self._render(population, selected=selected)
            plt.show()

            return None
        elif mode == "rgb_array":
            # Create the plot
            self._render(population, selected=selected)

            # Save the plot to a BytesIO buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            # Read the image from the buffer
            import PIL.Image
            image = PIL.Image.open(buf)
            rgb_array = np.array(image)

            buf.close()
            return rgb_array
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def _render(
        self,
        population: Sequence[T],
        connect: str = "line",
        selected: Sequence[T] | None = None,
        highlight: str = "ring",
        ring_size: float = 80.0,
        ring_edge_color: str = "black",
        ring_linewidth: float = 1.5,
        ring_alpha: float = 0.9,
    ) -> None:
        """
        (Optional) Visualize Pareto fronts for 2D objectives.
        Requires matplotlib.

        Parameters
        ----------
        population :
            The population to visualize.
        connect :
            How to connect points within each front:
            - "line": straight line through sorted points
            - "step": right/down step curve
            - None or "": do not connect
        selected :
            Optional subset of population to highlight in the plot.
            Equality is based on object identity to avoid ambiguity.
        highlight :
            Highlight style for selected. Supports:
            - "ring": draws an outer ring around selected points.
            - None or "": no special highlighting.
        ring_size :
            Marker size for the outer ring when highlight == "ring".
        ring_edge_color :
            Edge color for the ring when highlight == "ring".
        ring_linewidth :
            Line width for the ring when highlight == "ring".
        ring_alpha :
            Alpha transparency for the ring when highlight == "ring".
        """
        import logging
        # Temporarily suppress noisy matplotlib DEBUG logs (e.g., font_manager)
        mpl_logger = logging.getLogger("matplotlib")
        fm_logger = logging.getLogger("matplotlib.font_manager")
        prev_mpl_level = mpl_logger.level
        prev_fm_level = fm_logger.level
        prev_mpl_prop = getattr(mpl_logger, "propagate", True)
        prev_fm_prop = getattr(fm_logger, "propagate", True)
        mpl_logger.setLevel(logging.WARNING)
        fm_logger.setLevel(logging.WARNING)
        # Also stop propagating to root handlers that may be DEBUG
        mpl_logger.propagate = False
        fm_logger.propagate = False

        # Import matplotlib after raising log levels to avoid import-time DEBUG spam
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fronts = self.non_dominated_sort(population)
        selected_ids = set(id(x) for x in selected) if selected else set()
        selected_label_added = False

        try:
            plt.figure()
            # Use modern colormap API (avoid deprecated get_cmap); fallback for older Matplotlib
            cmap_module = getattr(mpl, "colormaps", None)
            if cmap_module is not None:
                cmap = cmap_module.get_cmap("tab10").resampled(max(1, len(fronts)))
                color_for_index = lambda idx: cmap(idx / max(1, len(fronts) - 1))
            else:  # pragma: no cover - compatibility path
                from matplotlib import cm as mpl_cm  # fallback
                cmap = mpl_cm.get_cmap("tab10", max(1, len(fronts)))
                color_for_index = lambda idx: cmap(idx)

            for i, front in enumerate(fronts):
                xs: List[float] = []
                ys: List[float] = []
                sel_xs: List[float] = []
                sel_ys: List[float] = []
                for ind in front:
                    obj_values = self.objective_fn(ind)
                    x, y = obj_values[0], obj_values[1]
                    xs.append(x)
                    ys.append(y)
                    if selected_ids and id(ind) in selected_ids:
                        sel_xs.append(x)
                        sel_ys.append(y)

                color = color_for_index(i)
                plt.scatter(xs, ys, color=color, label=f"Front {i}", zorder=3)

                if connect:
                    pts = sorted(zip(xs, ys), key=lambda p: (p[0], p[1]))
                    if pts:
                        xs_sorted, ys_sorted = zip(*pts)
                        if connect == "step":
                            plt.step(xs_sorted, ys_sorted, where="post", color=color, alpha=0.9, zorder=2)
                        else:
                            plt.plot(xs_sorted, ys_sorted, "-", color=color, alpha=0.9, zorder=2)

                # Overlay ring highlight for selected points
                if highlight and highlight == "ring" and sel_xs:
                    plt.scatter(
                        sel_xs,
                        sel_ys,
                        s=ring_size,
                        facecolors="none",
                        edgecolors=ring_edge_color,
                        linewidths=ring_linewidth,
                        alpha=ring_alpha,
                        zorder=4,
                        label=("Selected" if not selected_label_added else None),
                    )
                    selected_label_added = True

            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
            plt.title("Pareto Fronts")
            plt.legend()
            plt.tight_layout()
            #plt.show()
        finally:
            # Restore previous log levels and propagation
            mpl_logger.setLevel(prev_mpl_level)
            fm_logger.setLevel(prev_fm_level)
            mpl_logger.propagate = prev_mpl_prop
            fm_logger.propagate = prev_fm_prop

if __name__ == "__main__":
    from typing import List

    # Suppose each individual is just a vector of two objectives: [f1, f2]
    Individual = List[NeuroAgent]

    population: List[NeuroAgent] = [
        NeuroAgent(fitness=1.0, behavior_score=5.0),
        NeuroAgent(fitness=2.0, behavior_score=4.0),
        NeuroAgent(fitness=1.5, behavior_score=3.0),
        NeuroAgent(fitness=3.0, behavior_score=2.0),
        NeuroAgent(fitness=2.5, behavior_score=6.0),
        NeuroAgent(fitness=4.0, behavior_score=1.0),
        NeuroAgent(fitness=0.5, behavior_score=7.0),
        NeuroAgent(fitness=3.5, behavior_score=3.5),
        NeuroAgent(fitness=2.0, behavior_score=2.0),
    ]

    pareto_config = ParetoConfig(minimize=[False, False])  # Maximize both fitness and behavior_score

    selector = ParetoSelector(NeuroAgent.get_pareto_objectives, config=pareto_config)

    fronts = selector.non_dominated_sort(population)
    print("Front 0:", fronts[0])

    selected = selector.select(population, num_selection=2)
    print("Selected (k=2):", selected)

    # Optional: visualize Pareto fronts
    img = selector.render(population, selected=selected, mode="rgb_array")
    print("Rendered image shape:", img.shape)
    # To display the image using matplotlib (for testing purposes)
    plt.imshow(img)
    plt.axis('off')
    plt.show()