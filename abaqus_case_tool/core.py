from __future__ import annotations

import math
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Normalizer


TOOL_DIR = Path(__file__).resolve().parent
DATA_PATH = TOOL_DIR / "result.txt"
LEGACY_ABAQUS_SCRIPT = TOOL_DIR / "abaqus_model_legacy.py"
SURROGATE_CACHE = TOOL_DIR / "surrogate_cache.pkl"

DATABASE_RANGES = {
    "内机匣厚度": (3.0, 7.0, 1.0),
    "分流环厚度": (3.0, 9.0, 1.0),
    "外机匣厚度": (4.0, 10.0, 1.0),
    "支板厚度": (2.0, 6.0, 1.0),
}

OPTIMIZATION_RANGES = {
    "内机匣厚度": (3.0, 9.3, 0.1),
    "分流环厚度": (3.0, 9.3, 0.1),
    "外机匣厚度": (4.0, 10.3, 0.1),
    "支板厚度": (2.0, 8.3, 0.1),
}

OBJECTIVE_LABELS = ("质量", "最大应力", "Y向刚度", "Z向刚度")
FIXED_GEOMETRY = {
    "支板外轮廓宽度": 50.0,
    "外机匣加强筋宽度": 40.0,
    "外机匣加强筋高度": 765.0,
    "支板数量": 8,
    "加强筋数量": 8,
    "全局网格尺寸": 18.0,
    "关键部位网格尺寸": 15.0,
}


def configure_matplotlib_fonts() -> None:
    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


@dataclass(frozen=True)
class ParameterSet:
    inner: float
    split: float
    outer: float
    plate: float

    def as_array(self) -> np.ndarray:
        return np.array([self.inner, self.split, self.outer, self.plate], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {
            "内机匣厚度 (mm)": self.inner,
            "分流环厚度 (mm)": self.split,
            "外机匣厚度 (mm)": self.outer,
            "支板厚度 (mm)": self.plate,
        }

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.inner, self.split, self.outer, self.plate)


@dataclass(frozen=True)
class PredictionResult:
    mass_t: float
    stress_mpa: float
    stiff_y: float
    stiff_z: float

    @property
    def mass_kg(self) -> float:
        return self.mass_t * 1000.0

    def as_dict(self) -> dict[str, float]:
        return {
            "质量 (t)": self.mass_t,
            "质量 (kg)": self.mass_kg,
            "最大应力 (MPa)": self.stress_mpa,
            "Y向刚度 (N/m)": self.stiff_y,
            "Z向刚度 (N/m)": self.stiff_z,
        }


@dataclass(frozen=True)
class OptimizationRecord:
    generation: int
    feasible_count: int
    best_feasible_mass_t: float | None


@dataclass(frozen=True)
class OptimizationSolution:
    parameters: ParameterSet
    prediction: PredictionResult
    rank: int
    crowding_distance: float


def cross_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return np.concatenate(
        (
            values,
            values * values,
            values[:, 0:3] * values[:, 1:4],
            values[:, 0:2] * values[:, 2:4],
            values[:, 0:1] * values[:, 3:4],
        ),
        axis=1,
    )


def scale_targets(outputs: np.ndarray) -> np.ndarray:
    outputs = np.asarray(outputs, dtype=float)
    return np.concatenate(
        (
            outputs[:, [0]] * 1e3 - 100.0,
            outputs[:, [1]] - 50.0,
            outputs[:, [2]] * 2e-6,
            outputs[:, [3]] * 2e-6,
        ),
        axis=1,
    )


def inverse_scale_targets(outputs: np.ndarray) -> np.ndarray:
    outputs = np.asarray(outputs, dtype=float)
    return np.concatenate(
        (
            (outputs[:, [0]] + 100.0) * 1e-3,
            outputs[:, [1]] + 50.0,
            outputs[:, [2]] * 5e5,
            outputs[:, [3]] * 5e5,
        ),
        axis=1,
    )


class SurrogateModel:
    def __init__(self, data_path: Path = DATA_PATH, cache_path: Path = SURROGATE_CACHE):
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.normalizer = Normalizer()
        self.model = MLPRegressor(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.001,
            alpha=0.001,
            batch_size=16,
            max_iter=1000,
            random_state=42,
            momentum=0.9,
            n_iter_no_change=1000,
        )
        self._data: pd.DataFrame | None = None
        self.metrics: dict[str, float] = {}
        self.validation_preview: pd.DataFrame | None = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = pd.read_csv(self.data_path, sep=r"\s+")
        return self._data.copy()

    def fit(self, force_retrain: bool = False) -> "SurrogateModel":
        if self.cache_path.exists() and not force_retrain:
            with self.cache_path.open("rb") as fp:
                cached = pickle.load(fp)
            self.model = cached["model"]
            self.normalizer = cached["normalizer"]
            self.metrics = cached["metrics"]
            self.validation_preview = cached["validation_preview"]
            self._data = cached["data"]
            return self

        df = self.data
        inputs = df.iloc[:, :4].to_numpy(dtype=float)
        outputs = df.iloc[:, 4:].to_numpy(dtype=float)
        features = cross_features(inputs)
        features = self.normalizer.fit_transform(features)
        scaled_outputs = scale_targets(outputs)

        x_train, x_test, y_train, y_test, raw_train, raw_test = train_test_split(
            features,
            scaled_outputs,
            outputs,
            test_size=0.3,
            random_state=42,
        )

        self.model.fit(x_train, y_train)
        pred_test = inverse_scale_targets(self.model.predict(x_test))

        metric_rows = {}
        preview_rows = []
        for idx, label in enumerate(("质量", "最大应力", "Y向刚度", "Z向刚度")):
            mae = mean_absolute_error(raw_test[:, idx], pred_test[:, idx])
            denom = np.maximum(np.abs(raw_test[:, idx]), 1e-9)
            mape = np.mean(np.abs(pred_test[:, idx] - raw_test[:, idx]) / denom) * 100.0
            metric_rows[f"{label}MAE"] = float(mae)
            metric_rows[f"{label}MAPE%"] = float(mape)

        preview_count = min(60, len(raw_test))
        for i in range(preview_count):
            preview_rows.append(
                {
                    "样本序号": i + 1,
                    "真实质量(kg)": raw_test[i, 0] * 1000.0,
                    "预测质量(kg)": pred_test[i, 0] * 1000.0,
                    "真实应力(MPa)": raw_test[i, 1],
                    "预测应力(MPa)": pred_test[i, 1],
                    "真实Y刚度": raw_test[i, 2],
                    "预测Y刚度": pred_test[i, 2],
                    "真实Z刚度": raw_test[i, 3],
                    "预测Z刚度": pred_test[i, 3],
                }
            )

        self.metrics = metric_rows
        self.validation_preview = pd.DataFrame(preview_rows)
        self._data = df

        with self.cache_path.open("wb") as fp:
            pickle.dump(
                {
                    "model": self.model,
                    "normalizer": self.normalizer,
                    "metrics": self.metrics,
                    "validation_preview": self.validation_preview,
                    "data": self._data,
                },
                fp,
            )
        return self

    def predict_batch(self, parameters: Iterable[Iterable[float]]) -> np.ndarray:
        values = np.asarray(list(parameters), dtype=float)
        features = self.normalizer.transform(cross_features(values))
        return inverse_scale_targets(self.model.predict(features))

    def predict(self, parameter_set: ParameterSet) -> PredictionResult:
        values = self.predict_batch([parameter_set.as_tuple()])[0]
        return PredictionResult(
            mass_t=float(values[0]),
            stress_mpa=float(values[1]),
            stiff_y=float(values[2]),
            stiff_z=float(values[3]),
        )

    def nearest_database_case(self, parameter_set: ParameterSet) -> pd.Series:
        df = self.data
        params = df.iloc[:, :4].to_numpy(dtype=float)
        diffs = params - parameter_set.as_array()
        index = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        return df.iloc[index]


def _resolve_lines(points: list[tuple[float | str, float]]) -> np.ndarray:
    resolved: list[tuple[float, float]] = []
    for item in points:
        x_or_arrow, y = item
        if isinstance(x_or_arrow, str):
            prev_x, prev_y = resolved[-1]
            distance = y
            if x_or_arrow == "↑":
                resolved.append((prev_x, prev_y + distance))
            elif x_or_arrow == "↓":
                resolved.append((prev_x, prev_y - distance))
            elif x_or_arrow == "←":
                resolved.append((prev_x - distance, prev_y))
            elif x_or_arrow == "→":
                resolved.append((prev_x + distance, prev_y))
            else:
                raise ValueError(f"Unknown sketch direction: {x_or_arrow}")
        else:
            resolved.append((float(x_or_arrow), float(y)))
    return np.asarray(resolved, dtype=float)


def build_case_sections(parameter_set: ParameterSet) -> dict[str, np.ndarray]:
    ht_outer = 39.0 - parameter_set.outer
    inner_case = [
        (0.0, 187.0),
        ("→", 12.0),
        ("↑", 5.0),
        ("←", 9.0),
        ("↑", 12.0 - parameter_set.inner),
        ("→", 339.5),
        ("↓", 17.0 - parameter_set.inner),
        ("→", 103.0),
        ("↓", 55.0),
        ("→", 4.5),
        ("↑", 17.0),
        ("→", 6.0),
        ("↑", 29.0),
        ("→", 8.0),
        ("↑", 26.0),
        ("←", 464.0),
        ("↓", 17.0),
    ]
    split_case = [
        (-175.0, 485.0),
        ("→", 275.0),
        (349.5, 463.5),
        ("↑", 4.5),
        (200.0, 480.0),
        (200.0, 485.0),
        (349.5, 485.0),
        ("↑", parameter_set.split),
        ("←", 524.5),
        ("↓", parameter_set.split),
    ]
    outer_case = [
        (-120.5, 728.0),
        ("→", 470.0),
        ("↑", 39.0),
        ("←", 12.0),
        ("↓", ht_outer),
        ("←", 59.0),
        ("↑", ht_outer),
        ("←", 16.0),
        ("↓", ht_outer),
        ("←", 59.0),
        ("↑", ht_outer),
        ("←", 16.0),
        ("↓", ht_outer),
        ("←", 59.0),
        ("↑", ht_outer),
        ("←", 16.0),
        ("↓", ht_outer),
        ("←", 59.0),
        ("↑", ht_outer),
        ("←", 16.0),
        ("↓", ht_outer),
        (-113.5, 728.0 + parameter_set.outer),
        (-113.5, 770.0),
        ("←", 7.0),
    ]
    return {
        "内机匣截面": _resolve_lines(inner_case),
        "分流环截面": _resolve_lines(split_case),
        "外机匣截面": _resolve_lines(outer_case),
    }


def parameterization_summary() -> list[str]:
    return [
        "机匣主体由内机匣、分流环和外机匣三段截面草图构成，再通过 360° 回转生成三维实体。",
        "支板先生成外轮廓与内轮廓，再通过布尔切割形成单块支板，之后进行 8 份环向阵列。",
        "外机匣加强结构由矩形拉伸体生成，经边界切除后同样进行 8 份环向阵列。",
        "最终通过布尔切除和布尔合并得到完整中介机匣实体，再完成材料、耦合、载荷、网格和作业设置。",
    ]


def draw_cross_section(parameter_set: ParameterSet):
    sections = build_case_sections(parameter_set)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    colors = {
        "内机匣截面": "#1f77b4",
        "分流环截面": "#ff7f0e",
        "外机匣截面": "#2ca02c",
    }
    for name, coords in sections.items():
        closed = np.vstack([coords, coords[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=colors[name], linewidth=2.0, label=name)
        ax.fill(closed[:, 0], closed[:, 1], color=colors[name], alpha=0.08)
    ax.set_title("中介机匣参数化截面预览")
    ax.set_xlabel("轴向尺寸 / mm")
    ax.set_ylabel("半径方向尺寸 / mm")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    return fig


def _format_param(value: float) -> str:
    return f"{value:.1f}"


def generate_abaqus_runner_script(
    parameter_set: ParameterSet,
    output_path: Path,
    abaqus_script_path: Path = LEGACY_ABAQUS_SCRIPT,
    submit_job: bool = False,
    job_name: str = "Job-case-ui",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""# Auto-generated by abaqus_case_tool
import sys
sys.path.insert(0, r"{abaqus_script_path.parent}")

from abaqus_model_legacy import IntermediateCase, PostProcess

job_name = r"{job_name}"
case = IntermediateCase(
    i={_format_param(parameter_set.inner)},
    s={_format_param(parameter_set.split)},
    o={_format_param(parameter_set.outer)},
    p={_format_param(parameter_set.plate)},
    sub_job={str(submit_job)},
    job_name=job_name,
)
print("mass=", case.mass)
if {str(submit_job)}:
    print("result=", PostProcess(job_name).result)
"""
    output_path.write_text(script, encoding="utf-8")
    return output_path


def build_abaqus_command(abaqus_executable: str, runner_script: Path) -> list[str]:
    return [abaqus_executable, "cae", f"noGUI={runner_script}"]


def try_run_abaqus(abaqus_executable: str, runner_script: Path, workdir: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        build_abaqus_command(abaqus_executable, runner_script),
        cwd=workdir or runner_script.parent,
        capture_output=True,
        text=True,
        check=False,
    )


class NSGA2Optimizer:
    def __init__(
        self,
        surrogate: SurrogateModel,
        ranges: dict[str, tuple[float, float, float]] | None = None,
        stiffness_target: float = 1.0e8,
    ):
        self.surrogate = surrogate
        self.ranges = ranges or OPTIMIZATION_RANGES
        self.stiffness_target = stiffness_target
        self._range_items = list(self.ranges.values())
        self._bits_per_gene = [
            len(format(int(round((end - start) / step)), "b")) for start, end, step in self._range_items
        ]
        self.chromosome_size = sum(self._bits_per_gene)

    def random_chromosome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=self.chromosome_size, dtype=int)

    def decode(self, chromosome: np.ndarray) -> ParameterSet:
        cursor = 0
        values = []
        for (start, end, step), digit in zip(self._range_items, self._bits_per_gene):
            gene = chromosome[cursor : cursor + digit]
            cursor += digit
            number = 0
            for power, bit in enumerate(gene[::-1]):
                number += int(bit) * (2**power)
            upper_index = int(round((end - start) / step))
            number = min(number, upper_index)
            values.append(round(start + number * step, 1))
        return ParameterSet(*values)

    def encode(self, parameter_set: ParameterSet) -> np.ndarray:
        chromosome = []
        for value, (start, end, step), digit in zip(parameter_set.as_tuple(), self._range_items, self._bits_per_gene):
            index = int(round((value - start) / step))
            upper_index = int(round((end - start) / step))
            index = max(0, min(index, upper_index))
            chromosome.extend(int(bit) for bit in format(index, f"0{digit}b"))
        return np.asarray(chromosome, dtype=int)

    def _constraint_violation(self, outputs: np.ndarray) -> float:
        return max(0.0, self.stiffness_target - outputs[2]) + max(0.0, self.stiffness_target - outputs[3])

    def _dominates(self, left: np.ndarray, right: np.ndarray) -> bool:
        left_violation = self._constraint_violation(left)
        right_violation = self._constraint_violation(right)
        if left_violation == 0.0 and right_violation > 0.0:
            return True
        if left_violation > 0.0 and right_violation == 0.0:
            return False
        if left_violation > 0.0 and right_violation > 0.0:
            return left_violation < right_violation

        better_or_equal = (
            left[0] <= right[0]
            and left[1] <= right[1]
            and left[2] >= right[2]
            and left[3] >= right[3]
        )
        strictly_better = (
            left[0] < right[0]
            or left[1] < right[1]
            or left[2] > right[2]
            or left[3] > right[3]
        )
        return better_or_equal and strictly_better

    def _fast_non_dominated_sort(self, outputs: np.ndarray) -> list[list[int]]:
        domination_sets = [set() for _ in range(len(outputs))]
        dominated_count = [0 for _ in range(len(outputs))]
        fronts: list[list[int]] = [[]]

        for p in range(len(outputs)):
            for q in range(len(outputs)):
                if p == q:
                    continue
                if self._dominates(outputs[p], outputs[q]):
                    domination_sets[p].add(q)
                elif self._dominates(outputs[q], outputs[p]):
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in domination_sets[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _crowding_distance(self, front: list[int], outputs: np.ndarray) -> dict[int, float]:
        if not front:
            return {}
        distance = {idx: 0.0 for idx in front}
        objectives = [
            (0, False),
            (1, False),
            (2, True),
            (3, True),
        ]
        for objective_index, descending in objectives:
            ordered = sorted(front, key=lambda idx: outputs[idx][objective_index], reverse=descending)
            distance[ordered[0]] = math.inf
            distance[ordered[-1]] = math.inf
            objective_values = [outputs[idx][objective_index] for idx in ordered]
            span = max(objective_values) - min(objective_values)
            if span == 0:
                continue
            for pos in range(1, len(ordered) - 1):
                prev_value = outputs[ordered[pos - 1]][objective_index]
                next_value = outputs[ordered[pos + 1]][objective_index]
                distance[ordered[pos]] += abs(next_value - prev_value) / span
        return distance

    def _select_parents(
        self,
        population: list[np.ndarray],
        fronts: list[list[int]],
        crowding: dict[int, float],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        rank_map = {}
        for rank, front in enumerate(fronts, start=1):
            for idx in front:
                rank_map[idx] = rank

        def tournament() -> np.ndarray:
            indices = rng.choice(len(population), size=2, replace=False)
            best = indices[0]
            for idx in indices[1:]:
                if rank_map[idx] < rank_map[best]:
                    best = idx
                elif rank_map[idx] == rank_map[best] and crowding.get(idx, 0.0) > crowding.get(best, 0.0):
                    best = idx
            return population[best]

        return tournament().copy(), tournament().copy()

    def optimize(
        self,
        generations: int = 50,
        population_size: int = 50,
        mutation_rate: float = 1.0 / 24.0,
        random_seed: int = 42,
    ) -> tuple[list[OptimizationRecord], list[OptimizationSolution], OptimizationSolution | None]:
        rng = np.random.default_rng(random_seed)
        population = [self.random_chromosome(rng) for _ in range(population_size)]
        history: list[OptimizationRecord] = []
        final_front: list[OptimizationSolution] = []
        best_solution: OptimizationSolution | None = None

        for generation_index in range(generations):
            params = [self.decode(chromosome) for chromosome in population]
            outputs = self.surrogate.predict_batch([param.as_tuple() for param in params])
            fronts = self._fast_non_dominated_sort(outputs)
            crowding = {}
            for front in fronts:
                crowding.update(self._crowding_distance(front, outputs))

            feasible_indices = [idx for idx, row in enumerate(outputs) if row[2] >= self.stiffness_target and row[3] >= self.stiffness_target]
            best_mass = None
            if feasible_indices:
                best_idx = min(feasible_indices, key=lambda idx: outputs[idx][0])
                best_mass = float(outputs[best_idx][0])
                solution = OptimizationSolution(
                    parameters=params[best_idx],
                    prediction=PredictionResult(*map(float, outputs[best_idx])),
                    rank=1,
                    crowding_distance=crowding.get(best_idx, 0.0),
                )
                if best_solution is None or solution.prediction.mass_t < best_solution.prediction.mass_t:
                    best_solution = solution

            history.append(
                OptimizationRecord(
                    generation=generation_index + 1,
                    feasible_count=len(feasible_indices),
                    best_feasible_mass_t=best_mass,
                )
            )

            ranked_population: list[np.ndarray] = []
            for front in fronts:
                distances = self._crowding_distance(front, outputs)
                ordered = sorted(front, key=lambda idx: distances.get(idx, 0.0), reverse=True)
                for idx in ordered:
                    ranked_population.append(population[idx])
                    if len(ranked_population) == population_size:
                        break
                if len(ranked_population) == population_size:
                    break

            offspring = []
            while len(offspring) < population_size:
                parent_a, parent_b = self._select_parents(population, fronts, crowding, rng)
                point = rng.integers(1, self.chromosome_size - 1)
                child = np.concatenate([parent_a[:point], parent_b[point:]]).astype(int)
                mutation_mask = rng.random(self.chromosome_size) < mutation_rate
                child[mutation_mask] = 1 - child[mutation_mask]
                offspring.append(child)

            population = ranked_population[: max(1, population_size // 5)]
            while len(population) < population_size:
                population.append(offspring.pop(0))

        params = [self.decode(chromosome) for chromosome in population]
        outputs = self.surrogate.predict_batch([param.as_tuple() for param in params])
        fronts = self._fast_non_dominated_sort(outputs)
        if fronts:
            first_front = fronts[0]
            front_crowding = self._crowding_distance(first_front, outputs)
            final_front = [
                OptimizationSolution(
                    parameters=params[idx],
                    prediction=PredictionResult(*map(float, outputs[idx])),
                    rank=1,
                    crowding_distance=front_crowding.get(idx, 0.0),
                )
                for idx in first_front
            ]
            final_front.sort(
                key=lambda item: (
                    item.prediction.stiff_y < self.stiffness_target or item.prediction.stiff_z < self.stiffness_target,
                    item.prediction.mass_t,
                )
            )
        return history, final_front, best_solution


def make_prediction_comparison_figure(preview: pd.DataFrame, value_key: str):
    labels = {
        "质量": ("真实质量(kg)", "预测质量(kg)", "质量 / kg"),
        "最大应力": ("真实应力(MPa)", "预测应力(MPa)", "最大应力 / MPa"),
        "Y向刚度": ("真实Y刚度", "预测Y刚度", "Y向刚度 / N/m"),
        "Z向刚度": ("真实Z刚度", "预测Z刚度", "Z向刚度 / N/m"),
    }
    real_col, pred_col, ylabel = labels[value_key]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(preview["样本序号"], preview[real_col], label="真实值", linewidth=1.8)
    ax.plot(preview["样本序号"], preview[pred_col], label="预测值", linewidth=1.8)
    ax.set_xlabel("样本序号")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{value_key}预测效果")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def make_pareto_figure(solutions: list[OptimizationSolution], stiffness_target: float = 1.0e8):
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    if solutions:
        mass = [solution.prediction.mass_kg for solution in solutions]
        stiff_y = [solution.prediction.stiff_y for solution in solutions]
        color = [solution.prediction.stiff_z for solution in solutions]
        scatter = ax.scatter(mass, stiff_y, c=color, cmap="viridis", s=60, alpha=0.85)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Z向刚度 / N/m")
    ax.axhline(stiffness_target, color="red", linestyle="--", linewidth=1.2, label="Y向刚度约束")
    ax.set_xlabel("质量 / kg")
    ax.set_ylabel("Y向刚度 / N/m")
    ax.set_title("最终非支配解集")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def make_history_figure(history: list[OptimizationRecord]):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    generations = [record.generation for record in history]
    feasible = [record.feasible_count for record in history]
    masses = [
        np.nan if record.best_feasible_mass_t is None else record.best_feasible_mass_t * 1000.0
        for record in history
    ]
    ax1.plot(generations, masses, color="#1f77b4", linewidth=2.0, label="最优可行质量")
    ax1.set_xlabel("代数")
    ax1.set_ylabel("质量 / kg", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(generations, feasible, color="#ff7f0e", alpha=0.25, label="可行个体数")
    ax2.set_ylabel("可行个体数", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.set_title("优化过程收敛概览")
    return fig


def default_generated_script_path() -> Path:
    return TOOL_DIR / "generated_jobs" / "generated_case_run.py"


def python_command() -> list[str]:
    return [sys.executable]
