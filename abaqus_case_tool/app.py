from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core import (
    DATA_PATH,
    DATABASE_RANGES,
    FIXED_GEOMETRY,
    LEGACY_ABAQUS_SCRIPT,
    OBJECTIVE_LABELS,
    OPTIMIZATION_RANGES,
    NSGA2Optimizer,
    ParameterSet,
    SurrogateModel,
    default_generated_script_path,
    draw_cross_section,
    generate_abaqus_runner_script,
    make_history_figure,
    make_pareto_figure,
    make_prediction_comparison_figure,
    python_command,
    try_run_abaqus,
)


st.set_page_config(
    page_title="中介机匣优化设计工具",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_surrogate() -> SurrogateModel:
    return SurrogateModel().fit()


@st.cache_data
def load_data_frame() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=r"\s+")


def section_title(text: str):
    st.markdown(f"### {text}")


def metric_card(label: str, value: str):
    st.metric(label, value)


def current_parameter_inputs() -> ParameterSet:
    st.sidebar.header("当前参数")
    return ParameterSet(
        inner=st.sidebar.number_input("内机匣厚度 / mm", min_value=3.0, max_value=9.3, value=8.9, step=0.1),
        split=st.sidebar.number_input("分流环厚度 / mm", min_value=3.0, max_value=9.3, value=6.2, step=0.1),
        outer=st.sidebar.number_input("外机匣厚度 / mm", min_value=4.0, max_value=10.3, value=9.9, step=0.1),
        plate=st.sidebar.number_input("支板厚度 / mm", min_value=2.0, max_value=8.3, value=6.1, step=0.1),
    )


def show_parameterization_tab(params: ParameterSet):
    left, right = st.columns([1.0, 1.3])
    with left:
        section_title("固定几何尺寸")
        st.dataframe(
            pd.DataFrame(
                [{"参数": key, "数值": value} for key, value in FIXED_GEOMETRY.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("固定几何尺寸、阵列数量和网格设置来自现有 Abaqus 建模脚本。")

    with right:
        section_title("截面预览")
        st.pyplot(draw_cross_section(params), use_container_width=True)

    section_title("参数范围")
    range_cols = st.columns(2)
    with range_cols[0]:
        st.write("数据库建模范围")
        st.dataframe(
            pd.DataFrame(
                [
                    {"参数": key, "起点": value[0], "终点": value[1], "步长": value[2]}
                    for key, value in DATABASE_RANGES.items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with range_cols[1]:
        st.write("优化搜索范围")
        st.dataframe(
            pd.DataFrame(
                [
                    {"参数": key, "起点": value[0], "终点": value[1], "步长": value[2]}
                    for key, value in OPTIMIZATION_RANGES.items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def show_surrogate_tab(params: ParameterSet, surrogate: SurrogateModel):
    prediction = surrogate.predict(params)
    nearest = surrogate.nearest_database_case(params)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("质量", f"{prediction.mass_kg:.1f} kg")
    with metric_cols[1]:
        metric_card("最大应力", f"{prediction.stress_mpa:.1f} MPa")
    with metric_cols[2]:
        metric_card("Y向刚度", f"{prediction.stiff_y:.3e} N/m")
    with metric_cols[3]:
        metric_card("Z向刚度", f"{prediction.stiff_z:.3e} N/m")

    section_title("当前参数的代理预测")
    prediction_df = pd.DataFrame(
        [
            {"指标": key, "预测值": value}
            for key, value in prediction.as_dict().items()
        ]
    )
    nearest_df = pd.DataFrame(
        [
            {"字段": col, "最近样本值": nearest[col]}
            for col in nearest.index
        ]
    )
    left, right = st.columns(2)
    with left:
        st.dataframe(prediction_df, hide_index=True, use_container_width=True)
    with right:
        st.write("最近数据库样本")
        st.dataframe(nearest_df, hide_index=True, use_container_width=True)

    section_title("代理模型验证效果")
    metric_table = pd.DataFrame(
        [{"指标": key, "值": value} for key, value in surrogate.metrics.items()]
    )
    st.dataframe(metric_table, hide_index=True, use_container_width=True)

    figure_cols = st.columns(2)
    preview = surrogate.validation_preview if surrogate.validation_preview is not None else pd.DataFrame()
    for idx, objective in enumerate(OBJECTIVE_LABELS):
        with figure_cols[idx % 2]:
            st.pyplot(make_prediction_comparison_figure(preview, objective), use_container_width=True)


def show_optimization_tab(surrogate: SurrogateModel):
    control_cols = st.columns(4)
    with control_cols[0]:
        generations = st.number_input("进化代数", min_value=10, max_value=200, value=50, step=10)
    with control_cols[1]:
        population = st.number_input("种群规模", min_value=20, max_value=200, value=50, step=10)
    with control_cols[2]:
        random_seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)
    with control_cols[3]:
        stiffness_target = st.number_input(
            "刚度约束 / N/m",
            min_value=5.0e7,
            max_value=2.0e8,
            value=1.0e8,
            step=1.0e7,
            format="%.1e",
        )

    if st.button("开始优化", type="primary", use_container_width=True):
        with st.spinner("正在执行 NSGA-II 优化，请稍候..."):
            optimizer = NSGA2Optimizer(surrogate=surrogate, stiffness_target=stiffness_target)
            history, pareto_solutions, best_solution = optimizer.optimize(
                generations=int(generations),
                population_size=int(population),
                random_seed=int(random_seed),
            )
        st.session_state["optimization_history"] = history
        st.session_state["pareto_solutions"] = pareto_solutions
        st.session_state["best_solution"] = best_solution
        st.session_state["stiffness_target"] = stiffness_target

    history = st.session_state.get("optimization_history")
    pareto_solutions = st.session_state.get("pareto_solutions")
    best_solution = st.session_state.get("best_solution")
    active_target = st.session_state.get("stiffness_target", stiffness_target)

    if not history or not pareto_solutions:
        st.info("点击“开始优化”后，这里会显示非支配解集、迭代历史和最优参数。")
        return

    section_title("最优可行解")
    if best_solution is not None:
        best_cols = st.columns(4)
        with best_cols[0]:
            metric_card("最优质量", f"{best_solution.prediction.mass_kg:.1f} kg")
        with best_cols[1]:
            metric_card("最优应力", f"{best_solution.prediction.stress_mpa:.1f} MPa")
        with best_cols[2]:
            metric_card("Y向刚度", f"{best_solution.prediction.stiff_y:.3e} N/m")
        with best_cols[3]:
            metric_card("Z向刚度", f"{best_solution.prediction.stiff_z:.3e} N/m")
        st.dataframe(
            pd.DataFrame(
                [{"参数": key, "值": value} for key, value in best_solution.parameters.as_dict().items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("当前约束下未找到满足双向刚度要求的可行解。")

    figure_cols = st.columns(2)
    with figure_cols[0]:
        st.pyplot(make_history_figure(history), use_container_width=True)
    with figure_cols[1]:
        st.pyplot(make_pareto_figure(pareto_solutions, active_target), use_container_width=True)

    section_title("最终非支配解集")
    pareto_df = pd.DataFrame(
        [
            {
                "内机匣厚度": solution.parameters.inner,
                "分流环厚度": solution.parameters.split,
                "外机匣厚度": solution.parameters.outer,
                "支板厚度": solution.parameters.plate,
                "质量(kg)": solution.prediction.mass_kg,
                "最大应力(MPa)": solution.prediction.stress_mpa,
                "Y向刚度": solution.prediction.stiff_y,
                "Z向刚度": solution.prediction.stiff_z,
            }
            for solution in pareto_solutions[:50]
        ]
    )
    st.dataframe(pareto_df, hide_index=True, use_container_width=True)


def show_abaqus_tab(params: ParameterSet):
    section_title("Abaqus 脚本导出")
    default_path = default_generated_script_path()
    generated_path = st.text_input("生成脚本路径", value=str(default_path))
    job_name = st.text_input("Job 名称", value="Job-case-ui")
    submit_job = st.checkbox("生成脚本时直接提交作业", value=False)
    abaqus_exe = st.text_input("Abaqus 可执行命令或路径", value="abaqus")

    if st.button("导出 Abaqus 运行脚本", use_container_width=True):
        path = generate_abaqus_runner_script(
            parameter_set=params,
            output_path=Path(generated_path),
            submit_job=submit_job,
            job_name=job_name,
        )
        st.success(f"已生成脚本: {path}")
        st.code(path.read_text(encoding="utf-8"), language="python")

    if st.button("尝试调用 Abaqus", use_container_width=True):
        path = generate_abaqus_runner_script(
            parameter_set=params,
            output_path=Path(generated_path),
            submit_job=submit_job,
            job_name=job_name,
        )
        result = try_run_abaqus(abaqus_exe, path)
        st.write("返回码:", result.returncode)
        st.text_area("标准输出", value=result.stdout, height=180)
        st.text_area("标准错误", value=result.stderr, height=180)

    section_title("当前接口说明")
    st.write(f"- 现有 Abaqus 脚本位置：`{LEGACY_ABAQUS_SCRIPT}`")
    st.write("- 工具会生成一个最小运行脚本，调用 `IntermediateCase` 和 `PostProcess`。")
    st.write("- 如果你的 Abaqus 环境可用，命令会按 `abaqus cae noGUI=脚本路径` 的形式执行。")
    st.code(" ".join(["abaqus", "cae", f"noGUI={default_path.name}"]))


def main():
    st.title("中介机匣参数化建模与优化设计工具")
    st.caption("基于现有 Abaqus 脚本、样本数据库和论文流程封装的本地可视化工具。")

    params = current_parameter_inputs()
    surrogate = load_surrogate()
    _ = load_data_frame()

    st.sidebar.divider()
    st.sidebar.write("当前 Python")
    st.sidebar.code(" ".join(python_command()))
    st.sidebar.write("数据库文件")
    st.sidebar.code(str(DATA_PATH))

    tabs = st.tabs(["参数化建模", "代理预测", "多目标优化", "Abaqus接口"])
    with tabs[0]:
        show_parameterization_tab(params)
    with tabs[1]:
        show_surrogate_tab(params, surrogate)
    with tabs[2]:
        show_optimization_tab(surrogate)
    with tabs[3]:
        show_abaqus_tab(params)


if __name__ == "__main__":
    main()
