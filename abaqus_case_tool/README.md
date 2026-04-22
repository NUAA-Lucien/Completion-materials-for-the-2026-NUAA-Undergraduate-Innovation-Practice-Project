# 中介机匣参数化建模与优化设计工具

这个目录把原始 `abaqus-case` 工程中的三块能力封装成了一个本地可视化工具：

- 参数化建模
- 代理模型快速预测
- NSGA-II 多目标优化

## 文件说明

- `app.py`：Streamlit 可视化界面
- `core.py`：参数化逻辑、代理模型、优化算法和 Abaqus 脚本导出
- `result.txt`：1225 组数据库样本
- `abaqus_model_legacy.py`：原始 Abaqus 参数化建模脚本备份
- `NN2_legacy.py`：原始代理模型脚本备份
- `optimization_legacy.py`：原始 NSGA-II 脚本备份
- `launch_tool.bat`：一键启动界面

## 启动方式

双击：

```bat
launch_tool.bat
```

或者手动执行：

```powershell
D:\anaconda3\python.exe -m streamlit run C:\Users\Lucien\Documents\Playground\abaqus_case_tool\app.py
```

## 工具功能

### 1. 参数化建模

- 输入 4 个厚度参数
- 生成机匣主体截面预览
- 展示固定几何尺寸、数据库范围和优化范围
- 支持导出 Abaqus 运行脚本

### 2. 代理预测

- 基于 `result.txt` 重新训练代理模型
- 特征维度为 14 维，隐藏层 128 单元
- 输出质量、最大应力、Y 向刚度、Z 向刚度
- 显示验证误差和 4 项预测效果曲线

### 3. 多目标优化

- 采用 NSGA-II
- 目标：质量最小、应力最小、Y/Z 刚度尽可能大
- 支持设置代数、种群规模和刚度约束
- 显示最优可行解、迭代历史和最终非支配解集

### 4. Abaqus 接口

- 根据当前参数导出可直接调用 `IntermediateCase` 的运行脚本
- 若本机 Abaqus 命令可用，可在界面中直接尝试调用

## 说明

- 当前界面中的代理模型采用 `scikit-learn` 重新训练，以避免运行环境缺少 `torch` 时无法启动。
- Abaqus 实体建模与求解仍然依赖本机已安装的 Abaqus 环境。
