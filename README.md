# LLMs Collab: 参与式预算（PB）四阶段处理流水线

本项目基于 `raw/Poland_Warszawa_2024.pb` 数据，构建了一个完整的 4-stage pipeline：

1. **Task 1**：项目特征工程（`f(x)`）
2. **Task 2**：选民偏好推断（`θ_i`）
3. **Task 3**：选民相似图构建（供 GNN/图学习使用）
4. **Task 4**：功利主义贪心分配基线（UG Baseline）

整体目标是把原始 PB 文本数据转化为可用于建模、分析和预算分配对比的结构化结果。

---

## 项目结构

```text
.
├── raw/
│   └── Poland_Warszawa_2024.pb
├── output/
│   ├── task1_features.npy
│   ├── task1_features.npz
│   ├── task1_features_feature_names.json
│   ├── task1_features_project_ids.json
│   ├── task2_preferences.npy
│   ├── task2_preferences.npz
│   ├── task3_graph.npz
│   └── task4_ug_baseline.json
├── task1_build_project_features.py
├── task2_infer_voter_preferences.py
├── task3_build_voter_graph.py
├── task4_utilitarian_greedy.py
└── inspect_outputs.py
```

---

## 运行环境

- Python 3.9（项目中已有 `.venv`）
- 核心依赖：
  - `numpy`
  - `scikit-learn`（仅 Task 2 的 `logistic` 方法需要）

建议使用虚拟环境解释器运行：

```bash
./.venv/bin/python <script_name>.py
```

---

## 数据与整体逻辑

输入文件 `raw/Poland_Warszawa_2024.pb` 包含三个区块：

- `META`：全局信息（例如 `budget`）
- `PROJECTS`：项目信息（`project_id`, `name`, `cost`, `category`, `target`, `selected` 等）
- `VOTES`：投票记录（`voter_id`, `vote`, `age`, `sex`, `neighborhood` 等）

四个任务之间是严格串联关系：

1. Task 1 从 `PROJECTS` 得到项目特征矩阵 `F = f(x)`。
2. Task 2 读取 Task 1 的构造逻辑 + `VOTES`，推断每位选民偏好向量 `Θ`。
3. Task 3 读取 Task 2 的 `Θ` + 投票者人口学信息，构建图结构和节点特征。
4. Task 4 使用 Task 1 特征与 Task 2 偏好，计算社会福利并执行贪心预算分配，与真实结果对比。

---

## Task 1：构建项目特征矩阵 `f(x)`

脚本：`task1_build_project_features.py`

### Task 1 任务目标

把每个项目编码为多维连续向量，为后续效用建模提供输入。

### Task 1 输入

- `--input`：PB 文件（默认 `raw/Poland_Warszawa_2024.pb`）

### Task 1 核心逻辑

1. 解析 `META` 和 `PROJECTS` 区块。
2. 将 `category`、`target` 视作多标签字段（逗号分隔），做 one-hot 编码。
3. 取项目成本 `cost`，先除以总预算 `budget` 得到比例，再做 Min-Max 归一化。
4. 组合特征：
   - `category::*`
   - `target::*`
   - `cost_ratio_minmax`

最终矩阵：

- `F` 形状为 `[num_projects, m_features]`

### Task 1 输出

- `output/task1_features.npy`：纯矩阵 `F`
- `output/task1_features.npz`：打包 `matrix`, `feature_names`, `project_ids`
- `output/task1_features_feature_names.json`
- `output/task1_features_project_ids.json`

### Task 1 运行

```bash
./.venv/bin/python task1_build_project_features.py
```

---

## Task 2：推断选民偏好向量 `θ_i`

脚本：`task2_infer_voter_preferences.py`

### Task 2 任务目标

基于“项目是否被某个选民支持”的二元行为，估计每位选民对各特征维度的偏好强度。

### Task 2 输入

- PB 文件（项目与投票）
- Task 1 的特征构造逻辑（脚本中直接复用）
- 关键参数：
  - `--method ridge|logistic`（默认 `ridge`）
  - `--alpha`（ridge 正则）
  - `--C`（logistic 正则倒数）

### Task 2 核心逻辑

1. 解析 `VOTES`，构建二元投票矩阵 `Y`，形状 `[num_voters, num_projects]`。
2. 以 Task 1 的 `F` 作为项目特征。
3. 推断 `Θ`（每行是一个选民的偏好向量）：
   - **ridge（默认，快）**  
     对每个选民：`θ_i = (F^T F + αI)^(-1) F^T y_i`，脚本采用向量化一次求全部选民。
   - **logistic（更贴合二元标签，但慢）**  
     对每位选民单独训练 `LogisticRegression(fit_intercept=False)`。

### Task 2 输出

- `output/task2_preferences.npy`：`Θ`
- `output/task2_preferences.npz`：`theta`, `feature_names`, `voter_ids`, `project_ids`

### Task 2 运行

```bash
# 推荐：速度快
./.venv/bin/python task2_infer_voter_preferences.py --method ridge --alpha 1.0

# 可选：更“分类模型化”，但显著更慢
./.venv/bin/python task2_infer_voter_preferences.py --method logistic --C 1.0
```

---

## Task 3：构建选民相似图（Graph）

脚本：`task3_build_voter_graph.py`

### Task 3 任务目标

构造图学习可用的数据（节点特征 + 边），用于偏好补全/传播等任务。

### Task 3 输入

- PB 文件中的选民信息（`age`, `sex`, `neighborhood`）
- Task 2 输出的 `Θ`（默认 `output/task2_preferences.npy`）
- 参数：`--age-threshold`（默认 5）

### Task 3 核心逻辑

1. 编码人口学属性：
   - `age`：浮点，缺失记 NaN
   - `sex`：`M->0`, `F->1`，缺失 NaN
   - `neighborhood`：字符串分组键
2. **仅在同 neighborhood 内连边**，边条件为：
   - `|age_i - age_j| <= threshold` **或**
   - `sex_i == sex_j`
3. 对每条无向边保存双向索引，形成 `edge_index`（`[2, 2E]`）。
4. 构建节点特征 `X = [θ_i | age_norm | sex_filled]`：
   - age 缺失用中位数填充后归一化
   - sex 缺失填 0.5

### Task 3 输出

- `output/task3_graph.npz`，包含：
  - `x`（节点特征）
  - `edge_index`（图边）
  - `theta`, `ages`, `sex_enc`, `voter_ids`, `neighborhoods`

### Task 3 运行

```bash
./.venv/bin/python task3_build_voter_graph.py --age-threshold 5
```

---

## Task 4：UG 基线预算分配

脚本：`task4_utilitarian_greedy.py`

### Task 4 任务目标

实现功利主义贪心（Utilitarian Greedy）分配规则，作为与真实 PB 结果对比的 baseline。

### Task 4 输入

- PB 文件（项目成本、真实入选标记）
- Task 2 产出的 `Θ`（默认 `output/task2_preferences.npy`）

### Task 4 核心逻辑

1. 用 Task 1 逻辑构建项目特征 `F`。
2. 计算每个项目的聚合效用：
   - 单个选民效用：`u_i(x_j) = θ_i^T f(x_j)`
   - 聚合效用：`U_j = Σ_i u_i(x_j)`
3. 计算成本效益：`U_j / cost_j`。
4. 按成本效益降序贪心选项目，直到预算耗尽。
5. 与真实 PB `selected==1` 的项目集合比较：
   - 入选数
   - 总成本
   - 社会福利
   - 重叠项目数
   - 福利增益

### Task 4 输出

- `output/task4_ug_baseline.json`

### Task 4 运行

```bash
./.venv/bin/python task4_utilitarian_greedy.py
```

---

## 快速全流程执行

按顺序运行：

```bash
./.venv/bin/python task1_build_project_features.py
./.venv/bin/python task2_infer_voter_preferences.py --method ridge --alpha 1.0
./.venv/bin/python task3_build_voter_graph.py --age-threshold 5
./.venv/bin/python task4_utilitarian_greedy.py
```

可选检查输出：

```bash
./.venv/bin/python inspect_outputs.py --dir output --rows 2 --cols 4
```

---

## 当前输出规模（基于现有 `output/`）

- 项目数：`118`
- 特征维度：`11`
- 选民数：`75,255`
- Task 3 节点特征：`(75255, 13)`（`11` 个偏好维 + `age_norm` + `sex`）
- Task 3 边索引：`(2, 255239820)`（规模较大，内存压力高）

---

## 注意事项

- Task 2 的 `logistic` 模式会对每位选民单独拟合模型，计算成本高，建议默认先用 `ridge`。
- Task 3 可能产生超大图（尤其在年龄阈值较宽时），训练前建议检查内存占用和采样策略。
- 所有脚本均提供命令行参数，推荐先用默认参数复现实验，再逐步调参。
