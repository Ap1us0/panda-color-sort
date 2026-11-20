# Panda 随机红蓝分拣验收

## 100 轮合并结果

保留种子 `20261120–20261149` 的首批 30 轮，继续用 4 个 headless 工作进程执行种子 `20261150–20261219` 的 70 轮，追加批次实测耗时 **360.64 s**。合并后种子连续且无重复，2 块与 3 块模式全程交替，各 50 轮。

| 组别 | 单块正确落槽 | 整轮完整分拣 | 仿真耗时中位数 |
| --- | ---: | ---: | ---: |
| 追加 70 轮 | 175/175（100%） | 70/70（100%） | 56.73 s |
| 合并后 2 块模式 | 100/100（100%） | 50/50（100%） | 42.30 s |
| 合并后 3 块模式 | 150/150（100%） | 50/50（100%） | 63.45 s |
| 合计 | **250/250（100%）** | **100/100（100%）** | 56.73 s |

100 轮中记录 250 次抓取尝试，失败尝试与失败场景均为 0。最大落槽中心误差为 **0.744 mm**。逐表核对了合并 CSV 等于两批原始记录的并集，首批 30 轮的全部 6 个文件 SHA-256 未变。39 项自动测试通过，其中合并回归测试检查了多行失败原因的保留和重复种子的拒绝。

[100 轮统计报告](../results/robust_sort/combined_100/report.md) · [成功率与耗时图](../results/robust_sort/combined_100/metrics.png) · [场景 CSV](../results/robust_sort/combined_100/scenes.csv) · [方块 CSV](../results/robust_sort/combined_100/objects.csv) · [抓取尝试 CSV](../results/robust_sort/combined_100/attempts.csv) · [失败 CSV](../results/robust_sort/combined_100/failures.csv) · [原始数据来源和哈希](../results/robust_sort/combined_100/sources.json) · [追加 70 轮原始报告](../results/robust_sort/extension_70/evaluation/report.md)

## 首批 30 轮与冒烟验证

先运行种子 `7300–7304` 的 5 个冒烟场景，5/5 整轮通过，12/12 方块正确落槽。随后交替运行 15 个两块场景和 15 个三块场景。30 个随机场景全部完成，无整轮失败及外部碰撞报码。

| 组别 | 单块正确落槽 | 整轮完整分拣 | 仿真耗时中位数 | 单轮墙钟耗时中位数 |
| --- | ---: | ---: | ---: | ---: |
| 2 块 | 30/30（100%） | 15/15（100%） | 42.30 s | 18.75 s |
| 3 块 | 45/45（100%） | 15/15（100%） | 63.45 s | 29.28 s |
| 合计 | **75/75（100%）** | **30/30（100%）** | 56.73 s | 26.76 s |

批次种子范围为 `20261120–20261149`，4 个 headless 工作进程共用 **196.47 s** 墙钟时间。仿真耗时为 **42.30–71.15 s**，最终最大落槽中心误差为 **0.653 mm**，记录到 8 种抓取顺序。75 次实际尝试全部成功。首批验收时原有 32 项和新增 5 项自动测试均通过。

## 布局与控制

- `src/panda_sort_scenarios.py` 使用 PCG64 在 X `[0.40, 0.60]` m、Y `[-0.22, 0.04]` m 内采样连续 XY，偏航采样 `[-180°, 180°)`。两块为红、蓝各一，三块为两红一蓝，颜色顺序随机打乱，符合两个红槽和一个蓝槽的容量。
- 方块中心之间至少相距 135 mm，与每个槽位至少相距 115 mm。用方块外接圆检查桌面边界，对候选抓取高度和进入高度做六维 IK 检查，模型重置后检查初始接触。采样失败有明确的 400 次布局尝试上限。
- 排序器根据当前末端到方块上方的距离、方块到候选槽的距离，以及物块颜色和实际槽位占用对候选排序。每次抓取的轨迹预检用当前其他方块的姿态建立静态障碍几何，并预留 2 mm 余量。它也检查真实动力学中的接触，任何非预期接触都会停止执行。
- 原有七轴位置控制器、双指腱执行器、雅可比 IK、轨迹插值和夹持检查继续使用。初始方块位姿通过场景模型与重置建立。全轮共用同一份 live `MjData`，方块在重力、摩擦和接触作用下运动。规划模型中的关节初始化仅用于计算轨迹。
- 每轮最多 55,000 个物理步。规划被拒绝时记录原因并尝试下一个候选。执行时检测到掉落、异常接触或达到步数上限会停止推进。所有已完成和跳过的方块都计入预期总数。最终复查槽位、桌面支撑、偏航和物体速度后才计为正确落槽。

## 调试与原始记录

三块代表场景种子为 `7301`。首次回放视频完整生成，但 NumPy 整数使 JSON 导出报错。修复输出类型后，用同一 seed 复跑。该回放的布局、抓取顺序、最终位姿和冒烟原始 CSV 逐项一致。[原始视频](../results/robust_sort/representative_7301/sort.mp4) · [导出错误记录](../results/robust_sort/representative_7301/export_error.txt) · [修复后同种子记录](../results/robust_sort/representative_7301_replay/result.json)

另用种子 `7300` 将预算人为缩短至 23 步，得到 `SETTLE / timeout`，现场恰好在第 23 步停止。该诊断与正式分拣成功率分开记录。[超时诊断 JSON](../results/robust_sort/diagnostic_timeout_7300/result.json)

- [冒烟原始报告](../results/robust_sort/smoke/report.md)
- [30 轮批测报告](../results/robust_sort/evaluation/report.md) · [场景 CSV](../results/robust_sort/evaluation/scenes.csv) · [方块 CSV](../results/robust_sort/evaluation/objects.csv) · [抓取尝试 CSV](../results/robust_sort/evaluation/attempts.csv) · [失败 CSV](../results/robust_sort/evaluation/failures.csv) · [成功率与耗时图](../results/robust_sort/evaluation/metrics.png)

## 复现与合并

从仓库根目录运行。先按 [README](../README.md) 创建虚拟环境。以下命令会把新运行写到 `outputs/`。

```powershell
& '.\.venv\Scripts\python.exe' .\src\panda_sort_evaluate.py --smoke-count 5 --scenes 100 --seed 20261120 --jobs 4 --output-dir .\outputs\sort_reproduction
```

可独立复现追加的 70 轮，也可直接使用附带的原始 30 轮和 70 轮结果重建 100 轮图表。

```powershell
& '.\.venv\Scripts\python.exe' .\src\panda_sort_evaluate.py --smoke-count 0 --scenes 70 --seed 20261150 --jobs 4 --output-dir .\outputs\extension_70
& '.\.venv\Scripts\python.exe' .\src\panda_sort_merge.py .\results\robust_sort\evaluation .\results\robust_sort\extension_70\evaluation --expected-scenes 100 --output-dir .\outputs\combined_100
```
