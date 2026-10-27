# 10 个任务的权威文本

**来源**：HF `Zenith6666/yam-lerobot` 各子数据集的 `meta/tasks.parquet` 的 **index**
（LeRobot v3 把任务文本存在 parquet 的 index 里，不是列 —— 这点容易看漏）。
合计 **991 episodes**，与训练数据完全一致。

**为什么必须逐字一致**：模型的视觉-语言条件是这段文本。喂错文本 = 给错指令，
模型会按**另一个任务**的动作模式输出（不会超限，因为都是学过的轨迹，但动作完全不对）。

| 编号 | episodes | task 文本 |
|---|---:|---|
| `TASK-YAM-0001` | 101 | `Hit the ball with claws to score a goal` |
| `TASK-YAM-0002` | 100 | `Pick up the seal, dip it in the ink pad, and stamp.` |
| `TASK-YAM-0004` | 100 | `Grab the ham sausage to the foam box` |
| `TASK-YAM-0006` | 100 | `Pick up the small square and place it in the groove` |
| `TASK-YAM-0007` | 100 | `One gripper holds the bottle, and the inner side of the other gripper rubs open the bottle cap.` |
| `TASK-YAM-0008` | 91 | `One gripper clamps the bottle, and the other gripper rotates to open the cap.` |
| `TASK-YAM-0010` | 99 | `One gripper holds the gum bottle, and the other gripper pops open the lid.` |
| `TASK-YAM-0013` | 100 | `Sort the blocks from left to right according to their size.` |
| `TASK-YAM-0014` | 100 | `Pick up the towel and wipe the whiteboard with it.` |
| `TASK-YAM-0015` | 100 | `Put items in the drawer.` |

## 易错点（实测踩过的坑）

| 坑 | 具体 |
|---|---|
| 大小写 | 全部**首字母大写**；`0006` 曾用小写 `pick up...` 跑错过 |
| **句号** | 结尾**有**句号：`0002` `0008` `0010` `0013`；**没有**句号：`0001` `0004` `0006` `0007` `0014` `0015` |
| 长句 | `0007`/`0008`/`0010` 是 90+ 字符长句，手打极易漏词 |
| 逗号 | `0002` 和 `0007`/`0010` 中间有逗号，不能省 |

## 所以：用 `--task-id`，不要手打

`serving/bridge/vla_client.py` 内置了这张表：

```bash
# 看全部
python vla_client.py --list-tasks

# 用编号（推荐）
python vla_client.py --task-id TASK-YAM-0006 --host 127.0.0.1 --port 18879 ...
```

`--task-id` 与 `--task` 冲突时**以 `--task-id` 为准**并打 warning，避免手打错还悄悄生效。

## 部署包那边也要同步

部署包的 `live_configs/ar.json` 里 `task` 字段和 `motion_experiment.json` 的 `task`
是**单值**（我们装配时填了 `TASK-YAM-0006` 的文本）。要跑别的任务：

```bash
D=/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026
python3 - <<'PY'
import json, pathlib, sys
D = pathlib.Path("/mnt/sda/chenyuan/motion/g05_ft1ep_ours_20261026")
TASKS = {
 "TASK-YAM-0001": "Hit the ball with claws to score a goal",
 "TASK-YAM-0002": "Pick up the seal, dip it in the ink pad, and stamp.",
 "TASK-YAM-0004": "Grab the ham sausage to the foam box",
 "TASK-YAM-0006": "Pick up the small square and place it in the groove",
 "TASK-YAM-0007": "One gripper holds the bottle, and the inner side of the other gripper rubs open the bottle cap.",
 "TASK-YAM-0008": "One gripper clamps the bottle, and the other gripper rotates to open the cap.",
 "TASK-YAM-0010": "One gripper holds the gum bottle, and the other gripper pops open the lid.",
 "TASK-YAM-0013": "Sort the blocks from left to right according to their size.",
 "TASK-YAM-0014": "Pick up the towel and wipe the whiteboard with it.",
 "TASK-YAM-0015": "Put items in the drawer.",
}
tid = sys.argv[1] if len(sys.argv) > 1 else "TASK-YAM-0006"
txt = TASKS[tid]
for name in ("live_configs/ar.json", "models/ar_ft1ep/motion_experiment.json"):
    p = D / name
    d = json.loads(p.read_text())
    d["task"] = txt
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    print(f"  {name}: task = {txt!r}")
PY
```

> ⚠️ 注意：`yam_motion` 这条路径（官方 `serve_policy` + `G05Policy`）从 `ar.json` 读任务；
> 我们的 `vla_client.py` 这条路径从 `--task-id`/`--task` 读任务，**两者要一致**。

## 复现这张表

```bash
# 本机（Windows，有 HF token）：
python fetch_task_texts.py        # 会下载各 meta/tasks.parquet 到 dataset_meta/
```

脚本位置：`D:\G05\_upload\fetch_task_texts.py`；输出 `dataset_meta/task_texts.json`。
