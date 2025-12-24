Isaac-Ant-Direct-v0


# 1. 任务介绍

## 1.1. 核心目标

使用四足 Ant 机器人在平坦地面上保持稳定姿态，并沿正 X 方向持续前进，不摔倒。

## 1.2. 观测空间

### 1.2.1. 内容

1. 观测空间指的是机器人在某一时刻能够“感知到”的所有信息的集合，
在这个任务中观测空间为 36 维，由 LocomotionEnv 中拼接得到，主要包含：
   * 机身高度（1 维）
      Ant 机身质心在世界坐标系下的 z 坐标，用于判断是否站立或跌倒。
   * 机身线速度（3 维）
      机身在自身坐标系下的线速度。
   * 机身角速度（3 维）
      机身在自身坐标系下的角速度，经过 angular_velocity_scale 缩放。
   * 姿态与目标方向信息（3 维）
      包括 yaw、roll 和指向目标方向的角度差 angle_to_target。
   * 姿态投影（2 维）
      up_proj 表示机身 up 向量与世界 z 轴对齐程度，heading_proj 表示机身前向与目标方向的对齐程度。
   * 关节位置（8 维）
      dof_pos_scaled，为 8 个关节角度按上下限归一化后的值。
   * 关节速度（8 维）
      dof_vel 乘以 dof_vel_scale，表示各关节角速度。
   * 上一时刻动作（8 维）
      上一步执行的动作向量，用于提供动作历史信息。

2. 下面是 LocomotionEnv._get_observations 中对应的代码，可以看到 36 维观测是如何按顺序拼接出来的：

```python
def _get_observations(self) -> dict:
    obs = torch.cat(
        (
            self.torso_position[:, 2].view(-1, 1),                # 机身高度 (1)
            self.vel_loc,                                         # 机身线速度 (3)
            self.angvel_loc * self.cfg.angular_velocity_scale,    # 机身角速度 (3)
            normalize_angle(self.yaw).unsqueeze(-1),              # yaw (1)
            normalize_angle(self.roll).unsqueeze(-1),             # roll (1)
            normalize_angle(self.angle_to_target).unsqueeze(-1),  # 目标方向角度差 (1)
            self.up_proj.unsqueeze(-1),                           # 姿态投影 up_proj (1)
            self.heading_proj.unsqueeze(-1),                      # 姿态投影 heading_proj (1)
            self.dof_pos_scaled,                                  # 关节位置 (8)
            self.dof_vel * self.cfg.dof_vel_scale,                # 关节速度 (8)
            self.actions,                                         # 上一时刻动作 (8)
        ),
        dim=-1,
    )
    observations = {"policy": obs}
    return observations
```

### 1.2.2. 问题
1. 为什么观测里只用了 yaw 和 roll，而没有 pitch / 完整位姿？


* 这个环境是典型的平地行走/奔跑任务，观测设计是“够用且简洁”的，而不是把完整姿态都塞进去：

```python
self.torso_position[:, 2]          # 只用高度（z）
self.vel_loc                       # 线速度（本体坐标系）
self.angvel_loc                    # 角速度（本体坐标系，已经有3轴信息）
normalize_angle(self.yaw)
normalize_angle(self.roll)
normalize_angle(self.angle_to_target)
self.up_proj, self.heading_proj    # 身体朝向 / 竖直方向与环境的对齐程度
...
```

* 核心原因：

- 任务主要关心“朝哪走”和“会不会侧翻”  
  - yaw：决定机器人在平面上的朝向，与目标方向（`angle_to_target`、`heading_proj`）直接相关，是行走任务里最重要的姿态角。  
  - roll：决定左右侧倾，对“会不会摔倒”非常敏感，平地行走时维持 roll ≈ 0 就基本不会侧翻。  

- pitch 信息在这个设计里是“间接可见 + 相对没那么关键” 
  - 有 `torso_position[:, 2]`（高度）、`vel_loc`、`angvel_loc`、`up_proj` 等，这些量已经能让策略推断出身体前后俯仰的大致状态。  
  - 对大多数对称的四足/多足 locomotion 任务，前后轻微俯仰对“走向目标”的决策影响不如 yaw / roll 明显，而俯仰过大会直接反映到高度、up_proj、关节姿态等上面。  

- 观测要避免冗余，保持简单有利于学习  
  - 已经给了本体角速度 `angvel_loc`（包含 3 轴），再给完整姿态（含 pitch）+ 多个冗余角，可能会让网络更难训练（输入维度变大且相关特征更多）。  
  - 这里保留 “最直接影响控制决策的角度”：朝向（yaw）和侧倾（roll），其余通过速度、投影量（`up_proj`, `heading_proj`）、高度、关节角来间接表达。  

- 不提供完整 (x, y, quat) 也是有意为之 
  - 只用 z 高度，而不提供平面位置，是为了平移不变性：无论机器人在地面哪个地方，策略的输入形式是类似的，只和“相对目标的方向/距离”和自身姿态有关。  
  - 朝向差用 `angle_to_target`、`heading_proj` 表达，比直接给全局四元数更稳定、更贴合任务。  

所以，不是说 pitch 不重要，而是对这个平地 locomotion 任务来说，yaw + roll + 高度 + 各种速度/投影/关节状态已经足够表达姿态信息，在不明显提升效果的情况下去掉 pitch，可以让观测空间更小、更干净，有利于策略快速收敛。  

2. 为什么线速度和角速度使用的是自身坐标系，而不是世界坐标系？


## 1.3. 动作空间

### 1.3.1 内容

1. 动作空间为 Box(-1, 1, (8,), float32，对应 AntEnvCfg.action_space = 8)。
8 维连续动作分别对应 8 个可控关节的力矩指令，经过 action_scale 和 joint_gears 缩放后写入仿真。

2. 在 LocomotionEnv._apply_action 中，动作是这样被转换为关节力矩的：

```python
def _apply_action(self):
    # actions: [num_envs, 8]，每个元素通常在 [-1, 1] 范围
    forces = self.action_scale * self.joint_gears * self.actions
    # 将力矩写入到机器人所有关节
    self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
```

可以看到：
1. joint_gears(关节齿轮比) 在 AntEnvCfg 中定义为 [15, 15, 15, 15, 15, 15, 15, 15]；
2. action_scale 在 AntEnvCfg 中为 0.5；
3. 最终每个关节的力矩约为 0.5 × 15 × action_i ≈ 7.5 × action_i。

因此，如果觉得动作太“猛”，可以减小 action_scale 或 joint_gears；如果觉得机器人几乎不动，则可以适当增大这些系数。

### 1.3.2. 问题

1. 这个 Ant 机器人只有 8 个自由度，但是Go1 有12个， 他们之间的差异是什么？ Go1 多了四个有什么优势？

## 1.4. 终止条件

### 1.4.1. 成功

在训练脚本中没有显式“成功”终止条件，主要通过奖励函数鼓励以下行为：
1. 朝正 X 方向前进，缩短与目标位置的距离。
2. 保持机身直立，上下翻滚角度不过大。
3. 避免频繁触碰关节位置极限。
4. 尽量减少动作幅度和能耗。

### 1.4.2. 失败/重置

环境在以下情况会触发重置（Done）：

1. 倒地/摔倒：
   当机器人躯干（Torso）的高度 `torso_position[:, 2]` 低于设定阈值 `termination_height`（默认为 0.31m）时，判定为任务失败（died）。这通常意味着机器人翻倒或趴在地上。此时会触发 `death_cost` 惩罚。

2. **超时**：
   当前回合步数 `episode_length_buf` 达到最大限制 `max_episode_length`（默认为 15.0s / dt）时，环境会自动重置。

## 1.5. 任务难点与潜在挑战

1. 四足机器人需要同时协调 8 个关节，控制维度较高。
2. 如果奖励中前进项和姿态项权重不合适，可能出现“原地抖动”“乱跳”等不稳定策略。
3. 过大的动作尺度或关节齿轮系数可能导致机器人容易摔倒，训练初期不收敛。


# 2. 启动任务训练

## 2.1. 训练框架

   本课程使用 skrl 作为强化学习训练框架，底层基于 PyTorch，实现了 PPO 等常用算法。
Ant 任务的 skrl 配置文件为：

```bash
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ant/agents/skrl_ppo_cfg.yaml
```

## 2.2. 命令行启动训练示例

1. 方式一：通过 isaaclab.sh 启动

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
   --task Isaac-Ant-Direct-v0 \
   --algorithm PPO \
   --num_envs 4096 \
   --device cuda:3 \
   --headless   
```

2. 方式二：直接使用 python（需要先激活对应 Python 环境）
```bash
python scripts/reinforcement_learning/skrl/train.py \
   --task Isaac-Ant-Direct-v0 \
   --algorithm PPO \
   --num_envs 4096 \
   --device cuda:3 \
   --headless 
```


## 2.3. 关键参数说明

* ./isaaclab.sh -p
  使用 IsaacLab 提供的脚本启动 Python 解释器，可以替换为直接调用 python。 

* scripts/reinforcement_learning/skrl/train.py
  通用的 skrl 训练脚本，会根据 --task 查找对应环境注册信息和 skrl_ppo_cfg.yaml 配置。

* --task=Isaac-Ant-Direct-v0
  指定训练的任务环境名称，对应 direct/ant/__init__.py 中注册的 Gym 环境。

* --algorithm
  选择使用的强化学习算法，这里设置为 PPO，对应 skrl_ppo_cfg.yaml 中的 PPO 配置。

* --num_envs
  可选参数，不指定时使用 AntEnvCfg.scene.num_envs 的默认值（4096），可以根据显存情况手动减小，如 1024。

* --device
  指定运行仿真和训练的设备，例如 cuda:0、cuda:3 或 cpu，通常选择某块 GPU 来加速训练。

* --headless
  以无图形界面方式运行仿真，节省 GPU 资源，更适合大批量并行训练。


# 3. 运行训练好的模型

## 3.1. 训练输出路径

1. 训练脚本会按照 skrl_ppo_cfg.yaml 中的配置将日志和权重保存到：
logs/skrl/ant_direct/日期时间_ppo_torch_.../checkpoints

2. 典型路径示例（实际时间戳会不同）：
logs/skrl/ant_direct/2025-01-01_12-00-00_ppo_torch/checkpoints

## 3.2. 加载 checkpoint 与可视化执行

1. 使用官方预训练 checkpoint 运行

```bash 
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
   --task Isaac-Ant-Direct-v0 \
   --use_pretrained_checkpoint \
   --num_envs 16
```

2. 默认使用最新时间，最大步数的模型

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
   --task Isaac-Ant-Direct-v0 \
   --algorithm PPO \
   --num_envs 16
```

3. 使用自己的 checkpoint 并录屏

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task Isaac-Ant-Direct-v0 \
    --checkpoint logs/skrl/ant_direct/{替换成实际的路径}/checkpoints/agent_{实际step}.pt \
    --video \
    --video_length 500 \
    --num_envs 16  
```

## 3.3. 参数解析

* scripts/reinforcement_learning/skrl/play.py
  skrl 的推理脚本，用于加载训练好的权重并进行可视化运行。

* --task=Isaac-Ant-Direct-v0
  指定要创建的环境名称，与训练阶段一致。

* --num_envs
  指定环境数量，默认是较大的并行数，为了可视化流畅，一般设置为较小值（如 4 或 16）。

* --checkpoint
  可选参数，显式指定要加载的 .pt 权重文件路径；不指定时默认从日志目录中寻找最新权重。

* 不加 --headless
  在推理时不使用 headless，可以看到 Ant 在 Isaac Sim 中的实际运动。

* --video
  开启训练过程的视频录制功能，配合 --video_length 和 --video_interval 使用。

* --video_length
  每段录制视频包含的环境步数，这里设置为 600 步，即大约记录 600 次环境交互。

* --video_interval
  相邻两段视频录制之间的步数间隔，这里设置为 5000，表示每训练 5000 个 step 录制一次视频。

# 4. 与任务相关的文件结构讲解

## 4.1. task 配置文件结构

本任务的核心代码位于 `source/isaaclab_tasks/isaaclab_tasks/direct/ant/` 目录下，主要文件结构如下：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/ant/
├── agents/
│   └── skrl_ppo_cfg.yaml       # PPO 算法超参数配置文件（网络结构、学习率等）
|   └── ...                     # 其他的强化学习配置
├── __init__.py                 # 环境注册入口，将环境 ID 绑定到类和配置
└── ant_env.py                  # Ant 环境的具体实现（配置类 AntEnvCfg + 环境类 AntEnv）
```

此外，通用 locomotion 逻辑位于父目录的 `locomotion/` 中：
```text
source/isaaclab_tasks/isaaclab_tasks/direct/locomotion/
└── locomotion_env.py           # 提供基础的移动任务逻辑（观测拼接、动作应用等）
```

## 4.2. 含 env 实现、奖励函数、观测定义

1. 环境注册与配置入口
   IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ant/__init__.py
   • 注册 Gym 环境 id = "Isaac-Ant-Direct-v0"
   • 指定 env_cfg_entry_point 为 ant_env.AntEnvCfg
   • 指定 skrl_cfg_entry_point 为 agents/skrl_ppo_cfg.yaml

   对应的注册代码大致如下：

   ```python
   import gymnasium as gym
   from . import agents

   gym.register(
       id="Isaac-Ant-Direct-v0",
       entry_point="isaaclab_tasks.isaaclab_tasks.direct.ant.ant_env:AntEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": "isaaclab_tasks.isaaclab_tasks.direct.ant.ant_env:AntEnvCfg",
           "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
           "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
           "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
       },
   )
   ```

   这样，在训练脚本中只要写 --task=Isaac-Ant-Direct-v0，gym.make 就会自动创建 AntEnv，并携带 AntEnvCfg 和 skrl_ppo_cfg.yaml。

2. 环境配置与机器人设置
   IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ant/ant_env.py
   ```
   • AntEnvCfg
     - episode_length_s：每个 episode 的物理时间长度（秒）
     - decimation：每次 RL 决策对应的物理子步数
     - action_scale、action_space、observation_space：动作和观测维度与尺度
     - scene.num_envs：并行环境数量（默认 4096）
     - terrain：地形为平面，摩擦系数等参数在此定义
     - robot（ANT_CFG）：加载 Ant 机器人模型，并设置 prim_path
     - joint_gears：每个关节的力矩缩放系数
     - 各种奖励相关系数：heading_weight、up_weight、actions_cost_scale、energy_cost_scale、alive_reward_scale 等
   • AntEnv
     - 继承自 LocomotionEnv，主要复用通用 locomotion 逻辑。
   ```

   AntEnvCfg 的核心结构大致如下（去掉了一些次要字段）：

   ```python
   @configclass
   class AntEnvCfg(DirectRLEnvCfg):
       # 环境相关
       episode_length_s = 15.0
       decimation = 2
       action_scale = 0.5
       action_space = 8
       observation_space = 36
       state_space = 0

       # 仿真相关
       sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
       terrain = TerrainImporterCfg(
           prim_path="/World/ground",
           terrain_type="plane",
           ...
       )

       # 场景相关
       scene: InteractiveSceneCfg = InteractiveSceneCfg(
           num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
       )

       # 机器人与奖励系数
       robot: ArticulationCfg = ANT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
       joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15]
       heading_weight: float = 0.5
       up_weight: float = 0.1
       energy_cost_scale: float = 0.05
       actions_cost_scale: float = 0.005
       alive_reward_scale: float = 0.5
       dof_vel_scale: float = 0.2
       death_cost: float = -2.0
       termination_height: float = 0.31
   ```

   通过修改这个配置类，可以在不改动环境逻辑代码的情况下快速尝试不同的物理参数和奖励权重。

3. 通用 locomotion 环境逻辑
   IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/locomotion/locomotion_env.py
   • _get_observations
     - 定义 36 维观测的拼接方式（机身姿态、速度、目标方向、关节状态、历史动作）。
   • _apply_action
     - 将 8 维动作乘以 action_scale 和 joint_gears 后，写入关节力矩。
   • _get_rewards 与 compute_rewards
     - 定义前进奖励、存活奖励、姿态奖励、动作和能耗惩罚、关节极限惩罚和死亡惩罚。
   • _get_dones
     - 判断是否因时间耗尽或机身高度低于 termination_height 而终止。

   奖励函数的关键逻辑可以简化理解为：

   ```python
   # 进度奖励：距离目标的负范数 / dt，越靠近目标，potentials 越大
   progress_reward = potentials - prev_potentials

   # 姿态相关奖励
   heading_reward = f(heading_proj)   # 面向目标方向时给予奖励
   up_reward = f(up_proj)             # 机身竖直时给予奖励

   # 成本惩罚
   actions_cost = torch.sum(actions ** 2, dim=-1)              # 动作幅值惩罚
   electricity_cost = torch.sum(torch.abs(actions * dof_vel * dof_vel_scale)
                                 * motor_effort_ratio.unsqueeze(0), dim=-1)
   dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)  # 关节接近极限惩罚

   # 存活奖励
   alive_reward = alive_reward_scale

   total_reward = (
       progress_reward + alive_reward + up_reward + heading_reward
       - actions_cost_scale * actions_cost
       - energy_cost_scale * electricity_cost
       - dof_at_limit_cost
   )

   # 若本步由于跌倒等原因终止，则直接给一个 death_cost
   total_reward = torch.where(reset_terminated,
                              torch.ones_like(total_reward) * death_cost,
                              total_reward)
   ```

   其中 potentials 的计算方式（在 compute_intermediate_values 中）是：
   1. 计算 to_target = targets - torso_position，将 z 分量设为 0，只关心水平距离。
   2. 使用 L2 范数计算距离 d。
   3. 设置 potentials = -d / dt。
   这样，potentials 越大代表越接近目标，progress_reward = potentials - prev_potentials 就可以看作“向目标方向前进的速度”。

## 4.3. 训练脚本

IsaacLab/scripts/reinforcement_learning/skrl/train.py
• 解析命令行参数，包括 --task、--num_envs、--algorithm、--ml_framework 等。
• 通过 gym.make(args_cli.task, cfg=env_cfg) 创建环境，自动根据 env_cfg_entry_point 载入 AntEnvCfg。
• 加载 skrl_ppo_cfg.yaml 作为 agent_cfg，配置模型结构和 PPO 超参数。
• 创建 Runner 并调用 runner.run() 执行训练。
• 将 env.yaml 和 agent.yaml 导出到日志目录，方便复现实验。

## 4.4. PPO 配置文件

IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ant/agents/skrl_ppo_cfg.yaml
• models
  - 定义策略网络和价值网络的层数、隐藏单元和激活函数。
• memory
  - 定义经验存储方式（RandomMemory）和容量。
• agent
  - 选择 PPO 算法，并配置学习率、折扣因子、GAE 参数、clip 范围等。
• trainer
  - 选择 SequentialTrainer，并设置训练步数 timesteps。


# 5. PPO 算法讲解（简明版）

## 5.1. 核心思想

PPO 是一种基于策略梯度的强化学习算法，属于 on-policy 方法。
它通过约束新旧策略之间的变化幅度（使用截断函数 clipping 或 KL 约束），避免每次更新过于激进导致性能崩溃。
相比传统策略梯度，PPO 可以在每次采样后重复使用同一批数据多次更新参数，提高数据利用率。

## 5.2. 为什么适用于机器人控制任务

1. 支持连续动作
   Ant 任务的动作是 8 维连续力矩控制，PPO 通过高斯策略输出连续动作，天然适配。
2. 更新稳定
   通过 ratio_clip（例如 0.2）和损失函数中的 min(surr1, surr2)，限制策略变化幅度，减少训练发散的概率。
3. 数据利用率高
   skrl_ppo_cfg.yaml 中通过 rollouts、learning_epochs 和 mini_batches 设定每次采样后可以多轮更新，提高样本利用效率。

## 5.3. 简单伪代码/流程

1. 初始化策略网络（Actor）和价值网络（Critic）。
2. 在 Isaac-Ant-Direct-v0 环境中运行若干步，收集 (state, action, reward, next_state, log_prob)。
3. 使用 GAE（lambda=0.95，discount_factor=0.99）计算每个时间步的优势函数。
4. 在多轮迭代中，基于同一批数据重复更新：
   - 计算新旧策略概率比 ratio。
   - 构造截断后的目标：min(ratio * advantage, clip(ratio, 1 - 0.2, 1 + 0.2) * advantage)。
   - 计算 Actor 损失和 Critic 损失，并进行反向传播更新参数。
5. 重复以上过程，直到达到设定的训练步数 timesteps。


# 6. 如何设置参数让模型训练得更好

## 6.1. 奖励塑造

1. 调整 heading_weight 和 up_weight
   如果机器人前进速度过慢，可以适当提高 heading_weight；如果容易摔倒，可以提高 up_weight。
2. 调整 actions_cost_scale 和 energy_cost_scale
   如果动作过于剧烈、抖动明显，可以适当增大这两个惩罚系数，鼓励更平滑的动作。
3. 调整 alive_reward_scale 和 death_cost
   若智能体频繁乱跳导致跌倒，可以提高 death_cost 的绝对值，或增加 alive_reward_scale，鼓励“活得更久”。

## 6.2. 动作空间调整

1. 调整 action_scale
   若 Ant 几乎不动，可以略微增大 action_scale；若动作过猛，经常摔倒，则适当减小。
2. 调整 joint_gears
   joint_gears 是每个关节的力矩放大系数，过大容易导致力矩过强、运动不稳定，可先减小再逐步增大。

## 6.3. PPO 参数调节

skrl_ppo_cfg.yaml 中主要可调参数包括：
1. learning_rate
   初始为 3e-4，若训练发散，可减小到 1e-4；若收敛过慢，可略微增大。
2. ratio_clip
   一般在 0.1 到 0.3 之间，越小更新越保守，稳定性更高但学习速度可能变慢。
3. rollouts 和 mini_batches
   增大 rollouts 会提高每次更新使用的数据量，减小梯度方差，但会增加单次迭代时间。
4. entropy_loss_scale
   若策略过早收敛到单一动作模式，可以适当增加熵系数，鼓励更多探索。

## 6.4. 场景随机化

当前 AntEnv 使用平坦地面作为基础场景。
在掌握基础行走后，可以在环境中逐步加入轻微的不平整地面、摩擦系数随机化等，提高策略对真实世界的鲁棒性（需要在 Ant 环境或地形配置中自行扩展）。

## 6.5. num_envs 的重要性

1. 作用
   决定每个时间步可以并行收集多少环境数据，数值越大，策略更新越稳定，训练速度越快。
2. 限制
   直接受 GPU 显存和算力限制，Ant 默认使用 4096 个并行环境，若显存不足可降低为 1024 或 512。
3. 建议
   在不触发 OOM 的前提下，num_envs 越大越好。

常见错误与解决方法（至少 3 条）

1. 错误：运行 train.py 时提示 Unsupported skrl version
   原因：当前安装的 skrl 版本低于脚本要求的最低版本（例如 1.4.3）。
   解决：根据提示升级 skrl，例如运行 pip install "skrl>=1.4.3"。

2. 错误：play.py 无法找到 checkpoint 或提示路径不存在
   原因：尚未完成训练，或路径拼写错误。
   解决：
   - 确认已经成功运行过 train.py，并在 logs/skrl/ant_direct 下生成了日志目录。
   - 使用 --checkpoint 显式指定 .pt 文件的绝对路径。

3. 错误：创建环境时报错 task not found 或类似 Gym 注册错误
   原因：任务名称拼写错误，或未正确导入 isaaclab_tasks。
   解决：
   - 确认命令中使用的任务名完全为 Isaac-Ant-Direct-v0。
   - 使用提供的 train.py / play.py 脚本，这些脚本内部已经导入 isaaclab_tasks。


# 7. 课程总结

任务复现关键点

1. 理解 Isaac-Ant-Direct-v0 环境的观测空间、动作空间以及奖励函数的组成。
2. 熟悉 skrl 训练和推理脚本的基本用法，掌握训练与加载 checkpoint 的命令。
3. 能够根据训练效果，结合 AntEnvCfg 和 skrl_ppo_cfg.yaml 对奖励权重和 PPO 超参数进行合理调节。

