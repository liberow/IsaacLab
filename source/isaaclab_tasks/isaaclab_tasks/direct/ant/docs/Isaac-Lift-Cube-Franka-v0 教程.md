
Isaac-Lift-Cube-Franka-v0


一 、任务介绍

核心目标

使用 Franka 机器人将立方体带到目标位置

观测空间

观测空间为 Box(-inf, inf, (36,), float32)，包含机器人的关节状态 、物体位姿 、 目标信息及历史动作。
1.  关节相对位置（ 9 维）：
。      当前关节角度减去默认关节角度 (default_joint_pos) 。包含 7 个臂关节和 2 个指关节。
2.  关节相对速度（ 9 维）：
。      当前关节速度减去初始速度（初始为0）。
3.  物体位置（ 3 维）：
。     立方体在世界坐标系下的位置 (3 维)

4.  目标位姿（ 7 维）：
。      当前任务指定的目标位置的世界坐标系下位置(3 维) 和姿态(4 维) 。
5.  上一时刻动作（ 8 维）：
。     上一步执行的动作向量

动作空间

动作空间为 Box(-inf, inf, (8,), float32)，前 7 维用于控制机械臂的关节运动 ，最后一维控制夹爪的开闭。

成功条件
机械臂成功将夹取立方体并转移至目标位置处

任务难点与潜在挑战
随机生成立方体初始点和目标点范围较大，机械臂难以到达部分点位

二 、 启动任务训练

训练框架

SKRL，一个开源的强化学习库，采用 Python（基于 PyTorch 和JAX）编写，设计重点在于算法实现的模块化 、可读性 、简洁性和透明度

命令行启动训练示例

Python
./isaaclab.sh -p scripts/reinforcement_learning/sk rl/train.py  --
task=Isaac-Lift-Cube-Frank a-v0 --headless
# 或



python scripts/reinforcement_learning/sk rl/train.py  --task=Isaac-
Lift-Cube-Franka-v0 --headless

关键参数说明
•         ./isaaclab.sh -p ：启用虚拟环境中的 python 解释器 可以用 python 替换
•        scripts/reinforcement_learning/skrl/train.py：skrl 的训练启动脚本
•        --task： 指示任务环境名称，这里是 Isaac-Lift-Cube-Franka-v0
•         --headless ：设置为不可视化运行，节省 GPU 资源

三 、运行训练好的模型
训练好模型后，默认权重保存在 IsaacLab 目录下的：logs/sk rl/franka_lift/日期/checkpoints


加载 checkpoint 与 可视化执行

Python
./isaaclab.sh -p  scripts/reinforcement_learning/sk rl/play.py --
task=Isaac-Lift-Cube-Frank a-v0 --num_envs 20 # 【可选】--checkpoint期望加载.pt 的路径
# 或
python  scripts/reinforcement_learning/sk rl/play.py --task=Isaac- Lift-Cube-Franka-v0 --num_envs 20 # 【可选】--checkpoint  期望加
载.pt 的路径

参数解析
•        scripts/reinforcement_learning/skrl/play.py：skrl 的推理启动脚本
•        --num_envs：指定创建的环境个数，默认是 4096，使用较小数量以流畅运行可视化界面

•        --checkpoint: 指定期望加载的权重路径，可以省略该选项，默认使用最新的最优权重
•         不加--headless 选项， 以进行可视化运行



四 、 与任务相关的文件结构讲解

task 配置文件(含 env 实现 、奖励函数 、观测定义）
•         .\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\ma
ni pulation\lift\lift_env_cfg.py 主要内容包括：

类名	作用
ObjectTableSceneCfg	场景定义
CommandsCfg	随机化目标位置指令
ActionsCfg	定义动作
ObservationsCfg	定义观测
EventCfg	定义重置
RewardsCfg	定义奖励
TerminationsCfg	定义终止
CurriculumCfg	定义课程
LiftEnvCfg	对以上类进行打包，配置强化学习环境和仿真参数
•         IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\mani
pul ation\lift\config\frank a\joint_pos_env_cfg.py主要内容包括：

类名	作用



FrankaCubeLiftEnvCfg	配置 franka机械臂和待抓立方体和可视化标记
FrankaCubeLiftEnvCfg_PLA Y	配置推理时的环境设置

训练脚本
•         IsaacLab\scripts\reinforcement_learning\sk rl\train.py定义了 main 函数:
i.           解析命令行参数并配置 agent_cfg参数
ii.          创建环境 env
iii.         创建 skrl runner
iv.         运行训练

PPO 配置文件
•         IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\mani pul ation\lift\config\frank a\agents\sk rl_ppo_cfg.yaml
a.        定义模型结构
i.           策略 policy 结构
ii.          价值 value 模型结构
b.        定义数据回放池
▪   memory
c.         定义 agent
▪  选择强化学习算法如 PPO
▪   配置算法相关的参数
d.        定义训练器
▪  选择训练器类型
▪  规定训练步数

五 、 PPO 算法讲解（简明版）

核心思想





PPO 是一种基于策略梯度的强化学习算法 。它的核心思想是通过一个‘截断函数
(Clipping Function)’来限制新旧策略之间的差异，防止模型因为更新步长过大而导致训练崩溃 。这保证了学习过程既能持续上升，又足够安全稳定。


为什么适用于机器人控制任务
•         输出值连续
a.         PPO 可以直接输出连续的数值（比如：关节 1 输出 2.5 牛顿的力，关节 2旋转 30.12 度）。
b.        契合物理世界中力 、角度 、速度的连续性
•         稳定更新
a.         PPO 的截断 (Clipping) 机制强制要求“新策略不能偏离旧策略太远”
b.        保证了机器人的策略更新前后动作变化的渐进性 、平稳性
•         数据利用率高
a.        相比于早期的策略梯度算法（跑一次数据就扔） ，PPO 允许利用同一批数据进行多次更新
b.        使得昂贵的机器人数据利用更高效

简单伪代码/流程

Python
Actor = Network()  # 演员：负责控制机械臂动作Critic = Network() # 评论家：负责给动作打分 for episode in range(总训练次数):
memory = []
state = env.reset()
for step in range(rollouts_num): # 采样一个批次数据
action, old_log_prob = Actor.get_action(state)      # 1.
采样动作
next_state, reward, done = env.step(action)         # 2.
执行动作
memory.store(state, action, old_log_prob, reward)   # 3.记录数据





state = next_state
if done: break
return, advantages = calculate_gae(memory, Critic)
batch_data = memory.get_batches(returns, advantages)
for _ in range(K_epochs): # 拿着收集到的数据，反复学 K 遍(Epochs)
# 批量训练：从 Batch_data 中取出 Mini-batch
for states, actions, old_log_probs, advantages_batch,
returns_batch in batch_data:
# 1. 策略网络计算（Actor Loss）
new_log_probs = Actor.evaluate(states, actions)
ratio = exp(new_log_probs - old_log_probs)
# PPO 的灵魂：计算被截断的损失函数
surr1 = ratio * advantages_batch
surr2 = clip(ratio, 1 - 0.2, 1 + 0.2) *
advantages_batch
actor_loss = -mean(min(surr1, surr2)) # 对整个批次求平均
# 2. 价值网络计算（Critic Loss）
# 获取 Critic 对当前状态的预测价值
current_values = Critic.evaluate(states)
target_values = returns_batch # 目标是 GAE 计算出的回报value_loss = mean((target_values - current_values) **
2)
# 3. 总损失 = Actor Loss + Critic Loss + 熵奖励 (可选)分开更新模型，或使用权重平衡
total_loss = actor_loss + 0.5 * value_loss
# 4. 更新网络
Optimizer.update(total_loss) # 一次性对两个网络进行反向传播和更新
memory.clear() # 训练完一轮，清空内存，准备下一轮








六 、如何设置参数让模型训练得更好

奖励塑造


动作空间调整
•        机械臂关节： 策略输出 Action 为初始位置的偏移量（ use_default_offset=True）
•         夹爪动作： 为二元离散动作（开/合夹爪）

PPO 参数调节






场景随机化
•        初始调试： 如果训练不收敛，应减小立方体和目标位置的随机范围，降低难度 。 •         进阶训练： 待模型收敛后，再逐步增大随机范围， 以提高泛化能力和真实世界迁移性。
num_envs 的重要性
•        核心作用： 决定数据采集速度，越大则策略梯度方差越小，训练越高效稳定。
•         限制因素： 直接消耗 GPU 显存和计算资源。
•        选择原则： 在保证 GPU 显存不溢出的前提下，尽量取最大值。

课程总结

任务复现关键点
•         了解 Isaaclab 环境的基本操作和强化学习的相关代码文件与命令参数
•         了解本任务环境的观测空间 、动作空间 、奖励函数