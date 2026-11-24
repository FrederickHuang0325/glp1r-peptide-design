# GLP-1R激动肽结构约束设计与计算筛选

赛道：AI大分子与多肽药物设计。提交版本：1.0.0。

本项目基于6X18复合物，以ProteinMPNN进行两位点受限序列采样，利用AlphaFold Server输出复核结合姿态和界面，再用ESM-2评估序列兼容性。本包包括真实原始输入、12个AF3任务共60个模型、逐位点ESM结果、源代码、运行说明、模型卡和验证记录。

**未开展训练或微调。** 不是从零训练模型或大规模从头生成。Q23A及双突变是人工构建的比较对照，不是原32条ProteinMPNN采样输出。未进行MD、自由能、溶解性模型、cAMP/β-arrestin活性预测或湿实验反馈训练。教学模拟细胞/动物数据不包含在本代码包，也不作为实验验证证据。

## 1 评委快速运行

需要Python 3.11.9和NumPy 2.4.6。Windows PowerShell或Linux终端均可。先解压进入包含本README的`project`目录。

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py --out results/reproduced
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux对应命令：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python predict.py --out results/reproduced
.venv/bin/python -m unittest discover -s tests -v
```

也可在已有环境直接运行`python predict.py --out results/reproduced`。若输出目录已非空，程序拒绝覆盖，请改用新的输出目录。程序不自动安装依赖、不下载权重、不访问AF3服务。首次`pip install`本身需要网络；运行阶段可离线。

此默认命令执行：输入SHA256核验 → 原MPNN采样约束核验 → 两位点四种组合枚举 → 从AF3 ZIP重新解析并计算60个模型指标 → 核验并回放原ESM逐位点分数 → 生成候选清单。它不在CPU上重新执行ProteinMPNN、AF3或ESM神经网络，不能将其日志当作新的AI推理日志。

主输出`results/reproduced/results.csv`采用UTF-8 BOM，包含候选编号、赛道、序列、结构文件、结构与ESM指标、来源、模型版本、验证优先级、备注。`NA`表示本项目没有相应亲和力或功能预测，不能解释为0。`results/example/results.csv`是随附的实测复核示例。若大赛后续发布专门候选清单模板，应按该模板映射字段，不改变原数值。

## 2 目录

```text
project/
  README.md                  环境和运行说明
  config.json                序列、链编号、阈值、种子和来源
  requirements.txt           CPU复核的锁定依赖
  requirements-esm.txt       本地ESM推理附加依赖
  design.py                  设计归档复核、组合枚举或可选新MPNN采样
  prepare_af3.py             导出平台输入JSON，不自动提交任务
  screen.py                  重新计算结构指标
  predict.py                 一键生成最终清单
  src/                       核心代码
  tests/                     单元测试与复核脚本
  data/                      6X18、原FASTA、AF3 ZIP、原ESM表与数据说明
  models/                    模型调用、权重获取方式和Model Card
  notebooks/                 可执行复核示例及分阶段调用说明
  results/example/           已验证输出、60个CIF及代表结构路径
  logs/                      原始运行证据摘要、当前环境与验证报告
```

## 3 环境与资源

### CPU复核

参考配置为Windows 64位、Python 3.11.9、NumPy 2.4.6，不需要显卡；建议8 GB以上内存并预留1 GB工作空间。完整60模型核查耗时以`logs/validation_report.json`中的本次实测秒数为准，其他机器会不同。

### ESM重新推理

已用环境：Python 3.11.9、PyTorch 2.9.1+cu126、fair-esm 2.0.0、NumPy 2.4.6、NVIDIA GeForce RTX 4060 Laptop GPU。CUDA运行时12.6。显卡驱动等当前检查记录见`logs/environment.json`；这不是对历史运行环境的回填。建议16 GB系统内存、6 GB以上可用显存；代码逐位点、batch=1、float32运行，显存占用以实际机器为准，也允许显式`--device cpu`，但会更慢。

在独立环境安装，**不要覆盖已有Colab/JAX环境**：

```bash
python -m pip install -r requirements.txt fair-esm==2.0.0
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
```

CPU-only复算可改用PyTorch官方CPU安装索引。`requirements-esm.txt`锁定基础版本；cu126变体需上面的官方索引命令。权重路径通过参数传入，不写死个人盘符。

准备两份可信官方权重，见`models/README.md`，然后执行：

```bash
python predict.py --out results/new_ESM --esm-mode infer --checkpoint models/esm2_t33_650M_UR50D.pt --device cuda
```

评分执行30个位点逐一掩码，4条序列共120次前向计算；对全模型词表做log-softmax，对真实残基的自然对数概率求和/取平均。BOS/EOS不计分，酰胺化无ESM编码。种子17、eval、inference_mode、float32、TF32关闭。不同硬件/软件内核可能导致末位浮点差异。与归档ESM分数的核查容差为1e-4；不添加人为扰动或复制重复值构造显著性。

## 4 设计阶段及完整分阶段复现

### 4.1 归档设计和确定性优化对照

```bash
python design.py --mode archive --out results/design_audit
python design.py --mode enumerate --out results/design_enumeration
```

原MPNN设置：ColabDesign安装目标v1.1.1，ProteinMPNN权重v_48_020；输入6X18.pdb；链P,R；`fix_pos=P23,P30`、`inverse=True`；只允许A/Q；32条，温度0.1。P23/P30为作者编号，分别对应肽局部17/24。受体384个有坐标残基保持原序列/骨架，其余28个肽位点固定。原输出为27条A30Q和5条WT，去重后仅两条。局部17/24可选A/Q，因此整个受限空间最多四条；Q23A和双突变人工补齐为比较对照。

原采样随机种子、JAX版本及Colab硬件没有完整保存，不能承诺重新得到完全相同的27/5计数。原Notebook保存的关键输出与源码哈希见`logs/original_design_evidence.json`，原32条FASTA随包保留。原Notebook后续可选AlphaFold细胞未保存执行记录，实际结构分析使用独立的AlphaFold Server，不声称执行过Notebook内可选预测。

### 4.2 可选的新ProteinMPNN采样

`notebooks/02_design_and_server_workflow.ipynb`提供分阶段入口。在Linux/Colab独立环境按`models/DESIGN_ENVIRONMENT.md`配置ColabDesign后：

```bash
python design.py --mode mpnn --seed 20261003 --out results/new_mpnn
```

这是新包装的重跑入口，保存真实耗时、种子和采样序列；不冒充原历史采样。当前交付测试没有安装JAX或执行这一步，可能需要处理旧ColabDesign与当前JAX的兼容性，不能把“有入口”写成“已经重新验证”。参数约束及输入长度在代码中检查，防止误设计受体。新采样若出现其他两种序列，要标注为新MPNN来源，不覆盖本次提交的历史来源。

### 4.3 AlphaFold Server推理

```bash
python prepare_af3.py --out results/server_inputs
```

生成12个原任务请求JSON及一个合并JSON，手动上传AlphaFold Server，逐项检查实体和种子，再手动下载返回ZIP。该服务无本项目可调用的已验证批量API，代码不自动操控账户或提交任务。本包已保存12个实际返回ZIP，复核不需重新占用服务配额。第三方服务器升级后无法保证新结果逐数一致。

实际输入：440 aa受体构建体（由6X18序列核对，不是未经改造的完整天然受体声明）+30 aa肽，各1份；种子17/29/43；每任务5模型；两条链`useStructureTemplate=false`；无G蛋白、无膜、无显式C端酰胺化。6X18是**MPNN设计与后续比对的参考**，不是AF3中的固定模板。AF3未固定受体骨架。JSON的`version=3`是本批实际保存的请求格式，不是AF3神经网络版本。神经网络准确构建版本未披露，必须如实保留。

### 4.4 筛选及最终清单

```bash
python screen.py --out results/structure_review
python predict.py --out results/full_review
```

重新推理ESM时使用第3节的`--esm-mode infer`命令。由此完整流程为：受限AI设计（第三方Colab/JAX）→人工补齐比较组合→平台结构推理→本地结构分析和序列兼容性评分→候选清单。默认离线演示通过已保存AI输出复核这条流程，而非新调用全部模型。

## 5 指标定义和排序边界

- 参考PDB作者链R为受体、P为肽；肽GLP编号7–36。R440预测局部位置j对应参考作者位置j+23，逐个有坐标残基校验氨基酸身份；不适用于任意受体构建体，映射不一致立即停止。
- 受体作者140–330的191个共同CA做Kabsch拟合，对肽应用同一个变换。肽整体CA RMSD为30位，N端为GLP7–15的9位；不独立拟合肽。
- 接触为非H/D、正占有率原子距离≤5 Å的残基对。6X18参考109对，只使用双方都可比较的384个参考受体残基。未解析残基不当作零接触。
- 固定界面PAE选择6X18的55个界面受体残基×全部30个肽位点，保留受体→肽和肽→受体两个块的平均，再取双向均值；不假定PAE矩阵对称。它不是全局PAE，也不是平台`chain_pair_pae_min`。
- 肽pLDDT使用CIF中每残基CA的B列平均，非全原子加权平均；ipTM如平台提供则保留链对双向值，缺失记NA。
- GRAVY为Kyte–Doolittle序列均值，只是描述符，不能当作实测溶解度。
- ESM分数是短肽的探索性伪对数似然，不是活性、亲和力、通路偏向或稳定性预测。
- 每组报告15模型中位数、范围和3种子中位数；15个模型不是生物学重复，不做显著性检验。
- 默认候选清单A30Q与Q23A为同优先级1，双突变为补充优先级2，WT为对照0。优先级是公开的验证安排，非AI效能排名；同级按照固定名称顺序排列，不由微小PAE/RMSD差异决定药效优劣。
- 清单代表结构取肽RMSD最接近本组中位数的模型，并以种子/模型编号打破平局；全部60个原CIF也保留。

## 6 数据与防泄漏

数据来源、访问/归档时间和使用约束见`data/README.md`、`data/input_manifest.json`、`THIRD_PARTY_NOTICES.md`与Model Card。没有监督训练集或测试集划分，没有独立泛化性能结论，未接触赛事隐藏测试集。设计序列精确去重，AF3任务按完整序列/种子去重，60模型键唯一；代码阻止重复任务。公开训练集与GLP家族/PDB的潜在重叠不能完全排除，特别是6X18可能在预训练范围内；模板关闭不等于没有训练泄漏，不声称这是陌生靶点泛化测试。

Propedia曾列入研究计划，但本提交核心复核只用6X18及实际AF3/ESM输出，没有将未取得的Propedia界面或未收集的功能标签写成训练数据。数据获取时间无法从原文件精确恢复时标为未记录，当前归档时间不能替代历史获取时间。

## 7 原创贡献与真实性

项目实际贡献是GLP-1R任务的设计约束、母本及链编号核验、对照构建、序列/结构综合复核、候选验证安排和可复现工程封装。ProteinMPNN、AF3和ESM-2架构/权重不是本团队原创。主分析入口由原Colab/终端流程整理重构，已与原逐模型数值进行回归核对；整理期间使用AI辅助编程，参赛团队应阅读代码并承担结果真实性与提交责任。

没有提交`train.py`：附件5明确未训练/微调项目不强制训练入口，本项目不使用占位训练脚本伪装训练。MD、细胞信号偏向及动物“减重质量”都不由此包证明。

## 8 核验和提交检查

```bash
python -m unittest discover -s tests -v
python tests/verify_regression.py --results results/example
```

第一条验证CA拟合、参考接触、位点与ESM数据一致性；第二条将60结构重新计算值与原Colab CSV、种子表以及ESM表匹配，输出偏差报告。当前验证结果见`logs/validation_report.json`。

提交前阅读`SUBMISSION_CHECKLIST.md`。压缩包中不含2.6 GB的ESM权重，附件5允许提交模型调用/获取说明，下载URL和SHA256已经列明。本包是当前真实计算阶段提交，后续若加入真实湿实验、MD或模型迭代，需要新增原始数据和可运行代码，不应直接更改结论标签。
