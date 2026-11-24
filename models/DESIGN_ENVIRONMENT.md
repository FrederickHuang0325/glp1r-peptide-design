# ProteinMPNN可选重跑环境

原设计在Google Colab/JAX执行，不是ESM的Windows/PyTorch环境。ColabDesign安装目标为v1.1.1，原JAX/jaxlib、完整依赖、显卡型号和采样随机种子未完整保存。因此不能提供虚构的历史锁定文件，也不能承诺同样的27/5计数。

在单独Linux/Colab环境中按上游v1.1.1说明配置后，可安装指定来源：

```bash
python -m pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
python -m pip freeze
python design.py --mode mpnn --seed 20261003 --out results/new_mpnn
```

`pip freeze`输出应随任何新设计运行保存。该命令为新种子的采样，非原32条逐值复现。新环境的JAX兼容性需要实际测试，当前提交包装测试不安装此环境、不重新训练、不重新下载MPNN权重。安装完成后`model.prep_inputs`长度必须是[30,384]，自由位点必须是全拼接序列的0起算[16,23]，代码会检查。

**评审离线复核不需要安装该环境。** 已归档32条FASTA、6X18结构、原Notebook执行输出摘要、所有AF3原始输出和ESM逐位点分数。`predict.py`复核这些真实材料并重新计算结构指标；新的AI设计与平台推理是独立、资源受限阶段，README已经区分。
