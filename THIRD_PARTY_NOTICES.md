# 第三方工具与来源

本包不复制或重新授权上游模型源码/权重；项目封装代码仅调用已安装第三方库。公开依赖应按其原许可使用。原团队代码的公开授权/专利安排由参赛团队决定，本包不代团队作新的开源授权。

| 项目 | 实际用途 | 版本或参数 | 来源与许可入口 |
|---|---|---|---|
| ColabDesign | ProteinMPNN JAX接口 | 原安装目标v1.1.1 | https://github.com/sokrypton/ColabDesign/tree/v1.1.1 ，按对应组件许可；未为未核实许可贴标签 |
| ProteinMPNN | 受限两位点采样 | v_48_020 | https://github.com/dauparas/ProteinMPNN ，原模型及许可按上游 |
| AlphaFold Server | 两链结构预测 | 请求格式v3，模型构建版本未披露 | https://alphafoldserver.com/ ，https://alphafoldserver.com/output-terms |
| ESM-2/fair-esm | 掩码序列兼容性 | esm2_t33_650M_UR50D/fair-esm2.0.0 | https://github.com/facebookresearch/esm ，MIT代码许可 https://github.com/facebookresearch/esm/blob/main/LICENSE |
| PyTorch | ESM推理 | 2.9.1+cu126实际运行 | https://pytorch.org/ ，https://github.com/pytorch/pytorch/blob/main/LICENSE |
| NumPy | 数值计算 | 2.4.6提交环境 | https://numpy.org/ ，BSD-3-Clause https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| Biopython | 原Colab CA拟合及序列操作 | 原会话版本未保存 | https://biopython.org/ ，https://github.com/biopython/biopython/blob/master/LICENSE.rst ，提交管线使用NumPy重构并回归核对 |
| RCSB PDB | 6X18输入与参照 | 原下载日期未记录 | https://www.rcsb.org/structure/6X18 ，https://www.rcsb.org/pages/policies |

本提交默认核心只依赖NumPy，JAX/ColabDesign、PyTorch/fair-esm、Jupyter分别为可选阶段，避免把两个历史环境错误合并成已验证的单一环境。模型获取和使用条款以官方最新规定为准；本包的许可说明不是法律意见。
