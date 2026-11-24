# 模型权重与调用

本包不含巨型模型权重。附件5允许“权重或调用方式”；本包通过现有官方预训练模型及下载/调用说明实现这一项。

## ESM-2

文件名：`esm2_t33_650M_UR50D.pt`（约2.6 GB）。官方获取地址：

https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

辅助文件：`esm2_t33_650M_UR50D-contact-regression.pt`：

https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt

本次实际权重大小及SHA256见`weights_manifest.json`，评分入口自动校验。两个文件应放同一目录。禁止使用曾经下载不完整的`.broken_*`文件。旧格式权重需pickle反序列化，只允许可信官方、且SHA256与清单匹配的权重；不要加载陌生文件。程序不自动下载。

模型来源：https://github.com/facebookresearch/esm 。fair-esm代码MIT许可；模型权重使用遵守官方相关声明，项目不重新授权上游内容。

## ProteinMPNN

原设计通过ColabDesign安装目标v1.1.1使用随安装包提供的`v_48_020` JAX权重。上游：

https://github.com/sokrypton/ColabDesign/tree/v1.1.1

https://github.com/dauparas/ProteinMPNN

本包没有重新分发ColabDesign源码或转换权重；包装入口调用上游库。原执行Notebook关键参数和输出日志摘要在`logs/original_design_evidence.json`。旧版JAX依赖需要单独复现环境，见`DESIGN_ENVIRONMENT.md`。

## AlphaFold Server

使用 https://alphafoldserver.com/ 手动输入/上传JSON，真实返回ZIP在`data/af3`。原请求格式版本3不代表AF3模型版本。当前包没有AF3源代码或私有服务器权重，也不声称可在本地重新运行AF3。平台输入/输出说明：

https://github.com/google-deepmind/alphafold/blob/main/server/README.md

输出使用受平台条款约束：https://alphafoldserver.com/output-terms 。用于赛事学术复核，不将输出或上游许可重新声明为项目独占；商业使用应另行核查条款。
