# d2l-learn

依据《动手学深度学习》（Dive into Deep Learning，d2l）逐步复现关键概念和代码的学习仓库。

## 学习目标

- 从张量、数据处理、自动微分和线性模型开始，按实际学习顺序逐步推进。
- 理解每个关键步骤的数学含义、Python语法、张量形状和运行结果。
- 每次实现一个明确的学习单元，完成运行与检查，再记录结论和提交代码。

## 当前状态

仓库刚完成初始化，仅包含README.md和空的.gitignore。尚未添加练习代码、数据或实验结果。具体从哪一节开始，由后续学习对话确定。

本仓库独立于CIFAR-10项目，不自动复制该项目的baseline，也不按其原12阶段迁移计划推进。

## 本机使用

正式目录位于WSL Ubuntu-24.04：

```bash
conda activate cv-week1
cd /home/hao/research/00-getting-start/d2l-learn
code .
```

目前可复用已有cv-week1环境；是否增加依赖，由具体练习决定，不默认安装或降级d2l包。

远端仓库名称：research-isagoge-d2l-learn。远端创建与首次推送的实际状态以交接文档和GitHub为准。

## 组织方式

后续按学习章节逐步增加目录和代码，不提前建立整套实现。优先使用可直接运行的小例子，再整理为可复用模块。命令、输入、输出和结论随练习补充。

.gitignore按当前要求保持空文件。以后生成缓存、数据或模型权重时，先按需要补充忽略规则，再选择性暂存，避免把无关文件一起提交。

## 参考资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [Dive into Deep Learning](https://d2l.ai/)
- [PyTorch官方教程](https://docs.pytorch.org/tutorials/)
