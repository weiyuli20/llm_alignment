# 决策树
决策树是一种无参数的机器学习方法，具有很强的解释性和透明度

决策树通过不断地节点分裂将数据划分到不同的空间，整个切分过程可以看作一系列的if-else惠规则

节点分裂指的是根据不纯度度量来选择特征和阈值最大化降低系统的熵

决策树的决策过程基于贪心思想，每次决策基于当前最优


根据不同的度量方式，决策树可以划分为

<img width="843" height="193" alt="image" src="https://github.com/user-attachments/assets/699a40d5-4777-4dae-b0b7-d94cbd6d1ef7" />

Gini = 1 - Σpᵢ²，Entropy = -Σpᵢ·log(pᵢ)
Gini 是熵在 p→0/1 附近的近似，计算略快、对不纯度差异更敏感
Entropy 信息论含义更严谨、对不纯度变化更平滑
实践中两者选出的分裂点几乎一致——所以工业界默认 Gini（省计算），这也是你模型用 Gini 的理由


关于ROC_AUC,PR_AUC曲线
、
<img width="739" height="412" alt="image" src="https://github.com/user-attachments/assets/5a7aa994-84c8-4fa1-9f3b-d7ce9bff12e2" />

ROC-AUC ： 横轴是FP rate 误报率， 正常用户被当成窃电用户； 纵轴是召回率， 窃电用户有多少被分类正确了
PR-AUC： 横轴是Recall, 纵轴是Precision


其中 FP rate = fp/ (fp + tn)
 recall TP rate = tp /(tp+fn)
 精确率 precision  = tp(tp +fp)

ROC_AUC曲线可以看到 误报率取某个值时的召回率。 业务价值是在召回率和误报率之间取一个平衡
PR-AUC 可以看到召回率取某个值时的精确率


在样本分布极度不平衡的场景，主要看PR-AUC， 因为目标样本（窃电用户数量非常少），precison 指标变化会很明显

搭配这两个图：可以分析低误报率时precision也不一定高


boosting 有两个分支：
- adaboost
- GBDT -> XGBoost-> LightGBM

GBDT 通过逐步拟合残差（真实值和预测值的差值）进行学习，直至残差值达到可接受的范围

GBDT的基学习器是决策树
XGBoost 的基学习器可以是决策树或线性分类器

 



