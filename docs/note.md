MSE 评价指标
frame number 前20%帧数作为观测数据，预测后80%的双人动作
关节数 维度
动作标签或者自然语言的条件
用Regennet网络，来处理双人动作预测，前30帧作为条件，生成120帧
用Regennet的train test split 数据来训练和测试