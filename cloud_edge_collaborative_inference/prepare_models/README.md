## 执行代码将MobileNetV2分割为两个小模型

Step 1: 执行`python split_model.py`分割模型，存储为模型文件`mobilenet_v2_shallow.pts`和`mobilenet_v2_deep.pts`

Step 2: 执行`python check_splited_model.py`检查分割模型的正确性，如观察到返回结果`The differences between torch scripts and the original model: tensor(0.)`说明正确无误