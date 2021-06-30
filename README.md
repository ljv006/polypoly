<!--
 * @Author: 龙嘉伟
 * @Date: 2021-06-30 15:49:52
 * @LastEditors: 龙嘉伟
 * @LastEditTime: 2021-06-30 15:52:24
 * @Description: 
-->
# polypoly中文多音字注音模型
### 在开源数据上训练Bi-LSTM进行实体标注，实现多音字注音。
## 更新说明
### 2021-06-30
- 基本注音功能。
- 将模型打包成wheel格式，使用pip进行安装。

## 使用说明
  1. 调用get_wheel.sh生成安装文件，在dist目录下；
  2. 使用pip install XXX.whl文件；
  3. 在python中使用import polypoly引入注音包；
  4. 使用polypoly.predict进行注音。