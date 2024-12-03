### 报告见report.pdf
### 程序见src/edge.py
### 使用说明如下

直接运行，会得到三张以(100, 40)作为阈值参数的图片

可调参数：

阈值：修改line:116的high及下方的low

算子：

- Prewitt：注释line:33-34, 取消注释line:37-38

- Canny: 注释line:33-53, 取消注释line:56-71

cnt: 修改line:145的数值即可

### 其他

out文件存储latex生成的pdf等文件，pdf在主文件夹下有一个一样的副本

src文件夹存储python程序，其他文件夹存储制作报告时使用到的图片等。