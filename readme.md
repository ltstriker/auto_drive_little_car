项目结构：

logs:用于tensorboard的记录，本文件内的内容不同步到github中
models:存放网络训练后的模型

config.yml:配置网络，小车等分辨率的选项

manage_xxx.py:xxx网络运行的py文件

xxx_readme.md文件:运行xxx.py文件的使用说明


python manage2.py train --tub=carTrain/data/set1/tub_16_19-04-20 --model=lt-test

sudo python3 manage2.py drive
sudo python3 manage2.py calibrate --channel 1
sudo python3 manage2.py calibrate --channel 0


