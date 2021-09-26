# PSPNet_ccpd

# 环境配置

python == 3.8

tensorflow == 2.4.1

keras == 2.4.3

如果想切换到1.6.0的tensoflow版本，匹配python==3.6，需要在代码这个tf命令中进行修改：

tf.image.resize(x, (K.int_shape(y)[1], K.int_shape(y)[2]))改成tf.image.resize_images

# 运行

直接运行main.py。

1、ccpd_crop文件夹：原始数据集，内含3116张图片，单目标检测问题。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

2、demo文件夹：test_data的效果演示图，可以看出训练出来的非常不错，精度接近100%。

3、download_weights.h5：迁移学习，网上下载backbone的部分权重。

4、mobile_netv2.py：特征提取网络，PSPNet的backbone部分。

5、psp_model.py：PSPNet整体结构。

6、best_val0.02110.h5：自己训练出来的权重，语义分割效果理想。
