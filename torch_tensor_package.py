import torch as t


#torch.Tensor是一种包含单一数据类型元素的多维矩阵。


# Torch定义了七种CPU tensor类型和八种GPU tensor类型：
#
# Data type	                CPU tensor	        GPU tensor
# 32-bit floating point	    torch.FloatTensor	torch.cuda.FloatTensor
# 64-bit floating point	    torch.DoubleTensor	torch.cuda.DoubleTensor
# 16-bit floating point	    N/A	                torch.cuda.HalfTensor
# 8-bit integer (unsigned)	torch.ByteTensor	torch.cuda.ByteTensor
# 8-bit integer (signed)	torch.CharTensor	torch.cuda.CharTensor
# 16-bit integer (signed)	torch.ShortTensor	torch.cuda.ShortTensor
# 32-bit integer (signed)	torch.IntTensor	    torch.cuda.IntTensor
# 64-bit integer (signed)	torch.LongTensor	torch.cuda.LongTensor

# torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。

#每一个张量tensor都有一个相应的torch.Storage用来保存其数据。类tensor提供了一个存储的多维的、横向视图，并且定义了在数值运算。

#注意： 会改变tensor的函数操作会用一个下划线后缀来标示。比如，torch.FloatTensor.abs_()会在原地计算绝对值，并返回改变后的tensor，而tensor.FloatTensor.abs()将会在一个新的tensor中计算结果。

t.cpu()             #如果在CPU上没有该tensor，则会返回一个CPU的副本
#t.cuda(device=None, async=False)   #返回此对象在CPU内存中的一个副本 如果对象已近存在与CUDA存储中并且在正确的设备上，则不会进行复制并返回原始对象。参数： - device(int)-目的GPU的id，默认为当前的设备。 - async(bool)-如果为True并且资源在固定内存中，则复制的副本将会与原始数据异步。否则，该参数没有意义。


#Storage
# byte()          将此存储转为byte类型
# char()          将此存储转为char类型
# clone()         返回此存储的一个副本
# half()          将此存储转为half类型
# int()           将此存储转为int类型
# long()          将此存储转为long类型