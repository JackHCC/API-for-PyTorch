from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

#Parameters
#class torch.nn.Parameter()
# Parameters 是 Variable 的子类。Paramenters和Modules一起使用的时候会有一些特殊的属性，即：当Paramenters赋值给Module的属性的时候，他会自动的被加到 Module的 参数列表中(即：会出现在 parameters() 迭代器中)。将Varibale赋值给Module属性则不会有这样的影响。 这样做的原因是：我们有时候会需要缓存一些临时的状态(state), 比如：模型中RNN的最后一个隐状态。如果没有Parameter这个类的话，那么这些临时变量也会注册成为模型变量。
# Variable 与 Parameter的另一个不同之处在于，Parameter不能被 volatile(即：无法设置volatile=True)而且默认requires_grad=True。Variable默认requires_grad=False。
# 参数说明:data (Tensor) – parameter tensor.          requires_grad (bool, optional) – 默认为True，在BP的过程中会对其求微分。


#Containers（容器）
#class torch.nn.Module

#你的模型也应该继承这个类。Modules也可以包含其它Modules,允许使用树结构嵌入他们。你可以将子模块赋值给模型属性。
from torch import autograd
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)                # submodule: Conv2d
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.add_module("conv", nn.Conv2d(10, 20, 4))    #将一个 child module 添加到当前 modle。 被添加的module可以通过 name属性来获取。
        #self.conv = nn.Conv2d(10, 20, 4) 和上面这个增加module的方式等价
model = Model2()
print(model.conv)

for sub_module in model.children():     #children()返回当前模型子模块的迭代器。
    print(sub_module)


#eval()     将模型设置成evaluation模式,仅仅当模型中有Dropout和BatchNorm是才会有影响。
#load_state_dict(state_dict)    将state_dict中的parameters和buffers复制到此module和它的后代中。state_dict中的key必须和 model.state_dict()返回的key一致。 NOTE：用来加载模型参数。
#modules()      返回一个包含 当前模型 所有模块的迭代器。
#named_children()       返回包含模型当前子模块的迭代器，yield 模块名字和模块本身。
#parameters(memo=None)  返回一个包含模型所有参数的迭代器。一般用来当作optimizer的参数。

#register_buffer(name, tensor)  persistent buffer通常被用在这么一种情况：我们需要保存一个状态，但是这个状态不能看作成为模型参数。 例如：, BatchNorm’s running_mean 不是一个 parameter, 但是它也是需要保存的状态之一。

#class torch.nn.Sequential(* args)      一个时序容器。Modules 会以他们传入的顺序被添加到容器中。当然，也可以传入一个OrderedDict。
# Example of using Sequential
model1 = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
# Example of using Sequential with OrderedDict
model2 = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

#class torch.nn.ModuleList(modules=None)[source]        ModuleList可以像一般的Python list一样被索引。而且ModuleList中包含的modules已经被正确的注册，对所有的module method可见。

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed         using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

#class torch.nn.ParameterList(parameters=None)      ParameterList可以像一般的Python list一样被索引。而且ParameterList中包含的parameters已经被正确的注册，对所有的module method可见。
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x

#卷积层CNN

#一维
#class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# Parameters：
# in_channels(int) – 输入信号的通道
# out_channels(int) – 卷积产生的通道
# kerner_size(int or tuple) - 卷积核的尺寸
# stride(int or tuple, optional) - 卷积步长
# padding (int or tuple, optional)- 输入的每一条边补充0的层数
# dilation(int or tuple, `optional``) – 卷积核元素之间的间距
# groups(int, optional) – 从输入通道到输出通道的阻塞连接数
# bias(bool, optional) - 如果bias=True，添加偏置

m = nn.Conv1d(16, 33, 3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)
print(output)

#二维
#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# Parameters：
# in_channels(int) – 输入信号的通道
# out_channels(int) – 卷积产生的通道
# kerner_size(int or tuple) - 卷积核的尺寸
# stride(int or tuple, optional) - 卷积步长
# padding(int or tuple, optional) - 输入的每一条边补充0的层数
# dilation(int or tuple, optional) – 卷积核元素之间的间距
# groups(int, optional) – 从输入通道到输出通道的阻塞连接数
# bias(bool, optional) - 如果bias=True，添加偏置

m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = autograd.Variable(torch.randn(20, 16, 50, 100))
output = m(input)
#二维常用于图像处理


#三维
#class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# Parameters：
# in_channels(int) – 输入信号的通道
# out_channels(int) – 卷积产生的通道
# kernel_size(int or tuple) - 卷积核的尺寸
# stride(int or tuple, optional) - 卷积步长
# padding(int or tuple, optional) - 输入的每一条边补充0的层数
# dilation(int or tuple, optional) – 卷积核元素之间的间距
# groups(int, optional) – 从输入通道到输出通道的阻塞连接数
# bias(bool, optional) - 如果bias=True，添加偏置

m = nn.Conv3d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
output = m(input)


#解卷积

#一维
#class torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
#二维
#class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
#三维
#torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)

# With square kernels and equal stride
m = nn.ConvTranspose2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = autograd.Variable(torch.randn(20, 16, 50, 100))
output = m(input)
# exact output size can be also specified as an argument
input = autograd.Variable(torch.randn(1, 16, 12, 12))
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
h.size()
torch.Size([1, 16, 6, 6])
output = upsample(h, output_size=input.size())
output.size()
torch.Size([1, 16, 12, 12])

#池化层

#class torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# kernel_size(int or tuple) - max pooling的窗口大小
# stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
# padding(int or tuple, optional) - 输入的每一条边补充0的层数
# dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
# return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
# ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

m = nn.MaxPool1d(3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)

#class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
#kernel_size(int or tuple) - max pooling的窗口大小
# stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
# padding(int or tuple, optional) - 输入的每一条边补充0的层数
# dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
# return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
# ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

m = nn.MaxPool2d(3, stride=2)
# pool of non-square window
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = autograd.Variable(torch.randn(20, 16, 50, 32))
output = m(input)


#class torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# kernel_size(int or tuple) - max pooling的窗口大小
# stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
# padding(int or tuple, optional) - 输入的每一条边补充0的层数
# dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
# return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
# ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

# pool of square window of size=3, stride=2
m = nn.MaxPool3d(3, stride=2)
# pool of non-square window
m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
output = m(input)


#class torch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
#Maxpool1d的逆过程，不过并不是完全的逆过程，因为在maxpool1d的过程中，一些最大值的已经丢失。 MaxUnpool1d输入MaxPool1d的输出，包括最大值的索引，并计算所有maxpool1d过程中非最大值被设置为零的部分的反向。

#class torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)

#class torch.nn.MaxUnpool3d(kernel_size, stride=None, padding=0)


#class torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
#对信号的输入通道，提供1维平均池化（average pooling ）
# kernel_size(int or tuple) - 池化窗口大小
# # stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
# # padding(int or tuple, optional) - 输入的每一条边补充0的层数
# # dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
# # return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
# # ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

#class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

#class torch.nn.AvgPool3d(kernel_size, stride=None)


#class torch.nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
#对输入的信号，提供2维的分数最大化池化操作 分数最大化池化

#class torch.nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)
#对输入信号提供2维的幂平均池化操作

#class torch.nn.AdaptiveMaxPool1d(output_size, return_indices=False)
#对输入信号，提供1维的自适应最大池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H，但是输入和输出特征的数目不会变化。

#class torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
#对输入信号，提供2维的自适应最大池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。

#class torch.nn.AdaptiveAvgPool1d(output_size)
#对输入信号，提供1维的自适应平均池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。

#class torch.nn.AdaptiveAvgPool2d(output_size)
#对输入信号，提供2维的自适应平均池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。


#Non-Linear Activations


#激励函数

#relu

#sigmoid

#tanh

#softplus


#Normalization layers

#class torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True)
#对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作

# With Learnable Parameters
m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = autograd.Variable(torch.randn(20, 100))
output = m(input)

#class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
#对小批量(mini-batch)3d数据组成的4d输入进行批标准化(Batch Normalization)操作

#class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True)
#对小批量(mini-batch)4d数据组成的5d输入进行批标准化(Batch Normalization)操作


#Recurrent layers

#class torch.nn.RNN( args, * kwargs)[source]            将一个多层的 Elman RNN，激活函数为tanh或者ReLU，用于输入序列。

# input_size – 输入x的特征数量。
# hidden_size – 隐层的特征数量。
# num_layers – RNN的层数。
# nonlinearity – 指定非线性函数使用tanh还是relu。默认是tanh。
# bias – 如果是False，那么RNN层就不会使用偏置权重 $b_ih$和$b_hh$,默认是True
# batch_first – 如果True的话，那么输入Tensor的shape应该是[batch_size, time_step, feature],输出也是这样。
# dropout – 如果值非零，那么除了最后一层外，其它层的输出都会套上一个dropout层。
# bidirectional – 如果True，将会变成一个双向RNN，默认为False。

rnn = nn.RNN(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)

#class torch.nn.LSTM( args, * kwargs)[source]
# input_size – 输入的特征维度
# hidden_size – 隐状态的特征维度
# num_layers – 层数（和时序展开要区分开）
# bias – 如果为False，那么LSTM将不会使用$b_{ih},b_{hh}$，默认为True。
# batch_first – 如果为True，那么输入和输出Tensor的形状为(batch, seq, feature)
# dropout – 如果非零的话，将会在RNN的输出上加个dropout，最后一层除外。
# bidirectional – 如果为True，将会变成一个双向RNN，默认为False。

lstm = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = lstm(input, (h0, c0))

#class torch.nn.GRU( args, * kwargs)[source]
rnn = nn.GRU(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
output, hn = rnn(input, h0)

#class torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')[source]
#一个 Elan RNN cell，激活函数是tanh或ReLU，用于输入序列。 将一个多层的 Elman RNNCell，激活函数为tanh或者ReLU，用于输入序列。 $$ h'=tanh(w_{ih} x+b_{ih}+w_{hh} h+b_{hh}) $$ 如果nonlinearity=relu，那么将会使用ReLU来代替tanh。

rnn = nn.RNNCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx = rnn(input[i], hx)
   output.append(hx)


#class torch.nn.LSTMCell(input_size, hidden_size, bias=True)[source]
rnn = nn.LSTMCell(10, 20)
input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
cx = Variable(torch.randn(3, 20))
output = []
for i in range(6):
   hx, cx = rnn(input[i], (hx, cx))
   output.append(hx)


#Dropout layers
#class torch.nn.Dropout(p=0.5, inplace=False)
#随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的。
m = nn.Dropout(p=0.2)
input = autograd.Variable(torch.randn(20, 16))
output = m(input)


#Loss functions

#class torch.nn.L1Loss(size_average=True)[source]
#创建一个衡量输入x(模型预测输出)和目标y之间差的绝对值的平均值的标准。

#class torch.nn.MSELoss(size_average=True)[source]
#创建一个衡量输入x(模型预测输出)和目标y之间均方误差标准。

#class torch.nn.CrossEntropyLoss(weight=None, size_average=True)[source]
#此标准将LogSoftMax和NLLLoss集成到一个类中。

#class torch.nn.NLLLoss(weight=None, size_average=True)[source]
#负的log likelihood loss损失。用于训练一个n类分类器。

#class torch.nn.NLLLoss2d(weight=None, size_average=True)[source]
#对于图片的 negative log likehood loss。计算每个像素的 NLL loss。

#class torch.nn.KLDivLoss(weight=None, size_average=True)[source]
#计算 KL 散度损失。

#class torch.nn.BCELoss(weight=None, size_average=True)[source]
#计算 target 与 output 之间的二进制交叉熵。

#class torch.nn.MarginRankingLoss(margin=0, size_average=True)[source]
#class torch.nn.HingeEmbeddingLoss(size_average=True)[source]
#class torch.nn.MultiLabelMarginLoss(size_average=True)[source]

#class torch.nn.SmoothL1Loss(size_average=True)[source]
#平滑版L1 loss。

#class torch.nn.SoftMarginLoss(size_average=True)[source]
#class torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=True)[source]
#class torch.nn.CosineEmbeddingLoss(margin=0, size_average=True)[source]
#class torch.nn.MultiMarginLoss(p=1, margin=1, weight=None, size_average=True)[source]


#Utilities

#torch.nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)[source]
#正则項的值由所有的梯度计算出来，就像他们连成一个向量一样。梯度被in-place operation修改。

#torch.nn.utils.rnn.PackedSequence(_cls, data, batch_sizes)[source]
#All RNN modules accept packed sequences as inputs. 所有的RNN模块都接收这种被包裹后的序列作为它们的输入。

#torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False)[source]
#这里的pack，理解成压紧比较好。 将一个 填充过的变长序列 压紧。（填充时候，会有冗余，所以压紧一下）

#torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False)[source]
#填充packed_sequence。上面提到的函数的功能是将一个填充后的变长序列压紧。 这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。

from torch.nn import utils as nn_utils
batch_size = 2
max_length = 3
hidden_size = 2
n_layers =1

tensor_in = torch.FloatTensor([[1, 2, 3], [1, 0, 0]]).resize_(2,3,1)
tensor_in = Variable( tensor_in ) #[batch, seq, feature], [2, 3, 1]
seq_lengths = [3,1] # list of integers holding information about the batch size at each sequence step

# pack it
pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)

# initialize
rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))

# forward
out, _ = rnn(pack, h0)

# unpack
unpacked = nn_utils.rnn.pad_packed_sequence(out)
print(unpacked)


