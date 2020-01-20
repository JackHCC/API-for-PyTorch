import torch.nn as nn
import torch.nn.functional

#Convolution 函数
#torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

#torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

#torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

#torch.nn.functional.conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)
#torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)
#torch.nn.functional.conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)


#Pooling 函数
#torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
#torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
#torch.nn.functional.avg_pool3d(input, kernel_size, stride=None)

# torch.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
# torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
# torch.nn.functional.max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
# torch.nn.functional.max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
# torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
# torch.nn.functional.max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
# torch.nn.functional.lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
# torch.nn.functional.adaptive_max_pool1d(input, output_size, return_indices=False)
# torch.nn.functional.adaptive_max_pool2d(input, output_size, return_indices=False)
# torch.nn.functional.adaptive_avg_pool1d(input, output_size)
# torch.nn.functional.adaptive_avg_pool2d(input, output_size)

#非线性激活函数

# torch.nn.functional.threshold(input, threshold, value, inplace=False)
# torch.nn.functional.relu(input, inplace=False)
# torch.nn.functional.hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False)
# torch.nn.functional.relu6(input, inplace=False)
# torch.nn.functional.elu(input, alpha=1.0, inplace=False)
# torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
# torch.nn.functional.prelu(input, weight)
# torch.nn.functional.rrelu(input, lower=0.125, upper=0.3333333333333333, training=False, inplace=False)
# torch.nn.functional.logsigmoid(input)
# torch.nn.functional.hardshrink(input, lambd=0.5)
# torch.nn.functional.tanhshrink(input)
# torch.nn.functional.softsign(input)
# torch.nn.functional.softplus(input, beta=1, threshold=20)
# torch.nn.functional.softmin(input)
# torch.nn.functional.softmax(input)
# torch.nn.functional.softshrink(input, lambd=0.5)
# torch.nn.functional.log_softmax(input)
# torch.nn.functional.tanh(input)
# torch.nn.functional.sigmoid(input)


#Normalization 函数
# torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)

# 线性函数
# torch.nn.functional.linear(input, weight, bias=None)

# Dropout 函数
# torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)

# 距离函数（Distance functions）
# torch.nn.functional.pairwise_distance(x1, x2, p=2, eps=1e-06)

# 损失函数（Loss functions）
# torch.nn.functional.nll_loss(input, target, weight=None, size_average=True)









