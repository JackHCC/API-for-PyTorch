import torch as t
import numpy as np

# torch package ------------------------------------------------------------------------

tensor0=t.randn(1, 2, 3, 4, 5)
print(tensor0, t.numel(tensor0))  #numel返回张量元素个数
tensor1=t.eye(3)                  #生成对角线张量，对角线全1，其他为0

var1=np.array([1, 2, 3])
tensor2=t.from_numpy(var1)        #将numpy.ndarray 转换为pytorch的 Tensor

tensor2[0]=-1   #var1：[1,2,3]-->[-1,2,3]


# 生成Tensor的方法，对比Numpy
tensor3=t.linspace(3, 10, steps=5)  #返回一个1维张量，包含在区间start 和 end 上均匀间隔的steps个点。 输出1维张量的长度为steps
tensor4=t.logspace(start=-10, end=10, steps=5)   #返回一个1维张量，包含在区间 10的start次方到10的end次方上以对数刻度均匀间隔的steps个点。 输出1维张量的长度为steps
tensor5=t.ones(2, 3)    #返回一个2*3的全1张量 1 1 1；1 1 1（两行三列）
tensor6=t.rand(2, 3)    #返回一个张量，包含了从区间[0,1)的均匀分布中抽取的一组随机数，形状由可变参数sizes 定义
tensor7=t.randn(2, 3)   #返回一个张量，包含了从标准正态分布(均值为0，方差为 1，即高斯白噪声)中抽取的一组随机数，形状由可变参数sizes 定义
tensor8=t.randperm(4)   #给定参数n，返回一个从0 到n -1 的随机整数排列
tensor9=t.arange(1, 2.5, 0.5)  #torch.arange(start, end, step=1, out=None) → Tensor
# start (float) – 序列的起始点; end (float) – 序列的终止点; step (float) – 相邻点的间隔大小; out (Tensor, optional) – 结果张量
# 注：输出不包括end
tensor10=t.range(1, 2.5, 0.5)  #同arange：但是输出包括end，多一项
# 建议使用函数 torch.arange()
tensor11=t.zeros(2, 3)  #生成全0张量

#Tensor操作：索引，切片，连接，换位
tensor12=t.cat((tensor2, tensor2, tensor2), 0)   #连接：以0维连接张量（竖向），1为横向

var2 = t.Tensor([[1, 2], [3, 4]])
#tensor13=t.gather(var2, 1, t.LongTensor([[0, 0], [1, 0]]))   #索引:torch.gather(input, dim, index, out=None) → Tensor;沿给定轴dim，将输入索引张量index指定位置的值进行聚合。

var3=t.randn(3, 4)
indices = t.LongTensor([0, 2])
t.index_select(var3, 0, indices)    #按0维方向以indices索引var3张量

#t.squeeze(input, dim=None, out=None)     #在给定的dim维度去除张量形状1
t.t(var3)                           #张量转置（针对二维），新生成一个矩阵，改变不会影响源矩阵
t.transpose(var3, 0, 1)             #张量转置（针对二维），新生成一个矩阵，改变会影响源矩阵一起改变

t.unsqueeze(var3, 0)                #返回一个新的张量，对输入的制定位置插入维度 1； 注意： 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。

#随机抽样 Random sampling
seed=2
print(t.manual_seed(seed))          #设定生成随机数的种子，并返回一个 torch._C.Generator 对象.
t.initial_seed()                    #返回生成随机数的原始种子值（python long）。

print(t.get_rng_state())#[source]   #返回随机生成器状态(ByteTensor)
#t.set_rng_state(new_state)[source] #设定随机生成器状态 参数: new_state (torch.ByteTensor) – 期望的状态

var4 = t.Tensor(3, 3).uniform_(0, 1)
tensor13=t.bernoulli(var4)          #从伯努利分布中抽取二元随机数(0 或者 1)。

t.normal(means=t.arange(1, 11), std=t.arange(1, 0, -0.1))     #返回一个张量，包含从给定参数means,std的离散正态分布中抽取随机数。 均值means是一个张量，包含每个输出元素相关的正态分布的均值。 std是一个张量，包含每个输出元素相关的正态分布的标准差。 均值和标准差的形状不须匹配，但每个张量的元素个数须相同。

#序列化 Serialization

#torch.save(obj, f, pickle_module=<module 'pickle' from '/home/jenkins/miniconda/lib/python3.5/pickle.py'>, pickle_protocol=2)
#obj – 保存对象；f － 类文件对象 (返回文件描述符)或一个保存文件名的字符串；pickle_module – 用于pickling元数据和对象的模块；pickle_protocol – 指定pickle protocal 可以覆盖默认参数
#torch.load(f, map_location=None, pickle_module=<module 'pickle' from '/home/jenkins/miniconda/lib/python3.5/pickle.py'>)
#f – 类文件对象 (返回文件描述符)或一个保存文件名的字符串；map_location – 一个函数或字典规定如何remap存储位置；pickle_module – 用于unpickling元数据和对象的模块 (必须匹配序列化文件时的pickle_module )

#并行化 Parallelism
t.get_num_threads()     #获得用于并行化CPU操作的OpenMP线程数
int=3
t.set_num_threads(int)  #设定用于并行化CPU操作的OpenMP线程数

#数学操作Math operations
input=t.Tensor([[1,2][3,4]])
value=3
t.abs(input, out=None)          #返回张量：取绝对值
t.acos(input, out=None)         #返回张量：反余弦
t.add(input, value, out=None)   #返回张量：加value值
#t.addcdiv(input, value=1, tensor1, tensor2, out=None)   #用tensor2对tensor1逐元素相除，然后乘以标量值value 并加到tensor
#t.addcmul(input, value=1, tensor1, tensor2, out=None)   #用tensor2对tensor1逐元素相乘，并对结果乘以标量值value然后加到tensor。 张量的形状不需要匹配，但元素数量必须一致。 如果输入是FloatTensor or DoubleTensor类型，则value 必须为实数，否则须为整数。
t.asin(input, out=None)         #取反正弦
t.atan(input, out=None)         #取反正切
#t.atan2(input1, input2, out=None)      #返回一个新张量，包含两个输入张量input1和input2的反正切函数
t.ceil(input, out=None)         #天井函数，对输入input张量每个元素向上取整, 即取不小于每个元素的最小整数，并返回结果到输出。
#t.clamp(input, min, max, out=None)     #将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量。
t.cos(input, out=None)
t.cosh(input, out=None)
t.div(input, value, out=None)   #将input逐元素除以标量值value，并返回结果到输出张量out
t.exp(input, out=None)
t.floor(input, out=None)        #床函数: 返回一个新张量，包含输入input张量每个元素的floor，即不小于元素的最大整数。
#t.fmod(input, divisor, out=None)   #计算除法余数。 除数与被除数可能同时含有整数和浮点数。此时，余数的正负与被除数相同。
t.frac(input, out=None)         #返回小数部分
#t.lerp(start, end, weight, out=None)   #对两个张量以start，end做线性插值， 将结果返回到输出张量。
t.log(input, out=None)
t.log1p(input, out=None)        #计算 input+1的自然对数 yi=log(xi+1)
t.mul(input, value, out=None)   #用标量值value乘以输入input的每个元素，并返回一个新的结果张量。 out=tensor∗value
t.neg(input, out=None)          #返回一个新张量，包含输入input 张量按元素取负。 即， out=−1∗input
#t.pow(input, exponent, out=None)   #对输入input的按元素求exponent次幂值，并返回结果张量。 幂值exponent 可以为单一 float 数或者与input相同元素数的张量。
t.reciprocal(input, out=None)   #返回一个新张量，包含输入input张量每个元素的倒数，即 1.0/x。
#t.remainder(input, divisor, out=None)  #返回一个新张量，包含输入input张量每个元素的除法余数。 除数与被除数可能同时包含整数或浮点数。余数与除数有相同的符号。
t.round(input, out=None)        #返回一个新张量，将输入input张量每个元素舍入到最近的整数。
t.rsqrt(input, out=None)        #返回一个新张量，包含输入input张量每个元素的平方根倒数。
t.sigmoid(input, out=None)      #返回一个新张量，包含输入input张量每个元素的sigmoid值。
t.sign(input, out=None)         #符号函数：返回一个新张量，包含输入input张量每个元素的正负。
t.sin(input, out=None)
t.sinh(input, out=None)
t.sqrt(input, out=None)         #返回一个新张量，包含输入input张量每个元素的平方根。
t.tan(input, out=None)
t.tanh(input, out=None)
t.trunc(input, out=None)        #返回一个新张量，包含输入input张量每个元素的截断值(标量x的截断值是最接近其的整数，其比x更接近零。简而言之，有符号数的小数部分被舍弃)。


#Reduction Ops
dim=0   #维度
t.cumprod(input, dim, out=None)     #返回输入沿指定维度的累积积。例如，如果输入是一个N 元向量，则结果也是一个N 元向量，第i 个输出元素值为yi=x1∗x2∗x3∗...∗xi
t.cumsum(input, dim, out=None)      #返回输入沿指定维度的累积和。例如，如果输入是一个N元向量，则结果也是一个N元向量，第i 个输出元素值为 yi=x1+x2+x3+...+xi
#torch.dist(input, other, p=2, out=None)    #返回 (input - other) 的 p范数 。
t.mean(input)                       #返回输入张量所有元素的均值。
t.median(input, dim=-1, values=None, indices=None)      #返回输入张量给定维度每行的中位数，同时返回一个包含中位数的索引的LongTensor。dim值默认为输入张量的最后一维。 输出形状与输入相同，除了给定维度上为1.
t.mode(input, dim=-1, values=None, indices=None)        #返回给定维dim上，每行的众数值。 同时返回一个LongTensor，包含众数职的索引。dim值默认为输入张量的最后一维。
t.norm(input, p=2)                  #返回输入张量input 的p 范数。
t.prod(input)                       #返回输入张量input 所有元素的积。
t.std(input)                        #返回输入张量input 所有元素的标准差。
t.sum(input)                        #返回输入张量input 所有元素的和。
t.var(input)                        #返回输入张量所有元素的方差


#比较操作 Comparison Ops
#t.eq(input, other, out=None)       #比较元素相等性。第二个参数可为一个数或与第一个参数同类型形状的张量。
#t.equal(tensor1, tensor2)          #如果两个张量有相同的形状和元素值，则返回True ，否则 False。
#t.ge(input, other, out=None)       #逐元素比较input和other，即是否 input>=other。如果两个张量有相同的形状和元素值，则返回True ，否则 False。 第二个参数可以为一个数或与第一个参数相同形状和类型的张量
#t.gt(input, other, out=None)       #逐元素比较input和other ， 即是否input>other 如果两个张量有相同的形状和元素值，则返回True ，否则 False。 第二个参数可以为一个数或与第一个参数相同形状和类型的张量
#t.kthvalue(input, k, dim=None, out=None)   #取输入张量input指定维上第k 个最小值。如果不指定dim，则默认为input的最后一维。返回一个元组 (values,indices)，其中indices是原始输入张量input中沿dim维的第 k 个最小值下标。
#t.le(input, other, out=None)       #逐元素比较input和other ， 即是否input<=other 第二个参数可以为一个数或与第一个参数相同形状和类型的张量
#t.lt(input, other, out=None)       #逐元素比较input和other ， 即是否 input<other
t.max(input)
t.min(input)
#t.ne(input, other, out=None)       #逐元素比较input和other ， 即是否 input!=other。 第二个参数可以为一个数或与第一个参数相同形状和类型的张量
t.sort(input, dim=None, descending=False, out=None)     #对输入张量input沿着指定维按升序排序。如果不给定dim，则默认为输入的最后一维。如果指定参数descending为True，则按降序排序


#其它操作 Other Operations
#t.cross(input, other, dim=-1, out=None)        #返回沿着维度dim上，两个张量input和other的向量积（叉积）。 input和other 必须有相同的形状，且指定的dim维上size必须为3。
t.diag(input, diagonal=0, out=None)             #如果输入是一个向量(1D 张量)，则返回一个以input为对角线元素的2D方阵；如果输入是一个矩阵(2D 张量)，则返回一个包含input对角线元素的1D张量
t.histc(input, bins=100, min=0, max=0, out=None)    #计算输入张量的直方图。以min和max为range边界，将其均分成bins个直条，然后将排序好的数据划分到各个直条(bins)中。如果min和max都为0, 则利用数据中的最大最小值作为边界。
#t.renorm(input, p, dim, maxnorm, out=None)     #返回一个张量，包含规范化后的各个子张量，使得沿着dim维划分的各子张量的p范数小于maxnorm。
t.trace(input)                                  #返回输入2维矩阵对角线元素的和(迹)
t.tril(input, k=0, out=None)                    #返回一个张量out，包含输入矩阵(2D张量)的下三角部分，out其余部分被设为0。这里所说的下三角部分为矩阵指定对角线diagonal之上的元素。






