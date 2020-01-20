


#torch.nn.init.calculate_gain(nonlinearity,param=None)

# gain = nn.init.gain('leaky_relu')
# torch.nn.init.uniform(tensor, a=0, b=1)

#torch.nn.init.normal(tensor, mean=0, std=1)
#从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量



#torch.multiprocessing

# 封装了multiprocessing模块。用于在相同数据的不同进程中共享视图。
# 一旦张量或者存储被移动到共享单元(见share_memory_()),它可以不需要任何其他复制操作的发送到其他的进程中。
# 这个API与原始模型完全兼容，为了让张量通过队列或者其他机制共享，移动到内存中，我们可以
# 由原来的import multiprocessing改为import torch.multiprocessing。
# 由于API的相似性，我们没有记录这个软件包的大部分内容，我们建议您参考原始模块的非常好的文档。

#Strategy management

# torch.multiprocessing.get_all_sharing_strategies()
# 返回一组由当前系统所支持的共享策略
# torch.multiprocessing.get_sharing_strategy()
# 返回当前策略共享CPU中的张量。
# torch.multiprocessing.set_sharing_strategy(new_strategy)
# 设置共享CPU张量的策略

#Sharing CUDA tensors

# 共享CUDA张量进程只支持Python3，使用spawn或者forkserver开始方法。
# Python2中的multiprocessing只能使用fork创建子进程，并且不被CUDA支持。




#遗产包 - torch.legacy
#此包中包含从Lua Torch移植来的代码。为了可以使用现有的模型并且方便当前Lua Torch使用者过渡，我们创建了这个包。 可以在torch.legacy.nn中找到nn代码，并在torch.legacy.optim中找到optim代码。 API应该完全匹配Lua Torch。

