

# torch.cuda.current_blas_handle()
# 返回cublasHandle_t指针，指向当前cuBLAS句柄

# torch.cuda.current_device()
# 返回当前所选设备的索引。

# torch.cuda.current_stream()
# 返回一个当前所选的Stream

# class torch.cuda.device(idx)
# 上下文管理器，可以更改所选设备。
# 参数： - idx (int) – 设备索引选择。如果这个参数是负的，则是无效操作。

# torch.cuda.device_count()
# 返回可得到的GPU数量。

# class torch.cuda.device_of(obj)
# 将当前设备更改为给定对象的上下文管理器。可以使用张量和存储作为参数。如果给定的对象不是在GPU上分配的，这是一个无效操作。
# 参数： - obj (Tensor or Storage) – 在选定设备上分配的对象。

# torch.cuda.is_available()
# 返回一个bool值，指示CUDA当前是否可用。

# torch.cuda.set_device(device)
# 设置当前设备。不鼓励使用此函数来设置。在大多数情况下，最好使用CUDA_VISIBLE_DEVICES环境变量。
# 参数： - device (int) – 所选设备。如果此参数为负，则此函数是无效操作。

# torch.cuda.stream(stream)
# 选择给定流的上下文管理器。在其上下文中排队的所有CUDA核心将在所选流上入队。
# 参数： - stream (Stream) – 所选流。如果是None，则这个管理器是无效的。

# torch.cuda.synchronize()
# 等待当前设备上所有流中的所有核心完成。


#交流集

#torch.cuda.comm.broadcast(tensor, devices)
#向一些GPU广播张量。
# 参数： - tensor (Tensor) – 将要广播的张量 - devices (Iterable) – 一个可以广播的设备的迭代。注意，它的形式应该像（src，dst1，dst2，...），其第一个元素是广播来源的设备。
# 返回： 一个包含张量副本的元组，放置在与设备的索引相对应的设备上。

# torch.cuda.comm.reduce_add(inputs, destination=None)
# 将来自多个GPU的张量相加。所有输入应具有匹配的形状。
# 参数： - inputs (Iterable[Tensor]) – 要相加张量的迭代 - destination (int, optional) – 将放置输出的设备（默认值：当前设备）。
# 返回： 一个包含放置在destination设备上的所有输入的元素总和的张量。

# torch.cuda.comm.scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None)
# 打散横跨多个GPU的张量。
# 参数： - tensor (Tensor) – 要分散的张量 - devices (Iterable[int]) – int的迭代，指定哪些设备应该分散张量。 - chunk_sizes (Iterable[int], optional) – 要放置在每个设备上的块大小。它应该匹配devices的长度并且总和为tensor.size(dim)。 如果没有指定，张量将被分成相等的块。 - dim (int, optional) – 沿着这个维度来chunk张量
# 返回： 包含tensor块的元组，分布在给定的devices上。

# torch.cuda.comm.gather(tensors, dim=0, destination=None)
# 从多个GPU收集张量。张量尺寸在不同于dim的所有维度上都应该匹配。
# 参数： - tensors (Iterable[Tensor]) – 要收集的张量的迭代。 - dim (int) – 沿着此维度张量将被连接。 - destination (int, optional) – 输出设备（-1表示CPU，默认值：当前设备）。
# 返回： 一个张量位于destination设备上，这是沿着dim连接tensors的结果。


# 流和事件

# class torch.cuda.Stream
# CUDA流的包装。
# 参数： - device (int, optional) – 分配流的设备。 - priority (int, optional) – 流的优先级。较低的数字代表较高的优先级。



#torch.utils.ffi

# torch.utils.ffi.create_extension(name, headers, sources, verbose=True, with_cuda=False, package=False, relative_to='.', **kwargs)
# 创建并配置一个cffi.FFI对象,用于PyTorch的扩展。

# name (str) – 包名。可以是嵌套模块，例如 .ext.my_lib。
# headers (str or List[str]) – 只包含导出函数的头文件列表
# sources (List[str]) – 用于编译的sources列表
# verbose (bool, optional) – 如果设置为False，则不会打印输出（默认值：True）。
# with_cuda (bool, optional) – 设置为True以使用CUDA头文件进行编译（默认值：False）。
# package (bool, optional) – 设置为True以在程序包模式下构建（对于要作为pip程序包安装的模块）（默认值：False）。
# relative_to (str, optional) –构建文件的路径。package为True时需要。最好使用__file__作为参数。
# kwargs – 传递给ffi以声明扩展的附加参数。


#torch.utils.data

# class torch.utils.data.Dataset
# 表示Dataset的抽象类。 所有其他数据集都应该进行子类化。所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)。

# class torch.utils.data.TensorDataset(data_tensor, target_tensor)
# 包装数据和目标张量的数据集。通过沿着第一个维度索引两个张量来恢复每个样本。

# 参数：
# data_tensor (Tensor) －　包含样本数据
# target_tensor (Tensor) －　包含样本目标（标签）

# class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
# 数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。

# 参数：
# dataset (Dataset) – 加载数据的数据集。
# batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
# shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
# sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
# num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
# collate_fn (callable, optional) –
# pin_memory (bool, optional) –
# drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

# class torch.utils.data.sampler.Sampler(data_source)
# 所有采样器的基础类。每个采样器子类必须提供一个__iter__方法，提供一种迭代数据集元素的索引的方法，以及返回迭代器长度的__len__方法。

# class torch.utils.data.sampler.SequentialSampler(data_source)
# 样本元素顺序排列，始终以相同的顺序。

# 参数： - data_source (Dataset) – 采样的数据集。

# class torch.utils.data.sampler.RandomSampler(data_source)
# 样本元素随机，没有替换。
# 参数： - data_source (Dataset) – 采样的数据集。

# class torch.utils.data.sampler.SubsetRandomSampler(indices)
# 样本元素从指定的索引列表中随机抽取，没有替换。
# 参数： - indices (list) – 索引的列表

# class torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)
# 样本元素来自于[0,..,len(weights)-1]，给定概率（weights）。


# torch.utils.model_zoo
# torch.utils.model_zoo.load_url(url, model_dir=None)
# 在给定URL上加载Torch序列化对象。如果对象已经存在于 model_dir 中，则将被反序列化并返回。URL的文件名部分应遵循命名约定filename-<sha256>.ext，其中<sha256>是文件内容的SHA256哈希的前八位或更多位数字。哈希用于确保唯一的名称并验证文件的内容。

# model_dir 的默认值为$TORCH_HOME/models，其中$TORCH_HOME默认为~/.torch。可以使用$TORCH_MODEL_ZOO环境变量来覆盖默认目录。

# 参数：
# url (string) - 要下载对象的URL
# model_dir (string, optional) - 保存对象的目录
