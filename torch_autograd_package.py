

#torch.autograd提供了类和函数用来对任意标量函数进行求导。要想使用自动求导，只需要对已有的代码进行微小的改变。只需要将所有的tensor包含进Variable对象中即可。

#torch.autograd.backward(variables, grad_variables, retain_variables=False)
#Computes the sum of gradients of given variables w.r.t. graph leaves. 给定图的叶子节点variables, 计算图中变量的梯度和。 计算图可以通过链式法则求导。如果variables中的任何一个variable是 非标量(non-scalar)的，且requires_grad=True。那么此函数需要指定grad_variables，它的长度应该和variables的长度匹配，里面保存了相关variable的梯度(对于不需要gradient tensor的variable，None是可取的)。

# 参数：
# variables (variable 列表) – 被求微分的叶子节点，即 ys 。
# grad_variables (Tensor 列表) – 对应variable的梯度。仅当variable不是标量且需要求梯度的时候使用。
# retain_variables (bool) – True,计算梯度时所需要的buffer在计算完梯度后不会被释放。如果想对一个子图多次求微分的话，需要设置为True。

# Variable

# API 兼容性
# Variable API 几乎和 Tensor API一致 (除了一些in-place方法，这些in-place方法会修改 required_grad=True的 input 的值)。多数情况下，将Tensor替换为Variable，代码一样会正常的工作。由于这个原因，我们不会列出Variable的所有方法，你可以通过torch.Tensor的文档来获取相关知识。

# In-place operations on Variables
# 在autograd中支持in-place operations是非常困难的。同时在很多情况下，我们阻止使用in-place operations。Autograd的贪婪的 释放buffer和 复用使得它效率非常高。只有在非常少的情况下，使用in-place operations可以降低内存的使用。除非你面临很大的内存压力，否则不要使用in-place operations。

# In-place 正确性检查
# 所有的Variable都会记录用在他们身上的 in-place operations。如果pytorch检测到variable在一个Function中已经被保存用来backward，但是之后它又被in-place operations修改。当这种情况发生时，在backward的时候，pytorch就会报错。这种机制保证了，如果你用了in-place operations，但是在backward过程中没有报错，那么梯度的计算就是正确的。

#class torch.autograd.Variable [source]

#包装一个Tensor,并记录用在它身上的operations。
# Variable是Tensor对象的一个thin wrapper，它同时保存着Variable的梯度和创建这个Variable的Function的引用。这个引用可以用来追溯创建这个Variable的整条链。如果Variable是被用户所创建的，那么它的creator是None，我们称这种对象为 leaf Variables。
# 由于autograd只支持标量值的反向求导(即：y是标量)，梯度的大小总是和数据的大小匹配。同时，仅仅给leaf variables分配梯度，其他Variable的梯度总是为0.

#class torch.autograd.Function[source]

#Records operation history and defines formulas for differentiating ops. 记录operation的历史，定义微分公式。 每个执行在Varaibles上的operation都会创建一个Function对象，这个Function对象执行计算工作，同时记录下来。这个历史以有向无环图的形式保存下来，有向图的节点为functions，有向图的边代表数据依赖关系(input<-output)。之后，当backward被调用的时候，计算图以拓扑顺序处理，通过调用每个Function对象的backward()，同时将返回的梯度传递给下一个Function。
# 通常情况下，用户能和Functions交互的唯一方法就是创建Function的子类，定义新的operation。这是扩展torch.autograd的推荐方法。
# 由于Function逻辑在很多脚本上都是热点，所有我们把几乎所有的Function都使用C实现，通过这种策略保证框架的开销是最小的。

#backward(* grad_output)[source]
#forward(* input)[source]
#mark_dirty(* args)[source]
#mark_non_differentiable(* args)[source]
#mark_shared_storage(* pairs)[source]
#save_for_backward(* tensors)[source]


#torch.optim
from torch import optim

# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
# optimizer = optim.Adam([var1, var2], lr = 0.0001)

#class torch.optim.Optimizer(params, defaults) [source]
#load_state_dict(state_dict) [source]
#state_dict() [source]
#step(closure) [source]
#zero_grad() [source]

#class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)[source]
#class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)[source]
#class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]
#class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]
#class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)[source]
#class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)[source]
#class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)[source]
#class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))[source]
#class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)[source]



