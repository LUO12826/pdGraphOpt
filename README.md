## 描述图变换规则的DSL的编译器

### 基本说明
该编译器由llvm TableGen工具修改而来。TableGen工具本身也是一个前后端分离的编译器。其代码主要存放在/lib/TableGen（通用的代码，主要是前端部分）和/utils/TableGen(各个后端的代码)中。

对TableGen的修改主要在后端部分，因此增加的代码均在/utils/TableGen下。其中：

/utils/TableGen/TableGen.cpp为TableGen程序入口。

/utils/TableGen/PdLiteGraphOptPassEmitter.cpp为生成Paddle Lite图优化代码的后端。

/utils/TableGen/PdLiteGraphOpt/为辅助代码。

**预先编写好的DSL代码**存放于/paddle-lite-def下。

其中，FusePattern.td描述了Paddle Lite中的部分子图融合模式，Op.td描述了部分算子。其它两个文件提供了一些基础设施。



### 代码生成实验说明

对Paddle Lite框架的代码生成实验基于Paddle Lite origin/release/v2.10版本。

代码生成实验主要有两个步骤：

一、使用DSL编译器将DSL代码编译为C++代码；

二、将这些C++代码集成到Paddle Lite框架中。

下面分别介绍这两个主要步骤的细节。



#### 一、使用DSL编译器将DSL代码编译为C++代码
##### 1.克隆DSL编译器代码，也即本代码仓库。

```shell
git clone https://github.com/LUO12826/pdGraphOpt.git
```

目前的代码仓库还包含llvm仓库的历史提交记录（尽管大部分llvm的代码文件已经移除），因此体积较大。为加快下载速度，也可以只拉取最新提交：

```
git clone https://github.com/LUO12826/pdGraphOpt.git --depth=1
```

##### 2.进入代码仓库，并编译

```shell
cd pdGraphOpt
```

从cmake生成工程文件（默认cmake在PATH中）：

```shell
cmake -B build \
-DCMAKE_BUILD_TYPE=Debug \
-DLLVM_INCLUDE_TESTS=false \
-DLLVM_INCLUDE_UTILS=false \
-DLLVM_INCLUDE_BENCHMARKS=false \
-DLLVM_INCLUDE_EXAMPLES=false \
-DLLVM_INCLUDE_RUNTIMES=false \
-DLLVM_INCLUDE_TOOLS=false \
-G Ninja
```

这里将工程文件的子目录通过-B参数设为了build，并通过-G指定使用了Ninja作为编译工具。如果使用CLion IDE，这个编译工具已经内置。或者也可以将`Ninja` 改为`"Unix Makefiles"`以使用make编译（但这里未经过测试）。

执行编译：

```shell
cmake --build build --target llvm-tblgen
```

##### 3.运行llvm-tblgen程序（即我们的DSL编译器）。

上一步成功后，应该会在/build/bin下找到llvm-tblgen程序的二进制文件。按下面的方式运行它：

```shell
llvm-tblgen -gen-paddle-lite-graph-opt \
<FusePattern.td文件的路径> \
-I <paddle-lite-def目录的路径> \
-o <输出的C++文件的路径，包括文件名>
```

然后会在指定的输出文件路径下找到生成的C++代码。所有的代码都在一个文件中。下文中假设这个文件叫pdGen.cpp。

至此，第一个主要步骤结束。



#### 二、将生成的C++代码集成到Paddle Lite框架中

##### 1.克隆Paddle Lite代码并切换到release/v2.10分支

```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
```

```shell
cd Paddle-Lite
```

```shell
git checkout release/v2.10
```

##### 2.将生成的文件pdGen.cpp拷贝至Paddle-Lite/lite/core/optimizer/mir/fusion目录下。

##### 3.删除Paddle Lite框架中原有的、功能相同的计算图优化pass。

这是对Paddle Lite中的pass进行调整，可参考https://www.paddlepaddle.org.cn/lite/v2.10/develop_guides/add_new_pass.html。(无需查看也可以进行下面的步骤)

这里我们通过注释的方式使之不发挥作用。以fc fuse pass为例：

3.1 首先打开Paddle-Lite/lite/core/optimizer/mir/fusion/fc_fuse_pass.cc，将其中的REGISTER_MIR_PASS部分整体注释掉，即：

```c++
//REGISTER_MIR_PASS(lite_fc_fuse_pass, paddle::lite::mir::FcFusePass)
//    .BindTargets({TARGET(kAny)})
//    .ExcludeTargets({TARGET(kXPU)})
//#if (!defined(LITE_WITH_MLU) && !defined(LITE_WITH_HUAWEI_ASCEND_NPU) && \
//     !defined(LITE_WITH_NNADAPTER) && !defined(LITE_WITH_METAL))
//    .ExcludeTargets({TARGET(kX86)})
//#endif
//    .ExcludeTargets({TARGET(kBM)})
//    .BindKernel("fc");
```

这里需要注意的是上面代码中`REGISTER_MIR_PASS`后的`lite_fc_fuse_pass`。我们姑且称它为一个pass的标识符。

3.2 打开Paddle-Lite/lite/core/optimizer/optimizer.cc，定位到RunDefaultOptimizer函数，并找到其中的passes_local局部变量。它是一个字符串列表。找到`"lite_fc_fuse_pass",`（即上面说的pass标识符）这一行，将其注释掉。

3.3 打开Paddle-Lite/lite/api/paddle_use_passes.h，找到`USE_MIR_PASS(lite_fc_fuse_pass);`这一行，将其注释掉。

##### 4.将生成的pass注册到Paddle Lite中

这一步类似于第三步的逆过程。同样以fc fuse pass为例：

4.1 找到生成pass的标识符。有两种方式，（1）直接到生成的pdGen.cpp文件中寻找，查看生成的fc fuse pass代码`REGISTER_MIR_PASS`后的pass标识符。（2）根据命名规则推断：在目前DSL编译器设计中，如果FusePattern.td中的一条Pattern记录名为FcPat，则生成的pass标识符是`lite_fc_fuse_pass`。也即先将大驼峰命名转为蛇形命名，再将pat后缀去掉，最后加上lite前缀和fuse_pass后缀。**下面，我们假设一个生成的pass标识符为`gen_lite_fc_fuse_pass`。**

4.2 打开Paddle-Lite/lite/core/optimizer/optimizer.cc，定位到RunDefaultOptimizer函数，并找到其中的passes_local局部变量。它是一个字符串列表。向其中添加一个元素`"gen_lite_fc_fuse_pass"`。

4.3 打开Paddle-Lite/lite/api/paddle_use_passes.h，增加一行：`USE_MIR_PASS(gen_lite_fc_fuse_pass);`

##### 5.补充说明

可见，上述的第3步和第4部互为逆过程。因此，只要在FusePattern.td文件中恰当命名，就可以使得生成的pass和原有的pass的标识符完全一致。这样，第3步和第4步可以简化为3.1一步，即只需将旧的REGISTER_MIR_PASS代码块移除。

##### 6.编译Paddle Lite框架，得到模型优化工具opt

编译步骤因环境而异，Paddle Lite官网有说明。这里请参考https://www.paddlepaddle.org.cn/lite/v2.10/source_compile/compile_env.html。

由于计算图优化发生在模型优化工具opt中，这里可以不全量编译，即在cmake编译时指定`--target opt`。

##### 7.成功编译opt工具后，即可将其用于后续测试

opt工具使用方式参考https://www.paddlepaddle.org.cn/lite/v2.10/user_guides/model_optimize_tool.html。
