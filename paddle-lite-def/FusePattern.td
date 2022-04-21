include "Op.td"
include "PatternBase.td"

// fc系列
let attrToCopy = [(copier "#INPUT_SCALE", "float", "mul", "fc")],
    extraAssertions = [RankEquals<"W", "2">, RankInRange<"b", "1", "2">],
    bindTargets = ["kAny"],
    excludeTargets = ["kXPU", "kBM", "kX86"],
    kernelName = "fc"
    in {

  def FcPat: Pat<
    (ElementwiseAddOp<> (MulOp<"mul"> $x, $W, $x_num_col_dims), tVarWeight:$b),
    (FcOp<"fc"> $W, $x, $b, $x_num_col_dims)
    >;

  def FcV2Pat: Pat<
    (ElementwiseAddOp<>
      (MatMulV2Op<"mul"> $x, $W, OpAttr<BOOL, "false">, OpAttr<BOOL, "false">),
      tVarWeight:$b),
    (FcOp<"fc"> $W, $x, $b, OpAttr<INT, "1">)
    >;

  def FcWithActivationPat: Pat<
    (ReluOp<>
      (ElementwiseAddOp<>
        (MulOp<"mul"> $x, $W, $x_num_col_dims),
        tVarWeight:$b)),
    (FcOp<"fc"> $W, $x, $b, $x_num_col_dims, OpAttr<STRING, "relu">)>;

  def FcV2WithActivationPat: Pat<
    (ReluOp<> 
      (ElementwiseAddOp<> 
        (MatMulV2Op<"mul"> $x, $W, OpAttr<BOOL, "false">, OpAttr<BOOL, "false">),
        tVarWeight:$b)),
    (FcOp<"fc"> $W, $x, $b, OpAttr<INT, "1">, OpAttr<STRING, "relu">)>;
        
}

// element wise系列
foreach eleWiseType = ["elementwise_add", "elementwise_mul"] in {
  foreach actType = ["relu", "tanh"] in {
    def eleWiseType#_#actType#_#pat: Pat<
      (GenericActivationOp<actType> (ElementwiseOp<eleWiseType, "ele"> $x, $y)),
      (ElementwiseActivationOp<"fusion_" # eleWiseType # "_activation", "newEle"> 
              $x, $y, OpAttr<STRING, actType>)> {
      
      let bindTargets = ["kAny"];
      let excludeTargets = ["kXPU", "kX86", "kBM", "kRKNPU", "kNNAdapter"];
      let kernelName = "fusion_" # eleWiseType # "_activation";
      let extraAssertions = [
        RetainOp<"ele", "newEle">
      ];
    }
  }
}


//将卷积后的BatchNorm直接融合到卷积里
//
//   Conv     y = w * x + b1
//   BN       z = scale * (y - mean) / sqrt(var + eps) + b2
//
// 计算后就是
//   y = W * x + B
// 其中
//   W = scale * w / sqrt(var + eps)
//   B = b2 + scale * (b1 - mean) / sqrt(var + eps)
//
// 其中 scale / sqrt(var + eps)这个因子是可以复用的

def Conv2dBnPat: Pat<
    (BatchNormOp<>
        (Conv2dOp<"oldConv"> $input, $filter, $bias),
        $scale, $bias_2, $mean, $var, $epsilon),

    //对于一个32x3x3x3的卷积核（即有32个输出特征图）
    (Conv2dOp<"newConv">
        $input,
        //这里对32x(3x3x3)的卷积核应用1x32的Scale值
        //总共三十二个卷积核，每个卷积核应用一个Scale值
        //所以用RowWise的计算，并把后面三个维度认为是一行
        (DirectRowWiseMul
            $filter,
            (DirectEleWiseDiv:$filterScale
                $scale,
                (DirectEleWiseSqrt (DirectEleWiseAdd $var, $epsilon))
            ),
            OpAttr<INT, "3">
        ),
        (DirectEleWiseAdd
            $bias_2,
            (DirectEleWiseMul (DirectEleWiseSub $bias, $mean), $filterScale)
        )
    )
> {

  let extraAssertions = [
    RetainOp<"oldConv", "newConv">
  ];
  let bindTargets = ["kAny"];
  let excludeTargets = ["kXPU", "kBM", "kRKNPU"];
}


// 对于没有bias的卷积：
//   Conv     y = w * x
//   BN       z = scale * (y - mean) / sqrt(var + eps) + b
// 计算后就是
//   y = W * x + B
// 其中
//   W = scale * w / sqrt(var + eps)
//   B = b - (scale *  mean / sqrt(var + eps))

def Conv2dNoBiasBnPat: Pat<
    (BatchNormOp<>
        (Conv2dOp<"oldConv"> $input, $filter),
        $scale, $bias, $mean, $var, $epsilon),

    (Conv2dOp<"newConv">
        $input,
        (DirectRowWiseMul
            $filter,
            (DirectEleWiseDiv:$filterScale
                $scale,
                (DirectEleWiseSqrt (DirectEleWiseAdd $var, $epsilon))
            ),
            OpAttr<INT, "3">
        ),
        (DirectEleWiseSub
            $bias,
            (DirectEleWiseMul $filterScale, $mean)
        )
    )
> {

  let extraAssertions = [
    RetainOp<"oldConv", "newConv">
  ];

  let bindTargets = ["kAny"];
  let excludeTargets = ["kXPU", "kBM", "kRKNPU"];
}


let extraAssertions = [ InputRankAllEquals<"oldmul", "2"> ],
    bindTargets = ["kAny"] in {

  def MatmulPat: Pat<
    (MatMulOp<"oldmul"> $x, $y,
            OpAttr<FLOAT, "1.0">, OpAttr<BOOL, "false">, OpAttr<BOOL, "false">),
    (MulOp<"mul"> $x, $y, OpAttr<INT, "1">, OpAttr<INT, "1">)>;

  def MatmulV2Pat: Pat<
    (MatMulV2Op<"oldmul"> $x, $y,
          OpAttr<BOOL, "false">, OpAttr<BOOL, "false">,
          OpAttr<BOOL, "false">, OpAttr<FLOAT, "1.0">),
    (MulOp<"mul"> $x, $y, OpAttr<INT, "1">, OpAttr<INT, "1">)>;

}


def FlattenFcPat: Pat<
        (FcOp<"old_fc"> $W, (FlattenContiguousRangeOp<> $x), $b),
        (FcOp<"new_fc"> $W, $x, $b, OpAttr<INT, "1">)> {
  
  let extraAssertions = [
    RetainOp<"old_fc", "new_fc">
  ];

  let bindTargets = ["kOpenCL", "kARM"];
  let kernelName = "fc";
}


def Reshape2MatmulPat: Pat<
        (MatMulOp<"old_mul">
            (Reshape2Op<"reshape"> $x), $y,
            OpAttr<FLOAT, "1.0">, OpAttr<BOOL, "false">, OpAttr<BOOL, "false">),

        (MulOp<"new_mul"> $x, $y, OpAttr<INT, "1">, OpAttr<INT, "1">)> {

  let extraAssertions = [
    InputRankAllEquals<"old_mul", "2">
  ];

  let customTeller = [
    (teller "reshape", [{
    [](const Node* node) -> bool {
    auto reshape2_op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto reshape2_input_x_name = reshape2_op_desc.Input("X").front();
    auto* scope = const_cast<Node*>(node)->stmt()->op()->scope();
    auto reshape2_in_x_shape =
        scope->FindVar(reshape2_input_x_name)->Get<lite::Tensor>().dims();
    size_t reshape2_in_x_rank = reshape2_in_x_shape.size();

    auto shapeAssert = reshape2_op_desc.HasAttr("shape")
                       && reshape2_op_desc.GetAttr<std::vector<int>>("shape").size() == 2;

    return (reshape2_in_x_rank == 4 && reshape2_in_x_shape[2] == 1 &&
            reshape2_in_x_shape[3] == 1 && shapeAssert);
  }
  }])
  ];

  let bindTargets = ["kAny"];
}


def Squeeze2MatmulPat: Pat<
        (MatMulOp<"old_mul"> (Squeeze2Op<"sq"> $x), $y,
            OpAttr<FLOAT, "1.0">, OpAttr<BOOL, "false">, OpAttr<BOOL, "false">),

        (MulOp<"new_mul"> $x, $y, OpAttr<INT, "1">, OpAttr<INT, "1">)> {
  
  let extraAssertions = [
    InputRankAllEquals<"old_mul", "2">
  ];

  let customTeller = [
    (teller "sq", [{
      [](const Node* node) -> bool {
    auto squeeze2_op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto squeeze2_input_x_name = squeeze2_op_desc.Input("X").front();
    auto* scope = const_cast<Node*>(node)->stmt()->op()->scope();

    size_t squeeze2_in_x_rank = scope->FindVar(squeeze2_input_x_name)
                                    ->Get<lite::Tensor>()
                                    .dims()
                                    .size();
    std::vector<int> squeeze2_op_axes =
        squeeze2_op_desc.GetAttr<std::vector<int>>("axes");

    return (squeeze2_in_x_rank == 4 &&
            squeeze2_op_axes == std::vector<int>{2, 3});
  }
    }])
  ];

  let bindTargets = ["kAny"];
}

