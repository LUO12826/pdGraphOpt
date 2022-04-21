////////////////////////////////////////////////////////////////////////////////
//算子
////////////////////////////////////////////////////////////////////////////////

include "OpBase.td"

class Op<string opType, string oPkey>: PMNode {

  string type = opType;
  string key = oPkey;

  string summary = "";
  string description = "";

  dag arguments = (ins);
  dag results = (outs);

  list<OpAttr> attributes = [];
  string dataFormat = "";

    //考虑弃用
  bit typeIsVariable = false;
}


class MulOp<string key = ""> : Op<"mul", key> {
  let arguments = (ins 
    tVar:$X,
    tVar:$Y,
    DefaultAttr<INT, "1">:$x_num_col_dims,
    DefaultAttr<INT, "1">:$y_num_col_dims
  );

  let results = (outs tVar:$Out);
}


class MatMulOp<string key = ""> : Op<"matmul", key> {
  let arguments = (ins
    tVar:$X,
    tVar:$Y,
    DefaultAttr<FLOAT, "1.0">:$alpha,
    DefaultAttr<BOOL, "false">:$transpose_X,
    DefaultAttr<BOOL, "false">:$transpose_Y
  );

  let results = (outs tVar:$Out);
}


class MatMulV2Op<string key = ""> : Op<"matmul_v2", key> {
  let arguments = (ins  tVar:$X,
                        tVar:$Y,
                        DefaultAttr<BOOL, "false">:$trans_x,
                        DefaultAttr<BOOL, "false">:$trans_y,
                        DefaultAttr<BOOL, "false">:$enable_int8,
                        DefaultAttr<FLOAT, "1.0">:$alpha
                  );

  let results = (outs tVar:$Out);
}


class BatchNormOp<string key = "">: Op<"batch_norm", key> {

  let arguments = (ins tVar:$X,
                       tVar:$Scale,
                       tVar:$Bias,
                       tVar:$Mean,
                       tVar:$Variance,
                       DefaultAttr<FLOAT, "0.00001">:$epsilon);

  let results = (outs tVar:$Y,
                      tVar:$MeanOut,
                      tVar:$VarianceOut,
                      tVar:$SavedMean,
                      tVar:$SavedVariance);
}


class ScaleOp<string key = ""> : Op<"scale", key> {
  let arguments = (ins
    tVar:$X,
    DefaultAttr<INT, "1">:$start_axis,
    DefaultAttr<INT, "1">:$stop_axis
  );

  let results = (outs tVar:$Out, tVar:$XShape);
}


class FlattenContiguousRangeOp<string key = ""> :
                            Op<"flatten_contiguous_range", key> {
  let arguments = (ins
    tVar:$X,
    DefaultAttr<FLOAT, "1.0">:$scale,
    DefaultAttr<FLOAT, "0.0">:$bias,
    DefaultAttr<FLOAT, "6.0">:$alpha,
    DefaultAttr<BOOL, "true">:$bias_after_scale,
    DefaultAttr<BOOL, "false">:$fuse_relu,
    DefaultAttr<STRING, "">:$activation_type,
    DefaultAttr<BOOL, "false">:$fuse_scaleact,
    DefaultAttr<FLOAT, "1.0">:$scale1,
    DefaultAttr<FLOAT, "0.0">:$bias1
  );

  let results = (outs tVar:$Out);
}


class TransposeOp<string key = ""> : Op<"transpose", key> {
  let arguments = (ins
    tVar:$X,
    DefaultAttr<INTs, "{}">:$axis,
    DefaultAttr<BOOL, "false">:$use_mkldnn,
    DefaultAttr<STRING, "AnyLayout">:$data_format
  );
  let results = (outs tVar:$Out);
}


class Transpose2Op<string key = ""> : Op<"transpose2", key> {
  let arguments = (ins
    tVar:$X,
    DefaultAttr<INTs, "{}">:$axis,
    DefaultAttr<BOOL, "false">:$use_mkldnn,
    DefaultAttr<STRING, "AnyLayout">:$data_format
  );
  let results = (outs tVar:$Out, tVar:$XShape);
}


class SoftmaxOp<string key = "">: Op<"softmax", key> {
  let arguments = (ins
    tVar:$X,
    DefaultAttr<INT, "-1">:$axis,
    DefaultAttr<BOOL, "false">:$use_cudnn
  );
  let results = (outs tVar:$Out);
}


class Squeeze2Op<string key = "">: Op<"squeeze2", key> {
  let arguments = (ins
    tVar:$X,
    tVar:$X,
    DefaultAttr<BOOL, "false">:$inplace
  );
  let results = (outs tVar:$Out, tVar:$XShape);
}


class Reshape2Op<string key = "">: Op<"reshape2", key> {
  let arguments = (ins
    tVar:$X,
    tVar:$Shape,
    tVarArr:$ShapeTensor,
    DefaultAttr<INTs, "{}">:$shape,
    DefaultAttr<BOOL, "false">:$inplace
  );
  let results = (outs tVar:$Out, tVar:$XShape);
}


class Conv2dOp<string key = ""> : Op<"conv2d", key> {

  let arguments = (ins tVar:$Input, tVar:$Filter, tVar:$Bias);
  let results = (outs tVar:$Output);
}


class ElementwiseAddOp<string key = ""> : Op<"elementwise_add", key> {

  let arguments = (ins
    tVar:$X,
    tVar:$Y,
    DefaultAttr<INT, "-1">:$axis
  );

  let results = (outs tVar:$Out);
}


class ElementwiseAddActivationOp<string key = "">:
              Op<"fusion_elementwise_add_activation", key> {

  let arguments = (ins
    tVar:$X,
    tVar:$Y,
    DefaultAttr<STRING, "relu">:$act_type
  );

  let results = (outs tVar:$Out);
}


class ElementwiseOp<string typ, string key = ""> : Op<typ, key> {
  let arguments = (ins
    tVar:$X,
    tVar:$Y,
    DefaultAttr<INT, "-1">:$axis
  );

  let results = (outs tVar:$Out);
}


class ElementwiseActivationOp<string typ, string key = "">: Op<typ, key> {
  let arguments = (ins
    tVar:$X,
    tVar:$Y,
    DefaultAttr<STRING, "relu">:$act_type
  );

  let results = (outs tVar:$Out);
}


class FcOp<string key = ""> : Op<"fc", key> {

  let arguments = (ins
    tVar:$W,
    tVar:$Input,
    tVar:$Bias,
    DefaultAttr<INT, "1">:$in_num_col_dims,
    DefaultAttr<STRING, "">:$activation_type
  );

  let results = (outs tVar:$Out);
}


class ActivationVarOp<string key = "">: Op<"activation_", key> {
  let typeIsVariable = true;
  let arguments = (ins tVar:$Input);
  let results = (outs tVar:$Out);
}


class GenericActivationOp<string typ, string key = "">: Op<typ, key> {
  let arguments = (ins tVar:$Input);
  let results = (outs tVar:$Out);
}


class ReluOp<string key = "">: Op<"relu", key> {
  let arguments = (ins tVar:$Input);
  let results = (outs tVar:$Out);
}


////////////////////////////////////////////////////////////////////////////////
// DirectCompute ops
////////////////////////////////////////////////////////////////////////////////

/*
 * DirectCompute 类不是真正的算子，而是用于指示编译器在编译期进行一些计算
 * 因为在进行融合时，可能要对参数进行重新计算
 * 比如Conv + Scale在融合的时候就对卷积核进行了Scale
 */
class DirectCompute<string key = "">: Op<"DirectCompute", key> {
  string directComputeType = "";
}


def DirectEleWiseAdd: DirectCompute<> {
  let directComputeType = "DirectEleWiseAdd";

  let arguments = (ins
    tVar:$X,
    tVar:$Y
  );

  let results = (outs tVar:$Out);
}


def DirectEleWiseSub: DirectCompute<> {
  let directComputeType = "DirectEleWiseSub";

  let arguments = (ins
    tVar:$X,
    tVar:$Y
  );

  let results = (outs tVar:$Out);
}


def DirectEleWiseMul: DirectCompute<> {
  let directComputeType = "DirectEleWiseMul";

  let arguments = (ins
    tVar:$X,
    tVar:$Y
  );

  let results = (outs tVar:$Out);
}


def DirectEleWiseDiv: DirectCompute<> {
  let directComputeType = "DirectEleWiseDiv";

  let arguments = (ins
    tVar:$X,
    tVar:$Y
  );

  let results = (outs tVar:$Out);
}


def DirectEleWiseSqrt: DirectCompute<> {
  let directComputeType = "DirectEleWiseSqrt";

  let arguments = (ins
    tVar:$X
  );

  let results = (outs tVar:$Out);
}

// row wise
def DirectRowWiseMul: DirectCompute<> {
  let directComputeType = "DirectRowWiseMul";

  let arguments = (ins
    tVar:$X,
    tVar:$Y,
    DefaultAttr<INT, "1">:$x_num_col_dims
  );

  let results = (outs tVar:$Out);
}

////////////////////////////////////////////////////////////////////////////////
// end DirectCompute ops
////////////////////////////////////////////////////////////////////////////////