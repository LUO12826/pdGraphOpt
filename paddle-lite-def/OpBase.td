
//输入槽标识
def ins;
//输出槽标识
def outs;
//属性标识
def attrs;

//类型标识
class MetaType;

//基本类型
def INT: MetaType;
def STRING: MetaType;
def FLOAT: MetaType;
def DOUBLE: MetaType;
def BOOL: MetaType;

//数组类型
def INTs: MetaType;
def STRINGs: MetaType;
def FLOATs: MetaType;
def DOUBLEs: MetaType;
def BOOLs: MetaType;

//Pattern中的结点
class PMNode {
  // options: roleUnknown, roleInput, roleOutput, roleIntermediate
  // 暂时没有用到
  string role = "roleUnknown";
}

//变量类型
class Var<string vname = "", string vtype = "">: PMNode {
  string name = vname;
  string type = vtype;
  // options:？
  string dataType = "float";

  bit isWeight = false;
  bit isPersistable = false;
}

//Tensor变量类型
class TensorVar<string vname = "", bit isW = false> : Var<vname, "tensor"> {
  let isWeight = isW;
  int dim = -1;
  list<int> shape = [];
  //NCHW, NHWC
  string dataFormat = "";
  list<string> supportedDataFormat = [];
}

class TensorVarArr<string vname = ""> : Var<vname, "tensorArray"> {
  list<TensorVar> tensors = [];
}

//标量类型
class ScalarVar<string vname = ""> : Var<vname, "scalar"> {
  string value = "";
}

//一般的tensor类型数据变量
def tVar: TensorVar;
def tVarArr: TensorVarArr;
def tVarWeight: TensorVar<"", true>;


// op的属性类型
class OpAttr<MetaType t, string val = ""> {
  string name = "";
  MetaType dataType = t;
  string value = val;
}

// 表示属性的默认值
class DefaultAttr<MetaType t, string val>: OpAttr<t, val> {
}

def default: DefaultAttr<INT, "0">;