//
// Created by 骆荟州 on 2022/4/6.
//

#ifndef LLVM_HELPER_H
#define LLVM_HELPER_H

#include <string>
#include <vector>
#include <sstream>
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

namespace PdGraphOpt {

enum ComparisonOperator {
  GreaterThan = 0,
  Equals = 1,
  LessThan = 2
};

struct ArgumentKeySourceInfo {
  std::string key;
  std::string opKey;

  long index;
  std::string slotName;

  TDOpArgument::ArgumentType argType;
  std::string dataType;
};

/**
 * 将一个std::vector<std::string>转为字符串列表字面量
 */
std::string genListString(std::vector<std::string> &list, std::string separator,
                          std::string head, std::string tail) {
  std::stringstream ss;
  ss << head << " ";
  bool first = true;
  for (std::string &item : list) {
    if (first) {
      ss << item;
      first = false;
    } else
      ss << separator << " " << item;
  }
  ss << " " << tail;
  return ss.str();
}

/**
 * 将驼峰命名法转为蛇形命名法
 */
std::string convertCamelToSnake(std::string camel) {
  if (camel.empty()) return "";
  std::stringstream ss;

  ss << (char)tolower(camel[0]);
  for (unsigned i = 1; i < camel.size(); i++) {
    char c = camel[i];
    if (isupper(c)) {
      ss << "_" << (char)tolower(c);
    }
    else {
      ss << c;
    }
  }
  return ss.str();
}

void genAttrAssertCode(const AttrToAssert& attr, std::string opKey, llvm::raw_ostream &os) {
  std::string dType = attr.dataType == "string" ? "std::string" : attr.dataType;
  if (attr.useCustomAssert) {
    os << llvm::formatv(
        "  {0}->assert_op_attr_satisfied<{1}>(\"{2}\", {3});\n", opKey,
        dType, attr.attrName, attr.customAssert);
  }
  //对于浮点数，考虑误差
  else if (dType == "float" || dType == "double") {
    os << llvm::formatv(
        "  {0}->assert_op_attr_satisfied<{1}>(\"{2}\", "
        "[]({3} attr) { return (std::fabs(attr - {4}) < 1e-5); });\n",
        opKey, attr.dataType, attr.attrName, attr.dataType, attr.value);
  }
  else {
    os << llvm::formatv(
        "  {0}->assert_op_attr<{1}>(\"{2}\", {3});\n", opKey, dType,
        attr.attrName,
        attr.dataType != "std::string" ? attr.value : "\"" + attr.value + "\"");
  }
}


void genSetAttrCode(std::string varName, const AttrToSet &attr, llvm::raw_ostream &os) {
  if (attr.dataType == "string") {
    os << llvm::formatv(
        "  {0}.SetAttr(\"{1}\", std::string(\"{2}\"));\n",
        varName, attr.attrName, attr.value);
  } else {
    os << llvm::formatv("  {0}.SetAttr(\"{1}\", {2});\n",
                        varName, attr.attrName, attr.value);
  }
}


//生成对一个算子输入参数的维度进行检查的代码
std::string EmitInputDimChecker(std::vector<std::string> inputNames,
                                std::vector<int> dims,
                                std::vector<ComparisonOperator> compOp
) {

  std::stringstream os;
  os << "[](const Node* node) -> bool {\n";
  os << "  auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();\n";
  for (auto &name : inputNames) {
    os << llvm::formatv("  auto input_{0}_name = op_desc.Input(\"{1}\").front();\n",
                        name, name).str();
  }
  os << "  auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();\n";
  for (auto &name : inputNames) {
    os << llvm::formatv("  size_t {0}_rank = scope->FindVar(input_{1}_name)->Get<lite::Tensor>().dims().size();\n",
                        name, name).str();
  }

  os << "  bool res = true;\n";

  bool assertAllEqual = true;
  for(int dim : dims) {
    if (dim != 0) assertAllEqual = false;
  }
  static std::string compOps[] = {">", "==", "<"};
  if (assertAllEqual) {
    for (unsigned i = 1; i < inputNames.size(); i++) {
      os << llvm::formatv("  res &= {0}_rank == {1}_rank;\n",
                          inputNames[i - 1],
                          inputNames[i]).str();
    }
  }
  else {
    for (unsigned i = 0; i < inputNames.size(); i++) {
      os << llvm::formatv("  res &= {0}_rank {1} {2};\n",
                          inputNames[i],
                          compOps[compOp[i]],
                          dims[i]).str();
    }
  }

  os << "  return res;\n";
  os << "}\n";

  return os.str();
}


}

#endif // LLVM_HELPER_H
