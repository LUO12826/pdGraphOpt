//
//  TDPattern.h
//  llvm-tblgen
//
//  Created by 骆荟州 on 2022/1/8.
//

#ifndef TDPattern_h
#define TDPattern_h

#include "llvm/TableGen/Error.h"
#include "TDOpArgument.h"
#include <map>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <queue>


namespace PdGraphOpt {

struct AttrOperationBase {
  std::string attrName;
  std::string dataType;

  void setDataType(std::string dType) {
    //全部用小写字母表示类型。
    std::transform(dType.begin(), dType.end(), dType.begin(), ::tolower);
    this->dataType = std::move(dType);
  }
};

/**
 * 记录需要给目标结点设置的属性条目
 */
struct AttrToSet : AttrOperationBase {
  std::string value;
  std::string target;

  std::string getDescription() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "value:" << value << "set for:" << target;
    return ss.str();
  }

  bool checkIntegrity() {
    return attrName != "" && dataType != "" && value != "" && target != "";
  }
};

/**
 * 记录需要从源结点拷贝到目标结点的属性条目
 */
struct AttrToCopy : AttrOperationBase {

  std::string from;
  std::string to;

  std::string getDescription() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "from:" << from << "  to:" << to;
    return ss.str();
  }

  bool checkIntegrity() {
    return attrName != "" && dataType != "" && from != "" && to != "";
  }
};

/**
 * 记录需要施加断言的结点和施加的断言
 */
struct AttrToAssert : AttrOperationBase {

  std::string target;
  std::string value;
  std::string customAssert;
  bool useCustomAssert{false};

  std::string getDescription() {
    std::stringstream ss;
    ss << "attrName:" << attrName << "(" << dataType << ") "
       << "value:" << value << "  target:" << target;
    return ss.str();
  }

  bool checkIntegrity() {
    return attrName != "" && dataType != "" && target != "" && value != "";
  }

};

/**
 * 记录需要对结点应用的自定义teller
 */
struct CustomTeller {
  std::string target;
  std::string name;
  std::string teller;
};


/**
 * 内存中以树的形式存储td中描述的pattern，该类为树的结点类。
 */
class TDPatternNode {
public:
  enum NodeType { Op, Var, Attr };

  virtual ~TDPatternNode() {}
  virtual NodeType getNodeType() = 0;
};

class TDPatternVarNode : public TDPatternNode {
public:
  TDPatternVarNode(std::shared_ptr<TDVariable> var) : var(var) {}

  NodeType getNodeType() override { return NodeType::Var; }

  TDVariable* getVar() const { return var.get(); }

private:
  std::shared_ptr<TDVariable> var;
};

class TDPatternAttrNode: public TDPatternNode {
public:

  TDPatternAttrNode(std::shared_ptr<TDAttribute> attr) : attr(attr) {}

  NodeType getNodeType() override { return NodeType::Attr; }

  TDAttribute* getAttr() const { return attr.get(); }

private:
  std::shared_ptr<TDAttribute> attr;
};

class TDPatternOpNode : public TDPatternNode {
public:
  TDPatternOpNode(std::shared_ptr<TDOperator> &op,
                  std::string& opKey,
                  std::vector<std::unique_ptr<TDPatternNode>> arguments,
                  std::vector<std::string> argNames)
      : op(op), designatedOutputKey(opKey),
        arguments(std::move(arguments)), argNames(std::move(argNames)) {
  }

  NodeType getNodeType() override { return NodeType::Op; }

  TDOperator* getOp() { return op.get(); }

  std::vector<std::unique_ptr<TDPatternNode>> &getArguments() {
    return arguments;
  }

  std::vector<std::string> &getArgNames() { return argNames; }

  std::pair<TDPatternNode *, std::string> getArgAndName(unsigned index) {
    if (index > argNames.size()) {
      llvm::PrintFatalError("Index out of bound while getting arguments "
                            "from `TDPatternOpNode`.");
    }
    return std::make_pair(arguments[index].get(), argNames[index]);
  }

  TDAttribute* getSetOrDefaultAttrAtIndex(unsigned index) {


    if (index >= op->getArgNames().size()) return nullptr;
    if (index >= argNames.size()) {
      return op->getArgumentAsAttrAtIndex(index).get();
    }
    else {
      auto attr =
          static_cast<TDPatternAttrNode*>(arguments[index].get())->getAttr();
      if (attr != nullptr) return attr;
      return op->getArgumentAsAttrAtIndex(index).get();
    }
  }

  std::string getArgSlotNameByActualArgName(std::string actualArgName) {
    auto pos = std::find(argNames.begin(),
                         argNames.end(), actualArgName);
    if (pos == argNames.end()) {
      return "";
    }
    long index = pos - argNames.begin();
    return op->getArgNames()[index];
  }

  long getIndexByActualArgName(std::string name) {
    auto pos = std::find(argNames.begin(),
                         argNames.end(), name);

    if (pos == argNames.end()) {
      return -1;
    }
    return pos - argNames.begin();
  }

  bool isArgTypeCorrect(unsigned index) {
    unsigned slotNum = op->getArgNames().size();
    unsigned realArgNum = argNames.size();
    if (index < 0) return true;
    if (index >= slotNum) return false;
    if (index >= realArgNum && index < slotNum) return true;

    if (op->getArgumentTypeAtIndex(index) == TDOpArgument::variable) {
      if (arguments[index]->getNodeType() == Var) return true;
      if (arguments[index]->getNodeType() == Op) return true;
      return false;
    }
    if (op->getArgumentTypeAtIndex(index) == TDOpArgument::attribute) {
      if (arguments[index]->getNodeType() == Attr) return true;
      return false;
    }
    return false;
  }

  std::string getDesignatedOutputKey() {
    return designatedOutputKey;
  }

private:
  std::shared_ptr<TDOperator> op;
  std::string designatedOutputKey{""};
  std::vector<std::unique_ptr<TDPatternNode>> arguments;
  std::vector<std::string> argNames;
};

/**
 * 用于表示td文件中描述的一个图变换pattern
 */
class TDPattern {

  friend class RecordConverter;
  // pattern的名字
  std::string name;
  //指向源pattern。源pattern和目标pattern都以树的结构表示
  std::unique_ptr<TDPatternOpNode> sourcePatternRoot;
  //指向目标pattern。
  std::unique_ptr<TDPatternOpNode> targetPatternRoot;

  std::vector<TDPatternNode*> sourcePatternTopological;
  
  std::vector<TDPatternNode*> targetPatternTopological;
  //TODO: 存储拓扑排序后的节点？
  std::vector<AttrToCopy> attrsToCopy;
  std::map<std::string, std::vector<AttrToSet>> attrsToSet;
  std::map<std::string, std::vector<AttrToAssert>> attrsToAssert;
  std::map<std::string, std::vector<CustomTeller>> customTellers;

  std::vector<std::string> conditionFlags;

  bool needCopyInputScale{false};

  bool hasMultipleTargets{false};

  int benefitDelta{0};

  std::string kernelName;

  std::vector<std::string> bindTargets;

  std::vector<std::string> excludeTargets;

  std::unordered_set<std::string> varKeysToHold;

  std::unordered_map<std::string, TDPatternNode*> src_outputKeys2Node;

  std::unordered_map<std::string, TDPatternNode*> tgt_outputKeys2Node;

  std::unordered_map<TDPatternNode*, TDPatternNode*> tgt_edges;


  void
  topologicalSort(std::map<TDPatternNode*, std::vector<TDPatternNode*>> &adjTable,
                  std::map<TDPatternNode*, int> &inDegree,
                  int nodeCount) {
    std::queue<TDPatternNode*> q;
    for (auto &&deg : inDegree) {
      if (deg.second == 0)
        q.push(deg.first);
    }

    int count = 0;
    while (!q.empty()) {
      TDPatternNode* v = q.front();
      q.pop();
      targetPatternTopological.push_back(v);
      count++;

      auto& vBeginEdges = adjTable[v];
      for (auto node: vBeginEdges) {

        if(0 == (--inDegree[node]))
          q.push(node);
      }
    }

    if (count < nodeCount) {
      llvm::PrintFatalError("target pattern has circle");
    }
  }

public:
  TDPattern(std::string name,
            std::unique_ptr<TDPatternOpNode> &&sourcePatternRoot,
            std::unique_ptr<TDPatternOpNode> &&targetPatternRoot)
      : name(name), sourcePatternRoot(std::move(sourcePatternRoot)),
        targetPatternRoot(std::move(targetPatternRoot)) {

    auto callback1 = [&](TDPatternNode *node) {
      if (node->getNodeType() == TDPatternNode::Op) {
        auto *opNode = static_cast<TDPatternOpNode*>(node);
        //如果用户为这个op指派了输出key，说明可能要引用这个op的输出，因此这里记录一下
        if (opNode->getDesignatedOutputKey() != "") {
          tgt_outputKeys2Node[opNode->getDesignatedOutputKey()] = opNode;
        }
        if (opNode->getOp()->getType() == "DirectCompute") return;
        //找出那些在target pattern里直接用到的绑定名。
        //这些绑定名对应的变量在source pattern和target pattern中都用到，
        //因此不用标记为intermediate节点，也就不会被移除。
        for (unsigned i = 0; i < opNode->getArgNames().size(); i++) {
          if (i >= opNode->getOp()->getArgNames().size()) break;
          varKeysToHold.insert(opNode->getArgNames()[i]);
        }
      }
    };
    traversePattern(this->targetPatternRoot.get(), callback1);

    std::map<TDPatternNode*, std::vector<TDPatternNode*>> adjTable;
    std::map<TDPatternNode*, int> inDegree;
    int nodeCount = 0;

    //构建邻接表和入度数表
    auto callback2= [&](TDPatternNode *node) {
      nodeCount++;
      inDegree[node] = 0;
      if (node->getNodeType() == TDPatternNode::Op) {
        auto *opNode = static_cast<TDPatternOpNode*>(node);

        auto &opActualArgNames = opNode->getArgNames();
        for (unsigned i = 0; i < opActualArgNames.size(); i++) {
          if (i >= opNode->getOp()->getArgNames().size()) {
            llvm::PrintFatalError(opNode->getOp()->getType() +
                                  ": Op node's actual arguments "
                                  "count bigger than argument slot count");
          }

          if (tgt_outputKeys2Node.count(opActualArgNames[i])) {
            adjTable[tgt_outputKeys2Node[opActualArgNames[i]]].push_back(node);
            inDegree[node]++;
          }
          else {
            adjTable[opNode->getArguments()[i].get()].push_back(node);
            inDegree[node]++;
          }
        }
      }
    };
    traversePattern(this->targetPatternRoot.get(), callback2);

    //target pattern 拓扑排序
    topologicalSort(adjTable, inDegree, nodeCount);
  }



  //getters and setters
  const std::vector<TDPatternNode*>& getTargetTopological() {
    return targetPatternTopological;
  }
  const std::vector<AttrToCopy> &getAttrsToCopy() { return this->attrsToCopy; }

  const std::vector<std::string> &getConditionAttribute() {
    return this->conditionFlags;
  }

  const std::map<std::string, std::vector<AttrToSet>> &getAttrsToSet() {
    return this->attrsToSet;
  }

  const std::map<std::string, std::vector<CustomTeller>> &getCustomTellers() {
    return this->customTellers;
  }

  const std::map<std::string, std::vector<AttrToAssert>> &getAttrsToAssert() {
    return this->attrsToAssert;
  }

  bool getNeedCopyInputScale() { return this->needCopyInputScale; }

  std::string getName() { return name; }

  std::string getNameWithoutPat() {
    std::string n = name;
    return n.erase(name.length() - 3);
  }

  TDPatternOpNode *getSourcePatternRoot() { return sourcePatternRoot.get(); }

  TDPatternOpNode *getTargetPatternRoot() { return targetPatternRoot.get(); }

  std::string getKernelName() {
    return this->kernelName;
  }

  const std::vector<std::string>& getBindTargets() {
    return bindTargets;
  }

  const std::vector<std::string>& getExcludeTargets() {
    return excludeTargets;
  }

  //end getter and setters

  TDPatternOpNode* getTargetOpByDesignatedKey(std::string key) {
    return static_cast<TDPatternOpNode*>(tgt_outputKeys2Node[key]);
  }

  /**
   * paddle中有这么一种情况，Fuser中需要提供一个op的类型，但是这个类型并不直接字符串
   * 硬编码在Fuser中，而是以字符串变量的方式提供。
   * 这样，同一个Fuser就可复用于匹配的结构相同，但有若干个op不同的情形。
   * 这个方法找出所有以字符串变量的方式提供的op类型。
   */
  std::vector<std::string> getVariableOpTypes() {
    std::vector<std::string> res;
    auto callback = [&](TDPatternNode *node) {
      if (node->getNodeType() == TDPatternNode::Op) {
        auto *opNode = static_cast<TDPatternOpNode *>(node);
        if (opNode->getOp()->getTypeIsVariable())
          res.push_back(opNode->getOp()->getType());
      }
    };
    traversePattern(sourcePatternRoot.get(), callback);
    return res;
  }

  bool isVarDirectlyUsedByTargetPattern(std::string varKey) {
    return varKeysToHold.count(varKey) != 0;
  }

  std::string getDescription() {
    std::string rep = "pattern name: " + name + "\n";
    return rep;
  }

private:
  /**
   * 遍历一个pattern树
   */
  void traversePattern(TDPatternNode *root,
                       std::function<void(TDPatternNode *)> nodeCallback) {
    nodeCallback(root);
    if (root->getNodeType() == TDPatternNode::Op) {
      auto *node = static_cast<TDPatternOpNode *>(root);
      for (auto &&arg : node->getArguments()) {
        traversePattern(arg.get(), nodeCallback);
      }
    }
  }

};

} // namespace PdGraphOpt

#endif /* TDPattern_h */
