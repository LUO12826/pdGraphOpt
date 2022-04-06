//
//  TDPattern.h
//  llvm-tblgen
//
//  Created by 骆荟州 on 2022/1/8.
//

#ifndef TDPattern_h
#define TDPattern_h

#include "llvm/TableGen/Error.h"
#include "TDPatternBase.h"
#include "TDOpArgument.h"
#include <map>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <queue>


namespace PdGraphOpt {

/**
 * 用于表示td文件中描述的一个图变换pattern
 */
class TDPattern {
public:
  friend class RecordConverter;

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

  bool isOpRetained(std::string opKey) {
    auto &retainList = extraAssertions["RetainOp"];
    for (auto &cond : retainList) {
      if (cond.sourceNode == opKey) return true;
    }
    return false;
  }

  ///给定一个op的（用户设定的）key，返回它从哪个op保留而来
  std::string getRetainKey(std::string thisOpKey) {
    auto &retainList = extraAssertions["RetainOp"];
    for (auto &cond : retainList) {
      if (cond.targetNode == thisOpKey) return cond.sourceNode;
    }
    return "";
  }

  const std::vector<ExtraCondition> getExtraCondByTargetKey(std::string key) {
    std::vector<ExtraCondition> res;

    for (auto &pair : extraAssertions) {
      for (auto &cond : pair.second) {
        if (cond.targetNode == key) {
          res.push_back(cond);
        }
      }
    }
    return res;
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

  const std::map<std::string, std::vector<ExtraCondition>> &getExtraAssertions() {
    return this->extraAssertions;
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

  TDPatternOpNode*getTargetOpByOutputKey(std::string key) {
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
  // pattern的名字
  std::string name;
  //指向源pattern。源pattern和目标pattern都以树的结构表示
  std::unique_ptr<TDPatternOpNode> sourcePatternRoot;
  //指向目标pattern。
  std::unique_ptr<TDPatternOpNode> targetPatternRoot;

  std::vector<TDPatternNode*> sourcePatternTopological;
  
  std::vector<TDPatternNode*> targetPatternTopological;
  
  std::vector<AttrToCopy> attrsToCopy;
  std::map<std::string, std::vector<AttrToSet>> attrsToSet;
  std::map<std::string, std::vector<AttrToAssert>> attrsToAssert;
  std::map<std::string, std::vector<CustomTeller>> customTellers;
  std::map<std::string, std::vector<ExtraCondition>> extraAssertions;

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
