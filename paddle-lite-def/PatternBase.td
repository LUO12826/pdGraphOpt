//已废弃
def setter;

// (copier "#INPUT_SCALE":$attrName, "float":$dataType, "mul"$srcNode, "fc"$targetNode)
def copier;

//已废弃
def mapper;
//已废弃
def assertor;
// (customAssertor "node":$target, [{...}]:$code)
def customAssertor;

// (teller "node":$target, [{...}]:$code)
def teller;

class ExtraCondition {

  string conditionType = "";

  string sourceNode = "";
  string targetNode = "";

  string dataType = "";
  
  string value1 = "";
  string value2 = "";
  string value3 = "";
}

class RankEquals<string key, string val> : ExtraCondition {
  let targetNode = key;
  let value1 = val;
  let conditionType = "RankEquals";
}

class InputRankAllEquals<string key, string val> : ExtraCondition {
  let targetNode = key;
  let value1 = val;
  let conditionType = "InputRankAllEquals";
}

class RankInRange<string key, string low, string high> : ExtraCondition {
  let targetNode = key;
  let value1 = low;
  let value2 = high;
  let conditionType = "RankInRange";
}

class RetainOp<string src, string tgt>: ExtraCondition {
  let sourceNode = src;
  let targetNode = tgt;
  let conditionType = "RetainOp";
}

////////////////////////////////////////////////////////////////////////////////
//图变换模式
////////////////////////////////////////////////////////////////////////////////

class Pattern<dag source, list<dag> results, int benefitAdded = 0> {

  string patName = "";

  dag sourcePattern = source;

  list<dag> targetPatterns = results;

  int benefitDelta = benefitAdded;

  list<dag> attrToSet = [];
  list<dag> attrToCopy = [];
  list<dag> attrToAssert = [];
  list<dag> attrToMap = [];

  list<dag> customTeller = [];

  list<string> conditionFlags = [];

  list<string> bindTargets = [];
  list<string> excludeTargets = [];

  list<ExtraCondition> extraAssertions = [];

  string kernelName = "";
}

class Pat<dag pattern, dag result, int benefitAdded = 0> :
                                  Pattern<pattern, [result], benefitAdded>;
