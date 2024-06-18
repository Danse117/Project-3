def getPriorCount(trainFile):
    priorCountsList = None    
    priorCounterT = 0 # Count prior recommend
    priorCounterF = 0 # Count prior not_recomm
    trainFileObj = open(trainFile,'r')
    trainFileLines = trainFileObj.readlines()

    for trainFileLine in trainFileLines:
        trainFileLine # Each Line
        prior = trainFileLine.strip().split(',') # prior is a list of each feature in a line, index 8 being the prior 
        # Increment both counters by +1 if True or Else
        if prior[8] == "recommend":
            priorCounterT += 1
        elif prior[8] == "not_recom":
            priorCounterF += 1
    priorCountsList = [priorCounterT, priorCounterF]
    trainFileObj.close()
    return priorCountsList


def getFeatureCPT(trainFile, feature_name, priorCountsList):
    feature_cpt= {}
    trainFileObj=open(trainFile,'r')
    trainFileLines=trainFileObj.readlines()

    # Make inital CPT with 0's
    for label in ["recommend", "not_recom"]:
        feature_cpt[label] = {}
        for feature_value in getFeatureValues(feature_name):  # Each feature's values will be placed in to feature_cpt dict
            feature_cpt[label][feature_value] = 0           
    

    for trainFileLine in trainFileLines:
        trainFileLines
        features = trainFileLine.strip().split(',')
        label = features[-1]
        feature_value = features[getFeatureIndex(feature_name)]
        feature_cpt[label][feature_value] += 1

    # Calculate the conditional probabilities
    for label in ["recommend", "not_recom"]:
        total_count = priorCountsList[0] if label == "recommend" else priorCountsList[1]
        for feature_value in getFeatureValues(feature_name):
            feature_cpt[label][feature_value] /= total_count

    trainFileObj.close()
    return feature_cpt

# Helper function for getting the index value of a feature
def getFeatureIndex(feature_name):
    feature_indices = {"occupation": 0, 
                       "nursery": 1, 
                       "family_form": 2, 
                       "children": 3,
                       "housing": 4,
                        "finance": 5, 
                        "social": 6, 
                        "health": 7}
    return feature_indices[feature_name]
# Helper function for getting the values of each feature
def getFeatureValues(feature_name):
    feature_values = {
        "occupation": ["usual", "pretentious", "great_pret"],
        "nursery": ["proper", "less_proper", "improper", "critical", "very_crit"],
        "family_form": ["complete", "completed", "incomplete", "foster"],
        "children": ["1", "2", "3", "more"],
        "housing": ["convenient", "less_conv", "critical"],
        "finance": ["convenient", "inconv"],
        "social": ["nonprob", "slightly_prob", "problematic"],
        "health": ["recommended", "priority", "not_recom"]
    }
    return feature_values[feature_name]



if __name__=="__main__":
    trainFile= "train_data.dat"
    valFile="val_data.dat"
    priorCountsList=getPriorCount(trainFile)
    priorProb=None
    occupationCPT=getFeatureCPT(trainFile,"occupation",priorCountsList)
    print(priorCountsList)