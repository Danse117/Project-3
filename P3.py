############################################################
# CISC3140- P3
# Due May 6th 11:59 Pm
#
################################################################

################################################################################################
#Function to compute the prior count of Label(L): Nursery admission recommendation
#Input: trainFile - train_data.dat
#Returns: priorCountsList- the number of counts for the two classes {recommend, not-recom}
#
################################################################################################
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


################################################################################################
# Function to compute the CPT for each feature: P(O|L) P(N|L) P(F|L) P(C|L) P(H|L) P(I|L) P(S|L) P(A|L)
#                                              Parents occupation (O): {usual, pretentious, great_pret}
#                                              Childs Nursery (N): {proper, less_proper, improper, critical, very_crit}
#                                              Family form (F): {complete, completed, incomplete, foster}
#                                              Number of children (C): {1, 2, 3, more}
#                                              Housing (H): {convenient, less_conv, critical}
#                                              Finance (I): {convenient, inconv}
#                                              Social (S): {non-prob, slightly_prob, problematic}
#                                              Health (A): {recommended, priority, not_recom}
# Inputs: trainFile-train_data.dat, 
#         priorCountList - the number of samples for different label values , 
#         feature_name - This is used to identify which feature the CPT is computed for. 
#                        It can take one of the following values:occupation, nursery, family_form, children, housing, finance, social, health  
# Returns: feature_cpt - The CPT for feature given in feature_name
#
################################################################################################
def getFeatureCPT(trainFile, feature_name, priorCountsList):
    feature_cpt= {}
    trainFileObj=open(trainFile,'r')
    trainFileLines=trainFileObj.readlines()

    # Make inital CPT with 0's
    for label in ["recommend", "not_recom"]:
        feature_cpt[label] = {}
        for feature_value in getFeatureValues(feature_name):
            feature_cpt[label][feature_value] = 0

    # Count the occurrences of each feature value for each label
    for trainFileLine in trainFileLines:
        trainFileLines
        features = trainFileLine.strip().split(',')   # Strip to get rid of '\n'
        label = features[-1]
        feature_value = features[getFeatureIndex(feature_name)]
        feature_cpt[label][feature_value] += 1

    # Calculate the CPT
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
################################################################################################
# Function to predict the labels for the samples in the validation file
#   The label is predicted as max( P(L|O,N, F, C, H, I, S, A)) = max(P(O|L) P(N|L) P(F|L) P(C|L) P(H|L) P(I|L) P(S|L) P(A|L) P(L)) 
#                           That is you will treat the label with maximum probability as the prediction from the above formulation
# Inputs: valFile - val_data.dat
#         priorProb - Prior for the label, Nursery admission recommendation
# Returns: predictions - the predicted label for each sample in valFile
#
################################################################################################
def getPredictions(valFile, priorProb, feature1CPT,feature2CPT,feature3CPT,feature4CPT,feature5CPT,feature6CPT,feature7CPT,feature8CPT):
    predictions = []    # Initialize predictions as an empty list
    valFileObj = open(valFile, 'r')     # Open valid file
    valFileLines = valFileObj.readlines()   # Read valid file

    for valFileLine in valFileLines:
        features = valFileLine.strip().split(',')
        recommend_prob = priorProb[0]
        not_recom_prob = priorProb[1]

        # Iterate through zip for each feature and each corresponding CPT
        for feature_name, cpt in zip(["occupation", "nursery", "family_form", 
                                      "children", "housing", "finance", "social", "health"],
                                     [feature1CPT, feature2CPT, feature3CPT, feature4CPT, 
                                      feature5CPT, feature6CPT, feature7CPT, feature8CPT]):
            
            feature_value = features[getFeatureIndex(feature_name)] # Get feature value
            recommend_prob *= cpt["recommend"][feature_value]   
            not_recom_prob *= cpt["not_recom"][feature_value]
        
        if recommend_prob >= not_recom_prob:
            predictions.append("recommend")
        else:
            predictions.append("not_recom")
    valFileObj.close()
    return predictions

if __name__=="__main__":
    trainFile= "train_data.dat"
    valFile="val_data.dat"
    priorCountsList=getPriorCount(trainFile)
    total_count = sum(priorCountsList)
    priorProb = [count / total_count for count in priorCountsList]

    occupationCPT=getFeatureCPT(trainFile,"occupation",priorCountsList)
    nusreryCPT=getFeatureCPT(trainFile,"nursery",priorCountsList)
    familyFormCPT=getFeatureCPT(trainFile,"family_form",priorCountsList)
    childrenCPT=getFeatureCPT(trainFile,"children",priorCountsList)
    housingCPT=getFeatureCPT(trainFile,"housing",priorCountsList)
    financeCPT=getFeatureCPT(trainFile,"finance",priorCountsList)
    socialCPT=getFeatureCPT(trainFile,"social",priorCountsList)
    healthCPT=getFeatureCPT(trainFile,"health",priorCountsList)

    lineCounter=0
    correctCount=0
    totalCount=0
    predicList=getPredictions(valFile, priorProb, occupationCPT, nusreryCPT, familyFormCPT, childrenCPT, housingCPT, financeCPT, socialCPT, healthCPT)
    valFileObj=open(valFile,'r')
    valFileLines=valFileObj.readlines()
    
    for valFileLine in valFileLines:
        label=valFileLine.strip('\n').split(',')[-1]
        if label==predicList[lineCounter]:
            correctCount+=1
        totalCount+=1
        lineCounter+=1
    print("The accuracy of the current predictions is {}".format((correctCount/totalCount)*100))
