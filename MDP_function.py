import csv
import sys
import random
import pandas
import numpy as np
from pdb import set_trace
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm,preprocessing,tree
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def discretize(file='MDP_Original_Data.csv'):
    originalData = pandas.read_csv(file)
    featureName = list(originalData)
    rewardIndex = featureName.index('reward')
    startIndex = rewardIndex + 1
    features = featureName[startIndex: len(featureName)]+['priorTutorAction']
    for key in features:
        if len(originalData[key].unique())>6:
            thres = originalData[key].median()
            for i,x in enumerate(originalData[key]):
                if x> thres:
                    originalData.set_value(i,key,1)
                else:
                    originalData.set_value(i,key,0)

    originalData.to_csv('dis_data.csv',index=False)

url = "MDP_Original_data.csv"
names = ['student', 'currProb', 'course', 'session', 'priorTutorAction', 'reward', 'Interaction', 'hintCount', 'TotalTime', 'TotalPSTime', 'TotalWETime', 'avgstepTime', 'avgstepTimePS', 'stepTimeDeviation', 'symbolicRepresentationCount', 'englishSymbolicSwitchCount', 'Level', 'probDiff', 'difficultProblemCountSolved', 'difficultProblemCountWE', 'easyProblemCountSolved', 'easyProblemCountWE', 'probAlternate', 'easyProbAlternate', 'RuleTypesCount', 'UseCount', 'PrepCount', 'MorphCount', 'OptionalCount', 'NewLevel', 'SolvedPSInLevel', 'SeenWEinLevel', 'probIndexinLevel', 'probIndexPSinLevel', 'InterfaceErrorCount', 'RightApp', 'WrongApp', 'WrongSemanticsApp', 'WrongSyntaxApp', 'PrightAppRatio', 'RrightAppRatio', 'F1Score', 'FDActionCount', 'BDActionCount', 'DirectProofActionCount', 'InDirectProofActionCount', 'actionCount', 'UseWindowInfo', 'NonPSelements', 'AppCount', 'AppRatio', 'hintRatio', 'BlankRatio', 'HoverHintCount', 'SystemInfoHintCount', 'NextStepClickCountWE', 'PreviousStepClickCountWE', 'deletedApp', 'ruleScoreMP', 'ruleScoreDS', 'ruleScoreSIMP', 'ruleScoreMT', 'ruleScoreADD', 'ruleScoreCONJ', 'ruleScoreHS', 'ruleScoreCD', 'ruleScoreDN', 'ruleScoreDEM', 'ruleScoreIMPL', 'ruleScoreCONTRA', 'ruleScoreEQUIV', 'ruleScoreCOM', 'ruleScoreASSOC', 'ruleScoreDIST', 'ruleScoreABS', 'ruleScoreEXP', 'ruleScoreTAUT', 'cumul_Interaction', 'cumul_hintCount', 'cumul_TotalTime', 'cumul_TotalPSTime', 'cumul_TotalWETime', 'cumul_avgstepTime', 'cumul_avgstepTimeWE', 'cumul_avgstepTimePS', 'cumul_symbolicRepresentationCount', 'cumul_englishSymbolicSwitchCount', 'cumul_difficultProblemCountSolved', 'cumul_difficultProblemCountWE', 'cumul_easyProblemCountSolved', 'cumul_easyProblemCountWE', 'cumul_probAlternate', 'cumul_easyProbAlternate', 'cumul_RuleTypesCount', 'cumul_UseCount', 'cumul_PrepCount', 'cumul_MorphCount', 'cumul_OptionalCount', 'cumul_probIndexinLevel', 'cumul_InterfaceErrorCount', 'cumul_RightApp', 'cumul_WrongApp', 'cumul_WrongSemanticsApp', 'cumul_WrongSyntaxApp', 'cumul_PrightAppRatio', 'cumul_RrightAppRatio', 'cumul_F1Score', 'cumul_FDActionCount', 'cumul_BDActionCount', 'cumul_DirectProofActionCount', 'cumul_InDirectProofActionCount', 'cumul_actionCount', 'cumul_UseWindowInfo', 'cumul_NonPSelements', 'cumul_AppCount', 'cumul_AppRatio', 'cumul_hintRatio', 'cumul_BlankRatio', 'cumul_HoverHintCount', 'cumul_SystemInfoHintCount', 'cumul_NextStepClickCountWE', 'cumul_PreviousStepClickCountWE', 'cumul_deletedApp', 'CurrPro_NumProbRule', 'CurrPro_avgProbTime', 'CurrPro_avgProbTimePS', 'CurrPro_avgProbTimeDeviationPS', 'CurrPro_avgProbTimeWE', 'CurrPro_avgProbTimeDeviationWE', 'CurrPro_medianProbTime']
dataframe = read_csv(url, names=names, dtype=object)
array = dataframe.values
X = array[1:,6:130]
y = array[1:,5]
ylist = y.flatten()
yfinal = y.ravel()

knn = KNeighborsClassifier(n_neighbors=4)

# Sequential Forward Selection
def callSFS():
    sfs1 = SFS(knn, 
               k_features=8, 
               forward=True, 
               floating=False, 
               verbose=2,
               scoring='accuracy',
               cv=0)

    sfs1 = sfs1.fit(X, yfinal)
    print sfs1.subsets_


# Genetic Algorithm
def geneticAlgorithm(file='dis_data.csv'):
    # To print the results of the ECR as and when it is produced, import a copy of the MDP_policy
    from MDP_policy2 import induce_policy_MDP2
    def eval(x):
        if not x in finalList:
            try:
                finalList[x]=induce_policy_MDP2(originalData, list(x))
            except:
                finalList[x]=0
        return finalList[x]

    # Call SFS function if need be - Modify bins based on the data set 
    # Used Silhoutte method to specify n values as 4

    # Uncomment for ECR
    # callSFS()

    originalData = pandas.read_csv(file)
    featureName = list(originalData)
    rewardIndex = featureName.index('reward')
    startIndex = rewardIndex + 1
    features = np.array(featureName[startIndex: len(featureName)])
    featuresNeeded=8

    # geneticAlgorithm parameters - modified to get better ECR values (W.I.P)
    numberOfGenerations=200
    numberOfactualParentSetMating=50
    mutationValue=0.1
    finalList={}

    # Initial candidates
    currentList={}
    for i in range(numberOfGenerations):
        x=tuple(set(np.random.choice(features, featuresNeeded, replace=False)))
        currentList[x]=eval(x)
    currentListSize = len(finalList)
    
    # Generations
    for generation in range(numberOfactualParentSetMating):
        # Selection based on Parameter
        # Sort in descending and take the highest value - Greedy Approach
        appendedFeatureSet=np.array(list(finalList.keys()))[np.argsort(finalList.values())[::-1][:int(numberOfGenerations/10)]]
        appendedFeatureSet=map(set,appendedFeatureSet)
        appendedFeatureSet=map(tuple,appendedFeatureSet)
        # Crossover Value
        actualParentSet=set([])
        for what in appendedFeatureSet:
            actualParentSet=actualParentSet|set(what)
        actualParentSet=tuple(actualParentSet)
        currentList={}
        for j in range(numberOfGenerations):
            if random.random()<mutationValue:
                ## mutationValue
                new=tuple(set(np.random.choice(features,featuresNeeded,replace=False)))
            else:
                new = tuple(set(np.random.choice(actualParentSet,featuresNeeded,replace=False)))
            currentList[new]=eval(new)
        if currentListSize==len(finalList):
            break
        currentListSize=len(finalList)

    # Sort in descending order and cextract the feature with the highest value
    best=tuple(set(np.array(list(finalList.keys()))[np.argsort(finalList.values())[::-1][0]]))
    print(best)
    print(finalList[best])
    finalTrainingSet = originalData[featureName[:startIndex]+list(best)]
    finalTrainingSet.to_csv('Training_data.csv',index=False)

if __name__ == "__main__":
    #Comment the next line after first iteration - Discretizing Value is not going to change
    discretize()
    geneticAlgorithm()