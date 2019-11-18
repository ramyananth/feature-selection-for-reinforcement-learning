import math
import pandas
import random
import argparse
import numpy as np
import mdptoolbox, mdptoolbox.example
from sklearn.preprocessing import KBinsDiscretizer

# load data set with selected or extracted features, features are discrete
# features are the columns after reward column
def generate_MDP_input(filename):
    original_data = pandas.read_csv(filename)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1

    students_variables = ['student', 'priorTutorAction', 'reward']
    features = feature_name[start_Fidx: len(feature_name)]

    # generate distinct state based on feature
    original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    # original_data['state'] = original_data[features].apply(tuple, axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    # quantify actions
    distinct_acts = ['PS', 'WE']
    Nx = len(distinct_acts)
    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1

    # initialize state transition table, expected reward table, starting state table
    # distinct_states didn't contain terminal state
    student_list = list(data['student'].unique())
    distinct_states = list()
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        # don't consider last row
        temp_states = list(student_data['state'])[0:-1]
        distinct_states = distinct_states + temp_states
    distinct_states = list(set(distinct_states))

    Ns = len(distinct_states)

    # we include terminal state
    start_states = np.zeros(Ns + 1)
    A = np.zeros((Nx, Ns + 1, Ns + 1))
    expectR = np.zeros((Nx, Ns + 1, Ns + 1))

    # update table values episode by episode
    # each episode is a student data set
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        # count the number of transition among states without terminal state
        for i in range(1, (len(row_list) - 1)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

        # count the number of transition from state to terminal
        state1 = distinct_states.index(student_data.loc[row_list[-2], 'state'])
        act = student_data.loc[row_list[-1], 'priorTutorAction']
        A[act, state1, Ns] += 1
        expectR[act, state1, Ns] += float(student_data.loc[row_list[-1], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        A[act, Ns, Ns] = 1
        # generate expected reward
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        # some states only have either PS or WE transition to other state
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act, l, l] = 1

        A[act] = np.divide(A[act].transpose(), np.sum(A[act], axis=1))
        A[act] = A[act].transpose()

    return [start_states, A, expectR, distinct_acts, distinct_states]


def calcuate_ECR(start_states, expectV):
        ECR_value = start_states.dot(np.array(expectV))
        return ECR_value

def output_policy(distinct_acts, distinct_states, vi):
    Ns = len(distinct_states)
    print('Policy: ')
    print('state -> action, value-function')
    # for s in range(Ns):
        # print(distinct_states[s]+ " -> " + distinct_acts[vi.policy[s]] + ", "+str(vi.V[s]))


def calcuate_Q(T, R, V, gamma):
    q = np.zeros((T.shape[1], T.shape[0]))

    for s in range(T.shape[1]):
        for a in range(T.shape[0]):
            r = np.dot(R[a][s], T[a][s])
            Q = r + gamma * np.dot(T[a][s], V)
            q[s][a] = Q

    return q

def output_Qvalue(distinct_acts, distinct_states, Q):
    Ns = len(distinct_states)
    Na = len(distinct_acts)
    print('Q-value in Policy: ')
    print('state -> action, Q value function')
    # for s in range(Ns):
    #     for a in range(Na):
            # print(distinct_states[s] + " -> " + distinct_acts[a] + ", " + str(Q[s][a]))


def calculate_IS(filename, distinct_acts, distinct_states, Q, gamma, theta):
    original_data = pandas.read_csv(filename)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1

    students_variables = ['student', 'priorTutorAction', 'reward']
    features = feature_name[start_Fidx: len(feature_name)]

    # generate distinct state based on feature
    original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    # original_data['state'] = original_data[features].apply(tuple, axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1


    IS = 0
    random_prob = 0.5

    student_list = list(data['student'].unique())
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()


        cumul_policy_prob = 0
        cumul_random_prob = 0
        cumulative_reward = 0

        # calculate Importance Sampling Value for single student
        for i in range(1, len(row_list)):
            state = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            action = student_data.loc[row_list[i], 'priorTutorAction']
            reward = float(student_data.loc[row_list[i], 'reward'])

            Q_PS = Q[state][0]
            Q_WE = Q[state][1]

            #
            diff = Q_PS - Q_WE
            if diff > 60:
                diff = 60
            if diff < -60:
                diff = -60

            if action == 0:  # PS
                prob_logP = 1 / (1 + math.exp(-diff * theta))
            else:  # WE
                prob_logP = 1 / (1 + math.exp(diff * theta))


            cumul_policy_prob += math.log(prob_logP)
            cumul_random_prob += math.log(random_prob)
            cumulative_reward += math.pow(gamma, i-1) * reward
            i += 1

        weight = np.exp(cumul_policy_prob - cumul_random_prob)
        IS_reward = cumulative_reward * weight

        # cap the IS value
        if IS_reward > 300:
            IS_reward = 300
        if IS_reward < -300:
            IS_reward = -300
        IS += IS_reward

    IS = float(IS) / len(student_list)
    return IS

def induce_policy_MDP():

    # extract filename from command
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-input")
    # args = parser.parse_args()
    # filename = args.input

    filename = 'Training_data.csv'

    # load data set with selected or extracted discrete features
    [start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input(filename)

    # apply Value Iteration to run the MDP
    vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)
    vi.run()

    # output policy
    output_policy(distinct_acts, distinct_states, vi)

    # evaluate policy using ECR
    ECR_value = calcuate_ECR(start_states, vi.V)
    print('ECR value: ' + str(ECR_value))


    # calculate Q-value based on MDP
    Q = calcuate_Q(A, expectR, vi.V, 0.9)

    # output Q-value for each state-action pair
    # output_Qvalue(distinct_acts, distinct_states, Q)

    # evaluate policy using Importance Sampling
    IS_value = calculate_IS(filename, distinct_acts, distinct_states, Q, 0.9, 0.1)
    print('IS value: ' + str(IS_value))
    return ECR_value, IS_value

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

def genetic():
    discretize()
    data = pandas.read_csv('MDP_Original_data.csv', index_col=None)
    headers = ["student","currProb","course","session","priorTutorAction","reward","Interaction","hintCount","TotalTime","TotalPSTime","TotalWETime","avgstepTime","avgstepTimePS","stepTimeDeviation","symbolicRepresentationCount","englishSymbolicSwitchCount","Level","probDiff","difficultProblemCountSolved","difficultProblemCountWE","easyProblemCountSolved","easyProblemCountWE","probAlternate","easyProbAlternate","RuleTypesCount","UseCount","PrepCount","MorphCount","OptionalCount","NewLevel","SolvedPSInLevel","SeenWEinLevel","probIndexinLevel","probIndexPSinLevel","InterfaceErrorCount","RightApp","WrongApp","WrongSemanticsApp","WrongSyntaxApp","PrightAppRatio","RrightAppRatio","F1Score","FDActionCount","BDActionCount","DirectProofActionCount","InDirectProofActionCount","actionCount","UseWindowInfo","NonPSelements","AppCount","AppRatio","hintRatio","BlankRatio","HoverHintCount","SystemInfoHintCount","NextStepClickCountWE","PreviousStepClickCountWE","deletedApp","ruleScoreMP","ruleScoreDS","ruleScoreSIMP","ruleScoreMT","ruleScoreADD","ruleScoreCONJ","ruleScoreHS","ruleScoreCD","ruleScoreDN","ruleScoreDEM","ruleScoreIMPL","ruleScoreCONTRA","ruleScoreEQUIV","ruleScoreCOM","ruleScoreASSOC","ruleScoreDIST","ruleScoreABS","ruleScoreEXP","ruleScoreTAUT","cumul_Interaction","cumul_hintCount","cumul_TotalTime","cumul_TotalPSTime","cumul_TotalWETime","cumul_avgstepTime","cumul_avgstepTimeWE","cumul_avgstepTimePS","cumul_symbolicRepresentationCount","cumul_englishSymbolicSwitchCount","cumul_difficultProblemCountSolved","cumul_difficultProblemCountWE","cumul_easyProblemCountSolved","cumul_easyProblemCountWE","cumul_probAlternate","cumul_easyProbAlternate","cumul_RuleTypesCount","cumul_UseCount","cumul_PrepCount","cumul_MorphCount","cumul_OptionalCount","cumul_probIndexinLevel","cumul_InterfaceErrorCount","cumul_RightApp","cumul_WrongApp","cumul_WrongSemanticsApp","cumul_WrongSyntaxApp","cumul_PrightAppRatio","cumul_RrightAppRatio","cumul_F1Score","cumul_FDActionCount","cumul_BDActionCount","cumul_DirectProofActionCount","cumul_InDirectProofActionCount","cumul_actionCount","cumul_UseWindowInfo","cumul_NonPSelements","cumul_AppCount","cumul_AppRatio","cumul_hintRatio","cumul_BlankRatio","cumul_HoverHintCount","cumul_SystemInfoHintCount","cumul_NextStepClickCountWE","cumul_PreviousStepClickCountWE","cumul_deletedApp","CurrPro_NumProbRule","CurrPro_avgProbTime","CurrPro_avgProbTimePS","CurrPro_avgProbTimeDeviationPS","CurrPro_avgProbTimeWE","CurrPro_avgProbTimeDeviationWE","CurrPro_medianProbTime"]


    number_of_years = 10
    number_of_parents = 6
    features_per_parents = 8

    parents = []

    selected_set = []

    list_of_ecr = []
    best_ecr = 0

    # Selection initial generation
    for i in range (0, number_of_parents):
        each_parent = data.iloc[0:, 0:6]
        for i in range (0, features_per_parents):
            a = random.randrange(6,130)
            while(a in selected_set):
                a = random.randrange(6,130)
            selected_set.append(a)
            # if(isinstance(sum(data.iloc[:,a].values),float)):
            each_parent[headers[a]] = pandas.cut(data.iloc[:,a].values, 40, labels=False)
            # else:
            # each_parent[headers[a]] = data.iloc[:, a]
        parents.append(each_parent)
    
    for p in parents:
        p.to_csv('Training_data.csv', index=False)
        ecr, isv = induce_policy_MDP()
        list_of_ecr.append(ecr)
        # if ecr > best_ecr:
        #     best_ecr = ecr
        #     p.to_csv('Best_GA.csv', index=False)
    
    print(list_of_ecr)
    
    for year in range (0,10):
        parents = [parents for _, parents in sorted(zip(list_of_ecr,parents))]
        print(list_of_ecr)
        list_of_ecr = []
        for i in range(2, 5):
            rand = random.randrange(1,3)
            for j in range (0,4):
                a = random.randrange(6,12)
                b = random.randrange(6,12)
                temp = parents[i].iloc[:,a]
                parents[i].iloc[:,a] = parents[5].iloc[:,b]
                #parents[i+1].iloc[:,b] = temp
            parents[i].iloc[:, random.randrange(6,12)] = data.iloc[:, random.randrange(6,130)]
            # parents[i+1].iloc[:, random.randrange(6,14)] = data.iloc[:, random.randrange(6,130)]

        for i in range (0, 2):
            each_parent = data.iloc[0:, 0:6]
            for j in range (0, features_per_parents):
                a = random.randrange(6,130)
                # if(isinstance(sum(data.iloc[:,a].values),float)):
                each_parent[headers[a]] = pandas.cut(data.iloc[:,a].values, 40, labels=False)
                # else:
                # each_parent[headers[a]] = data.iloc[:, a]
            parents[i] = (each_parent)
        best = 0
        for p in parents:
            p.to_csv('Training_data.csv', index=False)
            ecr, isv = induce_policy_MDP()
            if ecr > best:
                best = ecr
                p.to_csv('Best_ECR_new.csv', index=False)
            list_of_ecr.append(ecr)
        

    parents[0].to_csv('Best_Genetic.csv', index=False)


genetic()