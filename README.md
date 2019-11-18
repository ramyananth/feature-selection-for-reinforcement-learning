# Feature-selection-for-reinforcement-learning

Given: 


MDP_policy.py
Feature_Table.xlsx
MDP_Original_data.csv
MDP_Original_data


Goal:
Maximize ECR and IS values using RL


To Evaluate:
1. Run "python MDP_policy.py -input Training_data_ECR.csv" for Best ECR
2. Run "python MDP_policy.py -input Training_data_IS.csv" for Best IS



To Execute:
1. Keep all the data files in the same folder
2. Run "python MDP_function.py"
3. Run "python MDP_policy.py -input Training_data.csv" for Best ECR with Training_data.csv generated from 2
4. Run "python MDP_function2.py"
5. Run "python MDP_policy.py -input Training_data.csv" for Best IS with Training_data.csv generated from 4


Note: "python MDP_function2.py" automatically saves two files as best-ecr and isv based on the calculations that were carried out. Step 5 has to done to check if the value generated is actually the best ECR/ISV.
