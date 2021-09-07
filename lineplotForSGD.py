import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data1 = {'Accuracy':[0.747557972621284,0.747312894604782,0.747884743309953],'Iteration':[1000,5000,10000]}
data2 = {'Accuracy':[0.748094810181241,0.748071469417764,0.748223184380361],'Iteration':[1000,5000,10000]}
data3 = {'Accuracy':[0.748153162089932,0.748176502853408,0.74842158086991],'Iteration':[1000,5000,10000]}
data1a = pd.DataFrame(data=data1)
data2a = pd.DataFrame(data=data2)
data3a = pd.DataFrame(data=data3)
print(data1a,data2a,data3a)
fig = plt.figure(figsize=(14, 7))
sns.lineplot('Iteration','Accuracy',ci=None,data=data1a,label='LearningRate=1e-4')
sns.lineplot('Iteration','Accuracy',ci=None,data=data2a,label='LearningRate=1e-5')
sns.lineplot('Iteration','Accuracy',ci=None,data=data3a,label='LearningRate=1e-6')
plt.legend()
plt.title('Logistic Regression With SGD - Training Set Accuracy')
plt.show()