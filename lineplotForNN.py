import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data1 = {'Accuracy':[0.798417496236301,0.813472775333482],'Iteration':[1000,3000]}
data2 = {'Accuracy':[0.819692602145016,0.852521385974535],'Iteration':[1000,3000]}
data3 = {'Accuracy':[0.811605027600452,0.811605027600452],'Iteration':[1000,3000]}
data4 = {'Accuracy':[0.791391926429913,0.83382543442996],'Iteration':[1000,3000]}
data1a = pd.DataFrame(data=data1)
data2a = pd.DataFrame(data=data2)
data3a = pd.DataFrame(data=data3)
data4a = pd.DataFrame(data=data4)
# print(data1a,data2a,data3a)
fig = plt.figure(figsize=(14, 7))
sns.lineplot('Iteration','Accuracy',ci=None,data=data1a,label='LearningRate=1e-4')
sns.lineplot('Iteration','Accuracy',ci=None,data=data2a,label='LearningRate=1e-5')
sns.lineplot('Iteration','Accuracy',ci=None,data=data3a,label='LearningRate=1e-6')
sns.lineplot('Iteration','Accuracy',ci=None,data=data4a,label='LearningRate=1e-6')
plt.legend()
plt.title('Neural Network with Logistic Activation Funciton - Training Set Accuracy')
plt.show()