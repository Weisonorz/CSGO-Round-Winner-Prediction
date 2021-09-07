import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

train = pd.read_csv('C:/Users/huangw/OneDrive - Cincinnati Country Day School/Desktop/program1/train.csv')
# train.info()
features_and_result = ['time_left','ct_score','t_score','ct_health','t_health',
						'ct_armor','t_armor','ct_money','t_money','ct_helmets',
						't_helmets','ct_defuse_kits','ct_players_alive','t_players_alive',
						't_weapon_ak47', 't_grenade_flashbang', 't_weapon_sg553',
						'ct_weapon_sg553', 'ct_weapon_ak47', 'ct_grenade_incendiarygrenade',
						'ct_grenade_hegrenade', 'ct_weapon_m4a4', 'ct_weapon_awp',
						'ct_grenade_smokegrenade','ct_grenade_flashbang',
						'round_winner']
train_set_sub = train[features_and_result]
# print(train_set_sub)
# relace "CT" with 1.0 and "T" with 0.0
train_set_sub['round_winner'] = np.where((train_set_sub.round_winner=='CT'),1,train_set_sub.round_winner)
train_set_sub['round_winner'] = np.where((train_set_sub.round_winner=='T'),0,train_set_sub.round_winner)
train_set_sub['round_winner'] = train_set_sub['round_winner'].astype(float)
# print(train_set_sub)
# train_set_sub.info()
# train_set_sub.head(8)
# print(train_set_sub.describe().T)]

# # training
max_iter_list = [1000,5000,10000,150000]
eta_num_list = [0.0000001]
for i in max_iter_list:
	for j in eta_num_list:
		Y = train_set_sub['round_winner']
		X = train_set_sub[[i for i in features_and_result if i != 'round_winner']]

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=400)
		sc = StandardScaler()
		sc.fit(X_train)
		X_train_std = sc.transform(X_train)
		X_test_std = sc.transform(X_test)
		max_itera = i
		eta_num = j
		print("Iteration:",max_itera,"LearningRate:",eta_num)
		# logreg = linear_model.LogisticRegression(max_iter=max_itera)
		# logreg.fit(X_train_std, Y_train)
		# lr_pre = logreg.predict(X_test_std)
		# print("Metrics_Accuracy for  :",accuracy_score(Y_test,lr_pre))
		# print(classification_report(Y_test,lr_pre,target_names=['CT','T']))
		# print(' ends')
		# print("-"*50)
		sgdc = SGDClassifier(loss='log',max_iter=max_itera,learning_rate="constant",eta0=0.0001)
		sgdc.fit(X_train_std,Y_train)
		sgdc_pre_train = sgdc.predict(X_train_std)
		sgdc_pre_test = sgdc.predict(X_test_std)
		print("Metrics_Accuracy for Training Set:",accuracy_score(Y_train,sgdc_pre_train))
		print(classification_report(Y_train,sgdc_pre_train,target_names=['CT','T']))
		print("Metrics_Accuracy for Test Set:",accuracy_score(Y_test,sgdc_pre_test))
		print(classification_report(Y_test,sgdc_pre_test,target_names=['CT','T']))
		print('SGDClassifier ends')
		print("The Program Ends")
# def plot_reult():
# 	num = 1000
# 	x = np.arange(1,num+1)
# 	y = Y_test.values[:num]
# 	Y = sgdc_pre[:num]
# 	print(Y)
# 	print(y)
# 	fig = plt.figure(figsize=(10,5))
# 	sns.lineplot(x=x,y=np.log(Y-y),label='Error',color='coral')
# 	plt.ylim(ymin = -0.0001)
# 	plt.ylim(ymax = 0.0001)
# 	plt.legend()
# 	plt.title('SGD-Regression-Test-Set-Error')
# 	plt.ylabel('winPosibilityForCT')
# 	plt.show()
# plot_reult()
