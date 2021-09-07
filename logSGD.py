import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
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


# relace "CT" with 1.0 and "T" with 0.0
train_set_sub['round_winner'] = np.where((train_set_sub.round_winner=='CT'),1,train_set_sub.round_winner)
train_set_sub['round_winner'] = np.where((train_set_sub.round_winner=='T'),0,train_set_sub.round_winner)
train_set_sub['round_winner'] = train_set_sub['round_winner'].astype(float)
# print(train_set_sub)
# train_set_sub.info()
# train_set_sub.head(8)
# print(train_set_sub.describe().T)]

# # training

Y = train_set_sub['round_winner']
X = train_set_sub[[i for i in features_and_result if i != 'round_winner']]
hidden_layer_sizes_tuple = (50,50)
max_iter_num = 3000
print("Iteration:",max_iter_num,"Hidden Layer:", hidden_layer_sizes_tuple)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print("training...")
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes_tuple, activation='logistic', solver='adam', max_iter=max_iter_num)
mlp.fit(X_train_std,Y_train)

predict_train = mlp.predict(X_train_std)
predict_test = mlp.predict(X_test_std)

# print(confusion_matrix(Y_train,predict_train))
print("Metrics_Accuracy for Training Set:",accuracy_score(Y_train,predict_train))
print(classification_report(Y_train,predict_train))
print("-"*50)
# print(confusion_matrix(Y_test,predict_test))
print("Metrics_Accuracy for Test Set:",accuracy_score(Y_test,predict_test))
print(classification_report(Y_test,predict_test))
print('The Program Ends')