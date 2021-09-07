import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('C:/Users/huangw/OneDrive - Cincinnati Country Day School/Desktop/program1/train.csv')
# train.info()
features_and_result = ['time_left','ct_score','t_score','ct_health','t_health',
						'ct_armor','t_armor','ct_money','t_money','ct_helmets',
						't_helmets','ct_defuse_kits','ct_players_alive','t_players_alive',
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

def plot_heatmap():
	# making mask matrix
	corr_mat = train_set_sub.corr()
	mask = np.zeros_like(corr_mat,dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	new_train = np.corrcoef(corr_mat.values.T)
	# plotting the correlation data
	sns.set_style('whitegrid')
	plt.figure(figsize = (12,8))
	sns.heatmap(new_train,fmt='.2f',mask=mask,annot=True,
		xticklabels=corr_mat.columns,
		yticklabels=corr_mat.columns,
		cmap=sns.diverging_palette(20, 220, n=250))
	plt.show()
plot_heatmap()