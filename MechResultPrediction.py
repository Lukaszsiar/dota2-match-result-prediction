import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn import tree
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from tabulate import tabulate

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("dota2Train.csv")
df = df.drop(['EMPTY'],axis=1)
#print(df.info())
df_test = pd.read_csv("dota2Test.csv")
df_test = df_test.drop(['EMPTY'],axis=1)
#print(df_test.info())

for i in range(9000):
	random_index = random.randint(0, 92649)
	df.xs(random_index)['cluster_ID'] = 0

	random_index = random.randint(0, 92649)
	df.xs(random_index)['game_mode'] = 0

	random_index = random.randint(0, 92649)
	df.xs(random_index)['game_type'] = 0

df.cluster_ID.replace(0, np.nan, inplace=True)
df.game_mode.replace(0, np.nan, inplace=True)
df.game_type.replace(0, np.nan, inplace=True)

#print(df.isnull().sum())



#**************************** FILL NAN ****************************************#

df.cluster_ID = df.cluster_ID.fillna(round(df.cluster_ID.mean()))
df.game_mode = df.game_mode.fillna(round(df.game_mode.median()))
df.game_type = df.game_type.fillna(round(df.game_type.mean()))


#**************************** PLOTS ****************************************#
#print(df.isnull().any())
# plt.hist(df.cluster_ID)
# plt.title("cluster_ID")
# plt.show()
# plt.hist(df.game_mode)
# plt.title("game_mode")
# plt.show()
# plt.hist(df.game_type)
# plt.title("game_type")
# plt.show()

# heroes = df.drop(['won', 'cluster_ID', 'game_mode', 'game_type'],axis=1)
# heroes[:] = np.where(heroes == -1, 1, heroes)
# heroes_sum = heroes.sum()
#print(heroes_sum)

# heroes_names = []
# for hero in heroes:
# 	heroes_names.append(hero)
#print(heroes_names)

# plt.bar(heroes_names, heroes_sum.values)
# plt.title("heroes")
# plt.xticks(rotation=75, fontsize=6)
# plt.show()

# sns.boxplot(df['cluster_ID'])
# plt.show()
# sns.boxplot(df['game_mode'])
# plt.show()
# sns.boxplot(df['game_type'])
# plt.show()

# seaplpt = pd.read_csv("dota2Train.csv", usecols =['cluster_ID', 'game_mode'])
# sns.pairplot(seaplpt, kind = 'kde')
# plt.show()


#**************************** SKALING ****************************************#

def skaling(column, new_min, new_max):
	old_min = column.min()
	old_max = column.max()

	old_mean = column.mean()
	old_std = column.std()

	minmax_arr = []
	stand_arr = []

	for xi in column:
		minmax_arr.append(((xi - old_min)/(old_max - old_min)) * (new_max - new_min) + new_min)
		stand_arr.append((xi - old_mean)/old_std)
	
	minmax = pd.Series(minmax_arr)
	stand = pd.Series(stand_arr)
	
	return minmax, stand

minmax_cluster , stand_cluster = skaling(df['cluster_ID'], -3, 3)
minmax_mode , stand_mode = skaling(df['game_mode'], -3, 3)
minmax_type , stand_type = skaling(df['game_type'], -3, 3)


# plt.title('cluster ID')
# plt.hist(df['cluster_ID'], log=True)
# plt.show()
# plt.title('cluster ID standard')
# plt.hist(stand_cluster, log=True)
# plt.show()
# plt.title('cluster ID minmax')
# plt.hist(minmax_cluster, log=True)
# plt.show()

# plt.title('game mode')
# plt.hist(df['game_mode'], log=True)
# plt.show()
# plt.title('game mode standard')
# plt.hist(stand_mode, log=True)
# plt.show()
# plt.title('game mode minmax')
# plt.hist(minmax_mode, log=True)
# plt.show()

# plt.title('game_type')
# plt.hist(df['game_type'], log=True)
# plt.show()
# plt.title('game_type standard')
# plt.hist(stand_type, log=True)
# plt.show()
# plt.title('game_type minmax')
# plt.hist(minmax_type, log=True)
# plt.show()

#**************************** classyfication ****************************************#
print('classyfication')

df['cluster_ID'] = stand_cluster
df['game_mode'] = stand_mode
df['game_type'] = stand_mode

x_train = df.drop(['won'],axis=1)
y_train = df['won']


nwm1 , df_test['cluster_ID'] = skaling(df_test['cluster_ID'], -1, 1)
nwm2 , df_test['game_mode'] = skaling(df_test['game_mode'], -1, 1)
nwm3 , df_test['game_type'] = skaling(df_test['game_type'], -1, 1)

x_test = df_test.drop(['won'],axis=1)
y_test = df_test['won']



def clasyfication(x_train, y_train, x_test, y_test, clf):
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)

	conf_matrix = confusion_matrix(y_test, y_pred)

	tp = conf_matrix[0][0]
	fp = conf_matrix[0][1]
	fn = conf_matrix[1][0]
	tn = conf_matrix[1][1]

	suma = 0
	c = tp + fp + fn + tn
	for i in range(c):
		suma += (2*tp) / (2*tp + fp + fn)
	f1 = suma/c

	# print('accuracy_score: ', accuracy_score(y_test, y_pred))
	# print('confusion_matrix \n TP: ', tp, ' FP: ', fp, '\n FN: ', fn, 'TN: ', tn)
	# print('F1: ', f1,'\n\n')
	# cmp = ConfusionMatrixDisplay(conf_matrix).plot()
	# plt.show()

	return clf.__class__.__name__, accuracy_score(y_test, y_pred), f1, tp, fp, fn, tn



#klasyfikatory
random_forest_clf = RandomForestClassifier()
log_regr_clf = LogisticRegression(random_state=0, multi_class='auto', solver='lbfgs')
nearest_centroid_clf = NearestCentroid()
decision_tree_clf = tree.DecisionTreeClassifier()
out_put_code_clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)



# clasyfication(x_train, y_train, x_test, y_test, out_put_code_clf)


#podzial danych(9265 to len(df)/10)

random_forest_output = []
log_regr_output = []
nearest_centroid_output = []
decision_tree_output = []
out_put_code_output = []

n = 0
for i in range(10):


	test = df[n:n+9265]
	train = df.drop(df.index[n:n+9265])

	x_train = train.drop(['won'],axis=1)
	y_train = train['won']

	x_test = test.drop(['won'],axis=1)
	y_test = test['won']

	n += 9266
	print('\n iteration:', i+1)
	random_forest_output.append(clasyfication(x_train, y_train, x_test, y_test, random_forest_clf))

	log_regr_output.append(clasyfication(x_train, y_train, x_test, y_test, log_regr_clf))

	nearest_centroid_output.append(clasyfication(x_train, y_train, x_test, y_test, nearest_centroid_clf))

	decision_tree_output.append(clasyfication(x_train, y_train, x_test, y_test, decision_tree_clf))

	out_put_code_output.append(clasyfication(x_train, y_train, x_test, y_test, out_put_code_clf))


for i in random_forest_output:
	print(i)
print('\n')

for i in log_regr_output:
	print(i)
print('\n')

for i in nearest_centroid_output:
	print(i)
print('\n')

for i in decision_tree_output:
	print(i)
print('\n')

for i in out_put_code_output:
	print(i)


#**************************** StackingClassifier ****************************************#
# print('estimators')

# # estimators = [('rf', random_forest_clf),
# # 				('log_reg', log_regr_clf), 'dtc', decision_tree_clf]
# estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#              ('knn', KNeighborsClassifier(n_neighbors=5)),
#              ('dt', tree.DecisionTreeClassifier()) ]

# clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# x_train = df.drop(['won'],axis=1)
# y_train = df['won']

# x_test = df_test.drop(['won'],axis=1)
# y_test = df_test['won']

# print('fit')
# clf.fit(x_train, y_train).score(x_test, y_test)
# y_pred = clf.predict(x_test)

# conf_matrix = confusion_matrix(y_test, y_pred)

# tp = conf_matrix[0][0]
# fp = conf_matrix[0][1]
# fn = conf_matrix[1][0]
# tn = conf_matrix[1][1]

# suma = 0
# c = tp + fp + fn + tn
# for i in range(c):
# 	suma += (2*tp) / (2*tp + fp + fn)
# f1 = suma/c

# print('accuracy_score: ', accuracy_score(y_test, y_pred))
# print('confusion_matrix \n TP: ', tp, ' FP: ', fp, '\n FN: ', fn, 'TN: ', tn)
# print('F1: ', f1,'\n\n')
# cmp = ConfusionMatrixDisplay(conf_matrix).plot()
# plt.show()