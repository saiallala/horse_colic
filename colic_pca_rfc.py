import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

data = pd.read_csv('horse.csv')



null = data.isnull().sum()/len(data)*100 #Count the missing value
null = null[null>0]
null.sort_values(inplace=True, ascending=False)
#print(null)
null = null.to_frame()   #convert into dataframe
'''plt.figure(figsize=(20, 10))
plt.style.use('ggplot')
null.plot.bar()
'''

data = data.replace({'no':0, 'yes': 1, 'adult':1, 'young':2})
data.rectal_temp = data.rectal_temp.fillna(value=data.rectal_temp.mode()[0])
data.pulse = data.pulse.fillna(value=data.pulse.mean())
data.respiratory_rate = data.respiratory_rate.fillna(value=data.respiratory_rate.mean())
data.abdomo_protein = data.abdomo_protein.fillna(value=data.abdomo_protein.mode()[0])
data.total_protein = data.total_protein.fillna(value=data.total_protein.mean())
data.packed_cell_volume = data.packed_cell_volume.fillna(value=data.packed_cell_volume.mean())
data.nasogastric_reflux_ph = data.nasogastric_reflux_ph.fillna(value=data.nasogastric_reflux_ph.mean())

#data.drop('nasogastric_reflux_ph',axis=1).values


col = null.index
for i in col:
    data[i] = data[i].fillna(data[i].mode()[0])


from sklearn.preprocessing import LabelEncoder
col = data.columns
for i in col:
    lb = LabelEncoder()
    lb.fit(data[i].values)
    data[i] = lb.transform(data[i].values)

'''
plt.figure(figsize=(8, 5))
sb.countplot(x='outcome', data=data)
plt.show()
'''
#Dependent and Independent attributes
X = data.iloc[:, :-2].values
y = data.iloc[:,26 ].values

#X = data.drop('outcome', axis=1).values
#y = data['outcome'].values

#Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

algo = {'LR':LogisticRegression(),
        'DT':DecisionTreeClassifier(),
        'RFC':RandomForestClassifier(n_estimators=100),
        'SVM':SVC(gamma=0.001),
        'KNN':KNeighborsClassifier(n_neighbors=10)}


for k, v in algo.items():
    model = v
    model.fit(X_train, y_train)
    print('Score of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100))


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
newvar = pca.explained_variance_ratio_


'''#KPCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2,kernel='rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
'''

from matplotlib.colors import ListedColormap
plt.figure(figsize=(8, 5))
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap = ListedColormap(('red', 'green','blue')))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()


from matplotlib.colors import ListedColormap
plt.figure(figsize=(8, 5))
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap = ListedColormap(('red', 'green','blue')))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 800,criterion='entropy',random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
sb.set(font_scale=1)
sb.heatmap(cm, annot=True)
plt.show()


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model,X= X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()

'''
#RFE
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 2)
fit = rfe.fit(X_train, y_train)
X_train  = rfe.fit_transform(X_train,y_train)
X_test = rfe.transform(X_test)
numf = fit.n_features_
supp = fit.support_
rank = fit.ranking_


#Selecting KBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
skb = SelectKBest(score_func=chi2, k=2)
fitskb = skb.fit(X_train, y_train)
X_train  = skb.fit_transform(X_train,y_train)
X_test = skb.transform(X_test)
feat = skb.transform(X_train)
scores = fitskb.scores_
'''


#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{ 'bootstrap': [True,False],
 'max_depth': [11,12,13,],
 'max_features': [2],
 'min_samples_leaf': [2,3,4],
 'min_samples_split': [2, 5],
 }]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Training set')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


#Testset Visual
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Testset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
