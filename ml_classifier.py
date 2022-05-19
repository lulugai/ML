#%%
%matplotlib inline
import os
import glob
import xgboost
import numpy as np
from pylab import *
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,Lasso,LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV,KFold, cross_val_predict,StratifiedKFold,train_test_split,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve,confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,plot_confusion_matrix,auc,plot_roc_curve,ConfusionMatrixDisplay,mean_squared_error,roc_auc_score
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.combine import SMOTETomek,SMOTEENN
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.impute import SimpleImputer
import missingno as msno
from sklearn.decomposition import PCA
from joblib import dump, load

RANDOM_NUM = 24

#%%
# 导入特征
df = pd.read_excel("./features_1.xlsx")
features_names = list(df.columns)[3:]
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_NUM, stratify=df['label'])
y_train = df_train[list(df_train.columns)[2:3]].to_numpy().ravel()
X_train = df_train[list(df_train.columns)[3:]].to_numpy()
X_test = df_test[list(df_test.columns)[3:]].to_numpy()
y_test = df_test[list(df_test.columns)[2:3]].to_numpy().ravel()
scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(X_train.shape, y_train.shape, X_test.shape)
#%%
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10))#均方误差
    return(rmse)

#调用LassoCV函数，并进行交叉验证
def lasso_fn(X_train, y_train, X_test, features_names, plot=False):
    model_lasso = LassoCV(alphas = [0.1,0.01,0.001, 0.0001, 0.00001],random_state=RANDOM_NUM,cv=10, max_iter=1000).fit(X_train,y_train)

    #模型所选择的最优正则化参数alpha
    print('model_lasso.alpha_: ',model_lasso.alpha_)

    #各特征列的参数值或者说权重参数，为0代表该特征被模型剔除了
    print('model_lasso.coef_: ', model_lasso.coef_)
    supports = []
    for i in model_lasso.coef_:
        if i != 0:
            supports.append(True)
        else:
            supports.append(False)

    X_train = X_train[:,supports]
    X_test = X_test[:,supports]
    print('after Lasso: ', X_train.shape)
    #输出看模型最终选择了几个特征向量，剔除了几个特征向量
    coef = pd.Series(model_lasso.coef_, features_names)
    print(coef)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    #输出所选择的最优正则化参数情况下的残差平均值，因为是10折，所以看平均值
    print(rmse_cv(model_lasso).mean())


    #画出特征变量的重要程度，这里面选出前3个重要，后3个不重要的举例
    # imp_coef = pd.concat([coef.sort_values().head(7),
    #                      coef.sort_values().tail(7)])
    # imp_coef = pd.concat([coef.sort_values()])
    # print(coef.sort_values())
    if plot:
        plt.rcParams['figure.figsize'] = (10.0, 5.0)
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        coef.sort_values().plot(kind = "barh")
        plt.title("Coefficients in the Lasso Model")
        plt.show() 

    return X_train, y_train, X_test

X_train, y_train, X_test = lasso_fn(X_train, y_train, X_test, features_names)

#%%
def resample(X, y):
    # print('before resampling:', X.shape)
    sm = SMOTE(sampling_strategy='auto', random_state=RANDOM_NUM)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    # print('after resampling:', X_resampled.shape)
    return X_resampled, y_resampled

def Train_model(clf, X, y, name):
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = RANDOM_NUM)
    acc = []
    auc = []
    pre = []
    f1 = []
    sen = []
    spe = []
    #fnr = []
    #fpr = []
    order = 0
    save_path = './models/'
    for train_index, test_index in cv.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[test_index], y[test_index]

        X_resampled, y_resampled = resample(X_train, y_train)
        clf.fit(X_resampled, y_resampled)
        dump(clf, save_path + name + '_' + str(order) + '.joblib')
        order += 1
        y_pred = clf.predict(X_val)
        
        acc.append(accuracy_score(y_val, y_pred))
        auc.append(roc_auc_score(y_val, y_pred)) 
        pre.append(precision_score(y_val, y_pred))
        f1.append(f1_score(y_val, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        sen.append(tp/(tp+fn))
        spe.append(tn/(fp+tn))
        #fnr.append(fn/tp+fn)
        #fpr.append(fp/fp+tn)
    
    result_dict = {'Accuracy': [np.mean(acc)], 
                   'Auc': [np.mean(auc)], 
                   'Sensitivity(TPR)': [np.mean(sen)], 
                   'Specificity(TNR)': [np.mean(spe)],
                   #'FNR': [np.mean(fnr)],
                   #'FPR': [np.mean(fpr)],
                   'Precision': [np.mean(pre)],
                   'F1-score': [np.mean(f1)]
                   }
    result = pd.DataFrame(result_dict, index=[name])
    return result

#%%
#创建分类器
lr_clf = LogisticRegression(C=1, penalty = 'l2', solver= 'newton-cg')
knn_clf = KNeighborsClassifier(n_neighbors=2, p=1, weights='uniform')
svm_clf = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
forest_clf = RandomForestClassifier(random_state=RANDOM_NUM, criterion='gini', max_depth=21, 
                                    max_features='sqrt', min_samples_split=5, 
                                    n_estimators=300)
gra_clf = GradientBoostingClassifier(n_estimators=500)
ada_clf = AdaBoostClassifier(n_estimators=500)
dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=16, min_samples_leaf=1, 
                                min_samples_split=2, splitter= 'random', random_state=RANDOM_NUM)
lgbm_clf = LGBMClassifier(n_estimators=500)
xgb_clf = xgboost.XGBClassifier(n_estimators=500, learning_rate=0.2, 
                                gamma=0.5, max_depth=20, verbosity=0)
en_clf = VotingClassifier(estimators=[('rf', forest_clf),('ada', ada_clf), ('gb', gra_clf), ('xgb', xgb_clf), ('lgbm', lgbm_clf)],
                         voting='soft',weights=[5, 5, 2, 5, 5])

#训练分类器并输出结果
clfs = {'Logistic Regression': lr_clf, 'KNN': knn_clf, 'SVM': svm_clf, 'Decision Tree': dt_clf, 'Random Forest': forest_clf, 
        'AdaBoosting': ada_clf, 'Gradient Boosting': gra_clf, 'LGBM': lgbm_clf, 'XGBoost': xgb_clf, 'Ensemble': en_clf}
results = None
for clf_name in clfs:
    results = pd.concat([results, Train_model(clfs[clf_name], X_train, y_train, clf_name)])
display(results)
# result3.to_excel('./result.xlsx', index=False)

#%%
def test_model(X_test, y_test, clf_name, n_splits=5):
    acc = []
    auc = []
    pre = []
    f1 = []
    sen = []
    spe = []
    for i in range(n_splits):
        clf = load('./models/' + clf_name + '_' + str(i) + '.joblib')
        y_pred = clf.predict(X_test)
        
        acc.append(accuracy_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred)) 
        pre.append(precision_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sen.append(tp/(tp+fn))
        spe.append(tn/(fp+tn))
        #fnr.append(fn/tp+fn)
        #fpr.append(fp/fp+tn)
    
    result_dict = {'Accuracy': [np.mean(acc)], 
                   'Auc': [np.mean(auc)], 
                   'Sensitivity(TPR)': [np.mean(sen)], 
                   'Specificity(TNR)': [np.mean(spe)],
                   #'FNR': [np.mean(fnr)],
                   #'FPR': [np.mean(fpr)],
                   'Precision': [np.mean(pre)],
                   'F1-score': [np.mean(f1)]
                   }
    
    result = pd.DataFrame(result_dict, index=[clf_name])
    return result

test_results =  None
for clf in clfs:
    test_results = pd.concat([test_results, test_model(X_test, y_test, clf, 5)])
display(test_results)

#%%
X_resam, y_resam = resample(X_train, y_train)
#逻辑回归分类器调参
lr = LogisticRegression()
parameters = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000,2000],
    'penalty':['l2','l1'],
    'solver':["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}
grid_search_lr = GridSearchCV(lr, parameters, cv = 5, n_jobs = -1)
grid_search_lr.fit(X_resam, y_resam)
print("The best params are:",grid_search_lr.best_params_)
print("The best accuracy is:", grid_search_lr.best_score_)

#%%
#knn分类器调参
knn = KNeighborsClassifier()
parameters = {
    'n_neighbors':[1,2,3,4,5,8],
    'weights':["uniform",  "distance"],
    'p':[1,2,3,4]
}
grid_search_knn = GridSearchCV(knn, parameters, cv = 5, n_jobs = -1)
grid_search_knn.fit(X_resam, y_resam)
print("The best params are:",grid_search_knn.best_params_)
print("The best accuracy is:", grid_search_knn.best_score_)

#%%
#SVM分类器调参
svc = SVC()
parameters = {
    'kernel' : ['linear', 'rbf'],
    'gamma' : [0.0001, 0.001, 0.01, 0.1, 1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameters,cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(X_resam, y_resam)
print("The best params are:",grid_search.best_params_)
print("The best accuracy is:", grid_search.best_score_)

#%%
#决策树分类器调参
dtc = DecisionTreeClassifier()
parameters = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 32, 1),
    'min_samples_leaf' : range(1, 10, 1),
    'min_samples_split' : range(2, 10, 1),
    'splitter' : ['best', 'random']
}
grid_search_dt = GridSearchCV(dtc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_dt.fit(X_resam, y_resam)
print("The best params are:",grid_search_dt.best_params_)
print("The best accuracy is:", grid_search_dt.best_score_)

#%%
#随机森林分类器调参
forest = RandomForestClassifier()
parameters = {
    'criterion' : ['gini', 'entropy'],
    'max_features' : ["auto", "sqrt", "log2"],
    'max_depth': [19,20,21],
    'min_samples_leaf' : [1,2,3],
    'min_samples_split' : [4,5,6],
    'n_estimators': [300,500,700]
}
grid_search_forest = GridSearchCV(forest, parameters, cv = 5, n_jobs = -1,verbose = 1)
grid_search_forest.fit(X_resam, y_resam)
print("The best params are:",grid_search_forest.best_params_)
print("The best accuracy is:", grid_search_forest.best_score_)








