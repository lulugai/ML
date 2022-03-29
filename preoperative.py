#%%
from collections import Counter
from tokenize import group
from unicodedata import name
from cv2 import selectROI
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection  import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import time 
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from imblearn.ensemble import EasyEnsembleClassifier
#%%
names = ['Center','Age','HP=1','DM=1','CVD=1','Pesonal history','BMI＞28=1','CA125＞35=1','Adnexal_involvment','Myometrial_invasion','Lymph_node','	Grade','Pathology','Cervix','Extrauterine =1','Adnexa']
X_names = ['Age','HP=1','DM=1','CVD=1','Pesonal history','BMI＞28=1','CA125＞35=1','Adnexal_involvment','Myometrial_invasion','Lymph_node','Grade','Pathology','Cervix','Extrauterine =1']
frame = pd.read_csv('./2022.3.5_preoperative.csv',dtype=int, names=names )
c_name = frame.columns[-1]
center_name = frame.columns[0]
frame = frame.sort_values(by=c_name)
# print(frame.tail())

valid_nums = 126
valid_neg, valid_pos = frame[:valid_nums], frame[-valid_nums:]
valid = pd.concat([valid_neg, valid_pos], ignore_index=True)
train = frame[valid_nums:len(frame) - valid_nums]
# print(sum(valid[c_name]), sum(train[c_name]))

train_y = train[c_name]
valid_y = valid[c_name]
train = train.drop([center_name, c_name], axis=1)
valid = valid.drop([c_name, center_name], axis=1)
#%%
# Multinomial Naive Bayes Classifier 
def mul_naive_bayes_classifier(train_x, train_y): 
    model = MultinomialNB() 
    model.fit(train_x, train_y) 
    return model 

def naive_bayes_classifier(train_x, train_y): 
    model = GaussianNB(priors=None)
    model.fit(train_x, train_y) 
    return model 

# KNN Classifier 
def knn_classifier(train_x, train_y): 
    model = KNeighborsClassifier() 
    model.fit(train_x, train_y) 
    return model 

# Logistic Regression Classifier 
def logistic_regression_classifier(train_x, train_y): 
    model = LogisticRegression(random_state=0,class_weight={0:0.2,1:0.8}) 
    model.fit(train_x, train_y) 
    return model 
  
# Random Forest Classifier 
def random_forest_classifier(train_x, train_y): 
    model = RandomForestClassifier(n_estimators=98,max_depth=7,min_samples_leaf=9,max_features = 5,random_state=0,class_weight={0:0.2,1:0.8})
    model.fit(train_x, train_y) 
    return model 
  
# Decision Tree Classifier 
def decision_tree_classifier(train_x, train_y): 
    model = DecisionTreeClassifier() 
    model.fit(train_x, train_y) 
    return model 
  
# GBDT(Gradient Boosting Decision Tree) Classifier 
def gradient_boosting_classifier(train_x, train_y): 
    model = GradientBoostingClassifier(learning_rate=0.1,n_estimators = 21,random_state=0,max_depth = 4,min_samples_split = 10)  
    model.fit(train_x, train_y) 
    return model 

  
# SVM Classifier 
def svm_classifier(train_x, train_y): 
    model = SVC(probability=True,random_state=0,class_weight={0:0.2,1:0.8}) 
    model.fit(train_x, train_y) 
    return model 

def adaboost_classifier(train_x, train_y): 
    model = AdaBoostClassifier(n_estimators = 5,random_state = 0)
    model.fit(train_x, train_y)
    return model

def bagging_classifier(train_x, train_y): 
    model = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True)
    model.fit(train_x,train_y)
    return model

def multi_layer_perceptron_classifier(train_x, train_y): 
    model = MLPClassifier(max_iter=10000)
    model.fit(train_x,train_y)
    return model

# test_classifiers = ['LR(Logistic回归)','SVM(支持向量机)','RF(随机森林)','Adaboost','DT(决策树)','KNN(最近邻)','NB(高斯朴素贝叶斯)','MNB(多项式分布朴素贝叶斯)','MLP(多层感知机)','GBDT(梯度提升决策树)','Bagging'] 
test_classifiers = ['LR(Logistic回归)','SVM(支持向量机)','RF(随机森林)','Adaboost','DT(决策树)','KNN(最近邻)','NB(高斯朴素贝叶斯)','MLP(多层感知机)','GBDT(梯度提升决策树)'] 
#test_classifiers = ['LR(Logistic回归)','RF(随机森林)','Adaboost','GBDT(梯度提升决策树)']
classifiers = {
    'GBDT(梯度提升决策树)':gradient_boosting_classifier,
    'Adaboost':adaboost_classifier,
    'Bagging':bagging_classifier,
    'NB(高斯朴素贝叶斯)':naive_bayes_classifier,  
    'MNB(多项式分布朴素贝叶斯)':mul_naive_bayes_classifier,
    'KNN(最近邻)':knn_classifier,
    'LR(Logistic回归)':logistic_regression_classifier,
    'RF(随机森林)':random_forest_classifier,
    'DT(决策树)':decision_tree_classifier,
    'SVM(支持向量机)':svm_classifier,
    'MLP(多层感知机)':multi_layer_perceptron_classifier
}
X_train = np.array(train)
y_train = np.array(train_y)
X_test = np.array(valid)
y_test = np.array(valid_y)
print('X_train shape: ',X_train.shape)
print('X_test shape: ',X_test.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
n_splits=10
skf = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=0)
#%%

# LR = LogisticRegression(penalty='l2') 
# rfe = RFE(estimator=LR,n_features_to_select=8)
# fit = rfe.fit(X_train,y_train)
# print(fit.support_)
# print(fit.ranking_)
# X_train = X_train[:, fit.support_]
# X_test = X_test[:, fit.support_]
# print(X_train.shape)

# X = SelectKBest(f_classif, k=7).fit_transform(X, y)
# print(X.shape)

def resample(X, y):
    from imblearn.combine import SMOTEENN,SMOTETomek
    # smote_enn = SMOTEENN(random_state=23)
    smote_enn = SMOTETomek(random_state=0)
    print(X.shape)
    print(y.shape)
    print('样本类别:')
    print(sorted(Counter(y).items()))
    X_reset, y_reset = smote_enn.fit_resample(X, y)
    print('SMOTE + ENN后样本类别:')
    print(sorted(Counter(y_reset).items()))
    return X_reset, y_reset


# 1.标准化处理
scaler = StandardScaler()

X_train, y_train = resample(X_train, y_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
# select = SelectKBest(f_classif, k='all')
# X_train = select.fit_transform(X_train, y_train)
# X_test = select.transform(X_test)
# print(X_train.shape)
# print(X_test.shape)


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10))#均方误差
    return(rmse)

#调用LassoCV函数，并进行交叉验证
def lasso_fn(X_train, y_train, X_test):
    model_lasso = LassoCV(alphas = [0.1,0.01,0.001, 0.0001],random_state=42,cv=10, max_iter=1000).fit(X_train,y_train)

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
    print('after Lasso: ', X_train.shape)
    X_test = X_test[:,supports]
    #输出看模型最终选择了几个特征向量，剔除了几个特征向量
    coef = pd.Series(model_lasso.coef_,X_names)
    print(coef)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    #输出所选择的最优正则化参数情况下的残差平均值，因为是10折，所以看平均值
    print(rmse_cv(model_lasso).mean())


    #画出特征变量的重要程度，这里面选出前3个重要，后3个不重要的举例
    # imp_coef = pd.concat([coef.sort_values().head(7),
    #                      coef.sort_values().tail(7)])
    # imp_coef = pd.concat([coef.sort_values()])
    # print(coef.sort_values())

    plt.rcParams['figure.figsize'] = (10.0, 5.0)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    coef.sort_values().plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show() 

    return X_train, y_train, X_test

X_train, y_train, X_test = lasso_fn(X_train, y_train, X_test)

# # 1.标准化处理
# scaler = StandardScaler()
# X = scaler.fit_transform(X)


# # 2.构建RF模型
# RFC_ = RandomForestClassifier()                               # 随机森林
# c = RFC_.fit(X, y).feature_importances_    # 特征重要性
# print("重要性:")
# print(c)

# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier()
# model.fit(X, y)
# # display the relative importance of each attribute
# print(model.feature_importances_)

# # 3. 交叉验证递归特征消除法
# from sklearn.feature_selection import RFECV
# selector = RFECV(RandomForestClassifier(n_estimators=8),step=1, cv=10)       # 采用交叉验证，每次排除一个特征，筛选出最优特征
# selector = selector.fit(X, y)
# X_wrapper = selector.transform(X)          # 最优特征
# score =cross_val_score(RandomForestClassifier(n_estimators=8) , X_wrapper, y, cv=10).mean()   # 最优特征分类结果
# print(score)
# print("最佳数量和排序")
# print(selector.support_)                                    # 选取结果
# print(selector.n_features_)                                 # 选取特征数量
# print(selector.ranking_)                                    # 依次排数特征排序
# X = X[:,selector.support_]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# # 4.递归特征消除法
# selector1 = RFE(RFC_, n_features_to_select=7, step=1).fit(X, y)       # n_features_to_select表示筛选最终特征数量，step表示每次排除一个特征
# selector1.support_.sum()
# print(selector1.ranking_)                                             # 特征排除排序
# print(selector1.n_features_)                                          # 选择特征数量
# X_wrapper1 = selector1.transform(X)                                   # 最优特征
# score =cross_val_score(RFC_, X_wrapper1, y, cv=10).mean()
# print(score)

# # 5.递归特征消除法和曲线图选取最优特征数量
# score = []                                                            # 建立列表
# for i in range(1, 8):
#     X_wrapper = RFE(RFC_, n_features_to_select=i, step=1).fit_transform(X, y)    # 最优特征
#     once = cross_val_score(RFC_, X_wrapper, y, cv=5).mean()                      # 交叉验证
#     score.append(once)                                                           # 交叉验证结果保存到列表
# print(max(score), (score.index(max(score))*1)+1)                                 # 输出最优分类结果和对应的特征数量
# print(score)
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 8, 1), score)
# plt.xticks(range(1, 8, 1))
# plt.show()
# # random_forest_classifier_support = [False,False,False,False,False,False,True,True,False,False,True,False,False,True]
# # X = X[:,selector.support_]
# # X = X[:,random_forest_classifier_support]


# EEC = EasyEnsembleClassifier(random_state=0)
# model = EEC.fit(X_train, y_train)
# predict = model.predict(X_test)
# predict_score = model.predict_proba(X_test)
# pred = predict_score[:,1]
# accuracy = metrics.accuracy_score(y_test, predict) 
# fpr, tpr, thersholds = roc_curve(y_test,pred)   
# roc_auc = auc(fpr, tpr)

# C2= confusion_matrix(y_test,predict, labels=[1, 0])
# TP = C2[0,0]
# FN = C2[0,1] 
# FP = C2[1,0]
# TN = C2[1,1]
# # Sensitivity, hit rate, recall, or true positive rate
# Se = TP/(TP+FN)
# # Specificity or true negative rate
# Sp = TN/(TN+FP) 
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# Acc = (TP+TN)/(TP+TN+FP+FN)
# #f,ax=plt.subplots()
# # # sns.heatmap(C2,annot=True,ax=ax,xticklabels='10',yticklabels='10') #画热力图
# # ax.set_title('confusion matrix') #标题
# # ax.set_xlabel('predict') #x轴
# # ax.set_ylabel('true') #y轴
# # plt.imshow(C2, cmap=plt.cm.Blues)

# # indices = range(len(C2))
# # plt.xticks(indices, ['1', '0'])
# # plt.yticks(indices, ['1', '0'])
# # plt.colorbar()
# # #色块添加数字
# # for first_index in range(2):    #第几行
# #     for second_index in range(2):    #第几列
# #         plt.text(second_index, first_index, C2[first_index][second_index],fontsize=15)
# # # plt.savefig('./confusion_matrix/'+q+'.png')
# # plt.show()
# print('Auc值:         %s'%roc_auc)
# print('Sensitivity:   %s'%Se) 
# print('Specificity:   %s'%Sp)
# print('PPV:           %s'%PPV)
# print('NPV:           %s'%NPV)
# print('Accuracy:      %s'%Acc)
#%%
'''
for classifier in test_classifiers: 
    print('******************* %s ********************' % classifier+'StratifiedKFold为%s折' % n_splits) 
    
    
    aucs = []
    Sensitivity = []
    Specificity = []
    PPVs = []
    NPVs = []
    accuracys = []
    for train_index, valid_index in skf.split(X_train, y_train):
        
        train_X, valid_X = X_train[train_index], X_train[valid_index]
        train_y, valid_y = y_train[train_index], y_train[valid_index]
       
        # from imblearn.combine import SMOTEENN,SMOTETomek
        # smote_enn = SMOTEENN(random_state=42)
        # train_X, train_y = smote_enn.fit_resample(train_X, train_y)
        
        
        #smote_enn = SMOTETomek(random_state=42)
        # print(train_X.shape)
        # print(train_y.shape)
        # print(valid_X.shape)
        # print(valid_y.shape)
        # print('训练集样本类别:')
        # print(sorted(Counter(y_train).items()))
       
        # print('SMOTE + ENN后训练集样本类别:')
        # print(sorted(Counter(y_train).items()))

    
        # print('******************* %s ********************' % classifier)
        # start_time = time.time()
        
        # print(model)
        # print('training took %fs!' % (time.time() - start_time))
        
        # predict = model.predict(X_test)
        # predict_score = model.predict_proba(X_test)
        # pred = predict_score[:,1]

        # #     if model_save_file != None: 
        # #         model_save[classifier] = model )
        # # precision = metrics.precision_score(y_test, predict,average='macro') 

        # # print(predict_score.shape, y_test.shape)
        # #recall = metrics.recall_score(y_test, predict,average='macro')
        # # print('precision: %.2f%%, recall: %.2f%%' % (100 * score, 100 * recall)) 
        # accuracy = metrics.accuracy_score(y_test, predict) 
        # f1 = metrics.f1_score(y_test, predict,average='macro')
        # # print('accuracy: %.2f%%' % (100 * accuracy))
        # # print('\nclassification_report:')
        # # print(classification_report( y_test,predict,labels=[0,1]))
        # fpr, tpr, thersholds = roc_curve(y_test,pred)   
        # roc_auc = auc(fpr, tpr)
        # C2= confusion_matrix( y_test,predict, labels=[1, 0])
        # #print(C2) #打印出来看看
        model = classifiers[classifier](train_X, train_y)
        predict = model.predict(valid_X)
        predict_score = model.predict_proba(valid_X)
        pred = predict_score[:,1]
        accuracy = metrics.accuracy_score(valid_y, predict) 
        fpr, tpr, thersholds = roc_curve(valid_y,pred)   
        roc_auc = auc(fpr, tpr)
        C2= confusion_matrix( valid_y,predict, labels=[1, 0])
        # predict = model.predict(train_X)
        # predict_score = model.predict_proba(train_X)
        # pred = predict_score[:,1]
        # accuracy = metrics.accuracy_score(train_y, predict) 
        # fpr, tpr, thersholds = roc_curve(train_y,pred)   
        # roc_auc = auc(fpr, tpr)
        # C2= confusion_matrix( train_y,predict, labels=[1, 0])

        #     if model_save_file != None: 
        #         model_save[classifier] = model )
        # precision = metrics.precision_score(y_test, predict,average='macro') 

        # print(predict_score.shape, y_test.shape)
        #recall = metrics.recall_score(y_test, predict,average='macro')
        # print('precision: %.2f%%, recall: %.2f%%' % (100 * score, 100 * recall)) 
       
        
        # print('accuracy: %.2f%%' % (100 * accuracy))
        # print('\nclassification_report:')
        # print(classification_report( y_test,predict,labels=[0,1]))
        
        TP = C2[0,0]
        FN = C2[0,1] 
        FP = C2[1,0]
        TN = C2[1,1]
        # Sensitivity, hit rate, recall, or true positive rate
        Se = TP/(TP+FN)
        # Specificity or true negative rate
        Sp = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        
        aucs.append(roc_auc)
        Sensitivity.append(Se)
        Specificity.append(Sp)
        PPVs.append(PPV)
        NPVs.append(NPV)
        
        accuracys.append(accuracy)
        #f,ax=plt.subplots()
        # # sns.heatmap(C2,annot=True,ax=ax,xticklabels='10',yticklabels='10') #画热力图
        # ax.set_title('confusion matrix') #标题
        # ax.set_xlabel('predict') #x轴
        # ax.set_ylabel('true') #y轴
        # plt.imshow(C2, cmap=plt.cm.Blues)
        
        # indices = range(len(C2))
        # plt.xticks(indices, ['1', '0'])
        # plt.yticks(indices, ['1', '0'])
        # plt.colorbar()
        # #色块添加数字
        # for first_index in range(2):    #第几行
        #     for second_index in range(2):    #第几列
        #         plt.text(second_index, first_index, C2[first_index][second_index],fontsize=15)
        # # plt.savefig('./confusion_matrix/'+q+'.png')
        # plt.show()
    print('Auc值:         %s'%np.mean(aucs))
    print('Sensitivity:   %s'%np.mean(Sensitivity)) 
    print('Specificity:   %s'%np.mean(Specificity))
    print('PPV:           %s'%np.mean(PPVs))
    print('NPV:           %s'%np.mean(NPVs))
    print('Accuracy:      %s'%np.mean(accuracys))
    
        # plt.figure(figsize=(10,10))
        # plt.plot(fpr, tpr, color='darkorange',
        #         lw=2, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()


'''

for classifier in test_classifiers: 
    print('******************* %s ************术前********' % classifier) 
    model = classifiers[classifier](X_train, y_train)
    predict = model.predict(X_test)
    predict_score = model.predict_proba(X_test)
    pred = predict_score[:,1]
    accuracy = metrics.accuracy_score(y_test, predict) 
    fpr, tpr, thersholds = roc_curve(y_test,pred)   
    roc_auc = auc(fpr, tpr)
    
    C2= confusion_matrix(y_test,predict, labels=[1, 0])
    TP = C2[0,0]
    FN = C2[0,1] 
    FP = C2[1,0]
    TN = C2[1,1]
    # Sensitivity, hit rate, recall, or true positive rate
    Se = TP/(TP+FN)
    # Specificity or true negative rate
    Sp = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    Acc = (TP+TN)/(TP+TN+FP+FN)
    #f,ax=plt.subplots()
    # # sns.heatmap(C2,annot=True,ax=ax,xticklabels='10',yticklabels='10') #画热力图
    # ax.set_title('confusion matrix') #标题
    # ax.set_xlabel('predict') #x轴
    # ax.set_ylabel('true') #y轴
    # plt.imshow(C2, cmap=plt.cm.Blues)
    
    # indices = range(len(C2))
    # plt.xticks(indices, ['1', '0'])
    # plt.yticks(indices, ['1', '0'])
    # plt.colorbar()
    # #色块添加数字
    # for first_index in range(2):    #第几行
    #     for second_index in range(2):    #第几列
    #         plt.text(second_index, first_index, C2[first_index][second_index],fontsize=15)
    # # plt.savefig('./confusion_matrix/'+q+'.png')
    # plt.show()
    print('Auc:         %s'%roc_auc)
    print('Sensitivity:   %s'%Se) 
    print('Specificity:   %s'%Sp)
    print('Precision:     %s'%PPV)
    print('NPV:           %s'%NPV)
    print('Accuracy:      %s'%Acc)