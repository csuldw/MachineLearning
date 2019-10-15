from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
from functools import reduce
from sklearn.metrics import confusion_matrix, classification_report
 
class StackingClassifier(object):
    
    def __init__(self, modellist=[], meta_classifier=None):
        self.modellist = modellist
        if meta_classifier == None:
            from sklearn.linear_model import LogisticRegression
            meta_classifier = LogisticRegression()
        self.meta_classifier = meta_classifier

    def SelectModel(self, modelname):
    
        if modelname == "SVM":
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', C=16, gamma=0.125,probability=True)
        
        elif modelname == "lr":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()

        elif modelname == "GBDT":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier()
    
        elif modelname == "RF":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
    
        elif modelname == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                    learning_rate=0.01,
                    n_estimators=1000,
                    max_depth=4,
                    min_child_weight=3,
                    gamma=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1,
                    objective='binary:logistic', #multi:softmax
                    nthread=8,
                    scale_pos_weight=1,
                    seed=27,
                    random_state=27
                )    
        elif modelname == "KNN":
            from sklearn.neighbors import KNeighborsClassifier as knn
            model = knn()
        
        elif modelname == "MNB":
            from sklearn.naive_bayes import MultinomialNB
            model = MultinomialNB()
        else:
            pass
        return model
 
    def get_oof(self, clf, n_folds, X_train, y_train, X_test):
        ntrain = X_train.shape[0]
        ntest =  X_test.shape[0]
        print("kfolds: ", ntrain, ntest)
        classnum = len(np.unique(y_train))
        kf = KFold(n_splits=n_folds,random_state=1)
        oof_train = np.zeros((ntrain,classnum))
        oof_test = np.zeros((ntest,classnum))
    
        for i,(train_index, test_index) in enumerate(kf.split(X_train)):
            kf_X_train = X_train[train_index] # 数据
            kf_y_train = y_train[train_index] # 标签
    
            kf_X_test = X_train[test_index]  # k-fold的验证集
    
            clf.fit(kf_X_train, kf_y_train)
            oof_train[test_index] = clf.predict_proba(kf_X_test)
            # print("shape of oof_train:", oof_train[test_index].shape)
    
            print("fold{i}: oof_train: {a}, oof_test:{b}".format(i=i, a=oof_train.shape, b=oof_test.shape))
            oof_test += clf.predict_proba(X_test)
        oof_test = oof_test/float(n_folds)
        print("oof_train: {a}, oof_test:{b}".format(a=oof_train.shape, b=oof_test.shape))
        return oof_train, oof_test

    def first_layer(self, X_train, y_train, X_test, modellist=None):
        """modellist 需要重新修改
        """
        newfeature_list = []
        newtestdata_list = []
        for modelname in self.modellist:
            sub_clf = self.SelectModel(modelname)
            oof_train_, oof_test_= self.get_oof(clf=sub_clf,
                                                n_folds=5,
                                                X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test)
            print("oof_train: ", oof_train_.shape)
            print("model-{}".format(modelname),len(oof_train_), len(oof_test_))
            newfeature_list.append(oof_train_)
            print("newfeature_list: ", len(newfeature_list))
            newtestdata_list.append(oof_test_)
        
        # 特征组合
        X_train_stacking = reduce(lambda x,y:np.concatenate((x,y),axis=1),newfeature_list)    
        X_test_stacking = reduce(lambda x,y:np.concatenate((x,y),axis=1),newtestdata_list)

        return X_train_stacking, X_test_stacking

    def fit(self, X_train, y_train, clf=None):
        if clf != None:
            self.meta_classifier = clf
        self.meta_classifier.fit(X_train, y_train)
        return self.meta_classifier    

    #second_layer
    def second_layer(self, X_train, y_train, clf=None):
        return self.fit(X_train, y_train, clf)
    
    def predict(self, X_test, clf=None, type="label"):
        if clf == None:
            clf = self.meta_classifier
        if type == "proba":
            return clf.predict_proba(X_test)
        elif type == "label":
            return clf.predict(X_test)
    
    def get_accuracy(self, y_true, y_pred):
        accuracy = metrics.accuracy_score(y_true, y_pred)*100
        return accuracy

    def performance(self, y_true, y_pred):
        accuracy = self.get_accuracy(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        print("多模型融合预测accuracy：{}".format(accuracy))
        print("混淆矩阵：\n{}".format(confusion))
        print("预测结果：\n{}".format(report))
        return confusion, report
        
 
# 使用stacking方法的时候
# 第一级，重构特征当做第二级的训练集
if __name__ == "__main__":
    # 导入数据集切割训练与测试数据
    data = load_digits()
    data_D = preprocessing.StandardScaler().fit_transform(data.data)
    data_L = data.target
    X_train, X_test, y_train, y_test = train_test_split(data_D,data_L,random_state=100,test_size=0.7)
    print(set(y_train))

    # 单纯使用一个分类器的时候
    clf_meta = RandomForestClassifier()
    clf_meta.fit(X_train, y_train)
    pred = clf_meta.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred)*100
    print ("====================", accuracy)
    # 91.0969793323

    #layer 1：多模型融合
    modelist = ['SVM', 'GBDT', 'RF', 'KNN']
    stacking_clf = StackingClassifier(modelist)
    X_train_stacking, X_test_stacking = stacking_clf.first_layer(X_train, y_train, X_test)
    print("shape of X_train_stacking {}".format(X_train_stacking.shape))
    print("shape of X_test_stacking {}".format(X_test_stacking.shape))
    
    #layer 2： 单模型训练
    RF = stacking_clf.SelectModel(modelname="RF")
    clf = stacking_clf.second_layer(X_train_stacking, y_train, clf=RF)
    pred = stacking_clf.predict(X_test_stacking)

    #模型评估
    stacking_clf.performance(y_test, pred)
    # 96.4228934817
