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
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import xgboost as xgb

class SubClassifier(object):
    def __init__(self):
        # import lightgbm as lgb
        # import xgboost as xgb
        # from sklearn.svm import SVC
        # from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.svm import LinearSVC
        # clfs = {
        #     'lr': LogisticRegression(penalty='l1', C=0.1, tol=0.0001),
        #     'svm': LinearSVC(C=0.05, penalty='l2', dual=True),
        #     'svm_linear': SVC(kernel='linear', probability=True),
        #     'svm_ploy': SVC(kernel='poly', probability=True),
        #     'bagging': BaggingClassifier(base_estimator=base_clf, n_estimators=60, max_samples=1.0, max_features=1.0,
        #                                 random_state=1, n_jobs=1, verbose=1),
        #     'rf': RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=9),
        #     'adaboost': AdaBoostClassifier(base_estimator=base_clf, n_estimators=50, algorithm='SAMME'),
        #     'gbdt': GradientBoostingClassifier(),
        #     'xgb': xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=50),
        #     'lgb': lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.01, max_depth=5, n_estimators=250, num_leaves=90)
        # }
        pass

    def SelectModel(self, modelname):
        if modelname == "SVM":
            from sklearn.svm import SVC
            clf = SVC(kernel='rbf', C=16, gamma=0.125,probability=True)
        
        elif modelname == "lr":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()

        elif modelname == "GBDT":
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier()
    
        elif modelname == "RF":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100)
    
        elif modelname == "xgboost":
            from xgboost import XGBClassifier
            clf = XGBClassifier(
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
            clf = knn()
        
        elif modelname == "MNB":
            from sklearn.naive_bayes import MultinomialNB
            clf = MultinomialNB()
        else:
            pass
        return clf
    
    def performance(self, y_true, y_pred, modelname=""):
        accuracy = metrics.accuracy_score(y_true, y_pred)*100
        confusion = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        print("模型{}预测accuracy：{}".format(modelname, accuracy))
        print("混淆矩阵：\n{}".format(confusion))
        print("预测结果：\n{}".format(report))
        return confusion, report


class StackingClassifier(object):
    
    def __init__(self, classifiers, meta_classifier,
                use_clones=True, n_folds=2,
                n_classes=2, random_state=100,
                sample_weight=None, use_probas=True):

        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_clones=use_clones
        self.n_folds = n_folds
        self.n_classes = n_classes
        self.random_state = random_state
        self.sample_weight = sample_weight
        self.use_probas = use_probas

    def cross_valid_oof(self, clf, X, y, n_folds):
        """返回CV预测结果
        """
        ntrain = X.shape[0]
        n_classes = self.n_classes
        random_state = self.random_state
        oof_features = np.zeros((ntrain, n_classes))
        oof_pred = np.zeros(ntrain)
        kf = KFold(n_splits=n_folds, random_state=random_state)
        for i,(train_index, test_index) in enumerate(kf.split(X)):
            kf_X_train = X[train_index] # 数据
            kf_y_train = y[train_index] # 标签
    
            kf_X_test = X[test_index]  # k-fold的验证集
    
            clf.fit(kf_X_train, kf_y_train)
            if not self.use_probas:
                oof_features[test_index] = clf.predict(kf_X_test)
            else:
                oof_features[test_index] = clf.predict_proba(kf_X_test)
            oof_pred[test_index] = clf.predict(kf_X_test)
            print("fold-{i}: oof_features: {a}, cv-oof accuracy:{c}".format(i=i, 
                                            a=oof_features.shape,
                                            c=self.get_accuracy(y[test_index], oof_pred[test_index])))
        return oof_features

    def fit(self, X, y):
        self.clfs_ = self.classifiers
        self.meta_clf_ = self.meta_classifier
            
        n_folds = self.n_folds
        sample_weight = self.sample_weight
        meta_features = None

        #feature layer
        for name, sub_clf in self.clfs_.items():
            print("feature layer, current model: {}".format(name))
            meta_prediction = self.cross_valid_oof(sub_clf, X, y, n_folds)
            if meta_features is None:
                meta_features = meta_prediction
            else:
                meta_features = np.column_stack((meta_features, meta_prediction))

        for name, model in self.clfs_.items():
            print("fit base model using all train set: {}".format(name))
            if sample_weight is None:
                model.fit(X, y)
            else:
                model.fit(X, y, sample_weight=sample_weight)

        #meta layer
        if sample_weight is None:
            self.meta_clf_.fit(meta_features, y)
        else:
            self.meta_clf_.fit(meta_features, y, sample_weight=sample_weight)

        return self

    def predict_meta_features(self, X):
        """ Get meta-features of test-data.
        Parameters
        -------
        X : numpy array, shape = [n_samples, n_features]

        Returns:
        -------
        meta-features : numpy array, shape = [n_samples, n_classifiers]
        """
        per_model_preds = []

        for name, model in self.clfs_.items():
            print("model {} predict_meta_features".format(name))
            if not self.use_probas:
                pred_score = model.predict(X)
            else:
                pred_score = model.predict_proba(X)

            per_model_preds.append(pred_score)

        return np.hstack(per_model_preds)


    def predict(self, X):
        """ Predict class label for X."""
        meta_features = self.predict_meta_features(X)
        return self.meta_clf_.predict(meta_features)

    def predict_prob(self, X):
        """ Predict class probabilities for X."""
        meta_features = self.predict_meta_features(X)
        return self.meta_clf_.predict_proba(meta_features)
    
    def get_accuracy(self, y_true, y_pred):
        accuracy = round(metrics.accuracy_score(y_true, y_pred)*100,3)
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
if __name__ == "__main__":
    # 导入数据集切割训练与测试数据
    data = load_digits()
    data_D = preprocessing.StandardScaler().fit_transform(data.data)
    data_L = data.target
    X_train, X_test, y_train, y_test = train_test_split(data_D,data_L,random_state=100,test_size=0.7)
    print(set(y_train))

    #layer 1：多模型融合
    classifiers = {
            'KNN': SubClassifier().SelectModel(modelname="KNN"),
            'rf': SubClassifier().SelectModel(modelname="RF"),
            'svm':  SubClassifier().SelectModel(modelname="SVM"),
            'GBDT':  SubClassifier().SelectModel(modelname="GBDT")
        }
    
    meta_classifier = SubClassifier().SelectModel(modelname="RF")

    stacking_clf = StackingClassifier(classifiers, meta_classifier, n_classes=10,n_folds=5)

    stacking_clf.fit(X_train, y_train)
    pred = stacking_clf.predict(X_test)

    #模型评估
    stacking_clf.performance(y_test, pred)
    # 96.4228934817