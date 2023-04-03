import os
from tqdm import tqdm 
from Proccess import search_files
import json
import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neighbors import KNeighborsClassifier  
import pickle
import xgboost
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import cv

from collections import Counter
# ==== Initialise Paths ====

path =  'working directory/input_dir'#ENTER PATH TO DATA DIR'  # data dir
lang_json = 'working directory/languages.json'#'ENTER PATH TO LANG JSON'  # languages json path
model_dir =  'working directory/models'# model output dir

with open(lang_json) as f:
    languages = json.load(f)

n_classes = len(languages)

extensions = [ext for exts in languages.values() for ext in exts]
files = search_files(path, extensions)

print(len(files))
print(files[0])
print(len(extensions),extensions[0])
print(list(languages.keys()))
# retur
st = time.time()

# X = extract_from_files(files[:],languages)
# pickle.dump(X,open('input.pkl','wb'))

X = pickle.load(open('input.pkl','rb'))
print("extraction time : ", time.time()-st)

# x_train, x_test, y_train, y_test= train_test_split(X[0], X[1], test_size= 0.25, random_state=0) 

fold_no = 1
loss_per_fold = []
acc_per_fold = [0 for i in range(len(languages))]


kfold = KFold(n_splits=5, shuffle=True)
print(kfold)

for train , test in kfold.split(X[0],X[1]):
    x_train = X[0][train]
    y_train = X[1][train]

    x_test = X[0][test]
    y_test = X[1][test]

    smote = SMOTE()
    try:
        x_train , y_train = smote.fit_resample(x_train,y_train)
    except Exception as e:
        print('error in oversampling')
        pass
    
    
    # classifier= KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2 ) 
    # classifier = xgboost.XGBClassifier(max_depth=7,n_estimators=30)

    classifier = RandomForestClassifier( max_depth=35, random_state=10,n_estimators=20,criterion='gini')

    st = time.time()
    classifier.fit(x_train,y_train)
    pickle.dump(classifier,open('models/RF.pkl','wb'))
    print("time taken : ",time.time()- st)
    y_pred= classifier.predict(x_test)  

   
    y_pred = [list(languages.keys())[i] for i in y_pred]
    y_test = [list(languages.keys())[i] for i in y_test]

    cm = classification_report(y_test,y_pred)
    precision,recall,f1score,support=score(y_test,y_pred,average=None)

    acc_per_fold += f1score
    print('=====================================fold no {}================================='.format(fold_no))
    print(acc_per_fold)
    print(f1score)
    print(cm)

    fold_no = fold_no + 1


print(acc_per_fold/(fold_no-1))
print(classification_report(y_test,y_pred))








# smote = SMOTE()

# x_train ,y_train = smote.fit_resample(x_train,y_train)

# X, y = make_classification(n_samples=len(X[0]), n_features=1024, n_informative=15, n_redundant=5, random_state=1)
# # prepare the cross-validation procedure
# cv = KFold(n_splits=5, random_state=1, shuffle=True)

# classifier = xgboost.XGBClassifier(max_depth=7, n_estimators=30)

# # scores = cross_val_score(classifier, X, y, scoring='f1', cv=cv, n_jobs=-1)


# print('counter :', Counter(X[1]))
# print('counter :', Counter(y_train))
# print('test counter :', Counter(y_test))
# ///////////////////////data balancing
# from sklearn.utils.class_weight import compute_sample_weight
# sample_weights = compute_sample_weight(
#     class_weight='balanced',
#     y=y_train['class'] #provide your own target name
# )

# //////////////////////////////////////////////////


# classifier = RandomForestClassifier(max_depth=35, random_state=10,n_estimators=20,criterion='gini',class_weight = 'balanced_subsample')
# st = time.time()
# classifier = xgboost.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)

# classifier = xgboost.XGBClassifier(max_depth=7, n_estimators=30)
# classifier.fit(x_train, y_train, sample_weight=sample_weights)


# # classifier.fit(x_train, y_train)  

# print('training_time : ',time.time()-st)


# st = time.time()
# y_pred= classifier.predict(x_test)  

# y_pred = [list(languages.keys())[i] for i in y_pred]
# y_test = [list(languages.keys())[i] for i in y_test]


# print('pred_time : ',time.time()-st)


# cm= confusion_matrix(y_test, y_pred) 


# print(cm)

# print(classification_report(y_test,y_pred))

# print(pd.crosstab(y_test,y_pred))