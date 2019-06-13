import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# Import train data
train_df=pd.read_csv("data/train.csv")
train_df.head()

# Determine percentage of missing data
def missingdata(data):    
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)    
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])    
    ms= ms[ms["Percent"] > 0]    
    f,ax =plt.subplots(figsize=(8,6))    
    plt.xticks(rotation='90')    
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)    
    plt.xlabel('Features', fontsize=15)    
    plt.ylabel('Percent of missing values', fontsize=15)    
    plt.title('Percent missing data by feature', fontsize=15)    
    return ms

missingdata(train_df)


def cleandata(train_df, param1, param2, param_drop):
    # Fill empty fields
    # Find a way to automate the selection of these empty fields
    train_df[param1].fillna(train_df[param1].mode()[0], inplace = True)
    train_df[param2].fillna(train_df[param2].median(), inplace = True)
    drop_column = [param_drop]
    train_df.drop(drop_column, axis=1, inplace = True)
    print('check the nan value in data')
    print(train_df.isnull().sum())
    
    
    dataset = train_df
    # Create a new feature Familysize based on number if siblings and parch
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    # Define function to extract titles from passenger names
    import re
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    
    # Create a new feature Title, containing the titles of passenger names
    dataset['Title'] = dataset['Name'].apply(get_title)
    
    # Group all non-common titles into one single grouping "Rare"
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Group age and fare features into bins
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,14,20,40,120], labels=['Children','Teenage','Adult','Elder'])
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare', 'Average_fare','high_fare'])
    
    # Drop columns
    traindf=train_df
    drop_column = ['Age','Fare','Name','Ticket']
    dataset.drop(drop_column, axis=1, inplace = True)
    
    drop_column = ['PassengerId']
    traindf.drop(drop_column, axis=1, inplace = True)
    traindf = pd.get_dummies(traindf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],
                                 prefix=["Sex","Title","Age_type","Em_type","Fare_type"])
    return traindf

traindf = cleandata(train_df, 'Embarked', 'Age', 'Cabin')

# Plot heat to illustrate correlation and identify unessential features
sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
#data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()


# Import sklearn and split data into train and test set
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
all_features = traindf.drop("Survived",axis=1)
Targeted_feature = traindf["Survived"]
# X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=1,random_state=42)
# X_train.shape,X_test.shape,y_train.shape,y_test.shape


# Fit and test data
from sklearn.ensemble import RandomForestClassifier

#model = RandomForestClassifier(criterion='gini', n_estimators=700,
#                             min_samples_split=10,min_samples_leaf=1,
#                             max_features='auto',oob_score=True,
#                             random_state=1,n_jobs=-1)

model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=800,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

model.fit(all_features, Targeted_feature)
#prediction_rm=model.predict(X_test)
#print('--------------The Accuracy of the model----------------------------')
#print('The accuracy of the Random Forest Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
#kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
#result_rm=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')
#print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
#y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)
#sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")
#plt.title('Confusion_matrix', y=1.05, size=15)




# Import data
test_df=pd.read_csv("data/test.csv")
test_df.head()

missingdata(test_df)

testdf = cleandata(test_df, 'Fare', 'Age', 'Cabin')


# Plot heat to illustrate correlation and identify unessential features
sns.heatmap(testdf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
#data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()

prediction_rm=model.predict(testdf)
np.savetxt("submission.csv", prediction_rm, delimiter=",")

# Optimizing the model using GridSearch and the RandomForest Classifier
from sklearn.model_selection import GridSearchCV
# Random Forest Classifier Parameters tunning 
model = RandomForestClassifier()
n_estim=range(100,1000,100)

## Search grid for optimal parameters
param_grid = {"n_estimators" :n_estim}


model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

model_rf.fit(all_features, Targeted_feature)

# Best score
print(model_rf.best_score_)

#best estimator
model_rf.best_estimator_

#model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                       max_depth=None, max_features='auto', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=800,
#                       n_jobs=None, oob_score=False, random_state=None,
#                       verbose=0, warm_start=False)
#
#model.fit(all_features, Targeted_feature)