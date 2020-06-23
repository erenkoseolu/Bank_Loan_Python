from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


data=pd.read_csv(r"C:\\Users\\Lenovo\\Desktop\\Data_Mining\\train.csv")
data.drop(["Loan_ID"],axis=1, inplace=True)
#Looking at the missing values
print(data.info())

#Having insight about the data
print(data.describe().T)

#Identifying outliers
print(data.skew())

#Getting rid off applicantincome's outliers
outlier=data["ApplicantIncome"].quantile(0.9)
data["ApplicantIncome"]=np.where(data["ApplicantIncome"]>outlier, 
                                    outlier, 
                                    data["ApplicantIncome"])


#Getting rid off co applicantincome's outliers
outlier1=data["CoapplicantIncome"].quantile(0.9)

data["CoapplicantIncome"]=np.where(data["CoapplicantIncome"]>outlier1, 
                                    outlier1, 
                                    data["CoapplicantIncome"])
print(data)

#Deciding x and y

x=data.drop(["Loan_Status"],axis=1)
y=data["Loan_Status"]

#Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=41)

#Pipeline generation for numeric variables
numeric_pipeline= make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

#Pipeline generation for categorical variables
cat_pipeline= make_pipeline(SimpleImputer(strategy="most_frequent"), 
                            OneHotEncoder(handle_unknown='ignore'))

#Selecting the necessary columns

numeric_columns=data.select_dtypes(include=["int64","float64"]).columns
cat_columns=data.select_dtypes(include=["object"]).drop(["Loan_Status"], axis=1).columns

#Transform them

transform= ColumnTransformer(transformers= [("num", numeric_pipeline, numeric_columns),
                                            ("cat", cat_pipeline, cat_columns)],
                            remainder='passthrough')


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]

for classifier in classifiers:
    pipe = make_pipeline(transform, classifier)
    pipe.fit(x_train, y_train)   
    print(classifier)
    print("model score: %.3f" % pipe.score(x_test, y_test))







