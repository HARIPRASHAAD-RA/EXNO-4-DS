# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```

![Screenshot 2024-11-11 081009](https://github.com/user-attachments/assets/4a469fe4-29a2-4a60-8929-8886fb97152e)

```
data.isnull().sum()
```

![Screenshot 2024-11-11 081054](https://github.com/user-attachments/assets/0922150b-6364-4f8d-aa70-df35579926aa)

```
missing=data[data.isnull().any(axis=1)]
missing
```

![Screenshot 2024-11-11 081137](https://github.com/user-attachments/assets/2d17b74f-700c-4564-979c-970dbff19512)

```
data2=data.dropna(axis=0)
data2
```

![Screenshot 2024-11-11 081229](https://github.com/user-attachments/assets/4196a5d6-9aa0-4f6b-af5b-0b375e4ee505)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![Screenshot 2024-11-11 081323](https://github.com/user-attachments/assets/7d13698d-cd71-4214-9bac-6f2caff34f10)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![Screenshot 2024-11-11 081409](https://github.com/user-attachments/assets/f711772c-3069-44f2-bd93-cac72461a428)

```
data2
```

![Screenshot 2024-11-11 081451](https://github.com/user-attachments/assets/444745ac-bca3-48d0-8ebf-087e0c21fcdd)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![Screenshot 2024-11-11 081558](https://github.com/user-attachments/assets/a8641575-6073-4e25-9b2e-3e260235403a)

```
columns_list=list(new_data.columns)
print(columns_list)
```

![Screenshot 2024-11-11 081649](https://github.com/user-attachments/assets/f0b0e8e6-8b12-4609-94e6-42e5c18f90bc)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![Screenshot 2024-11-11 081750](https://github.com/user-attachments/assets/48b68965-04a6-43bd-9f23-89c30832dd53)

```
y=new_data['SalStat'].values
print(y)
```

![Screenshot 2024-11-11 081824](https://github.com/user-attachments/assets/09fce8f9-8443-4094-8099-46a542d7bd57)

```
x=new_data[features].values
print(x)
```

![Screenshot 2024-11-11 081856](https://github.com/user-attachments/assets/89c081f5-d198-4822-a0a8-cf4f2a1f6d03)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

![Screenshot 2024-11-11 081927](https://github.com/user-attachments/assets/559875d4-79f5-4a31-9efe-e91fd645ce35)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![Screenshot 2024-11-11 082006](https://github.com/user-attachments/assets/99681fb7-8be7-499b-a63f-82d2980c09e1)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![Screenshot 2024-11-11 082045](https://github.com/user-attachments/assets/f16f68b4-1029-4636-98f1-68c1b13a94db)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![Screenshot 2024-11-11 082128](https://github.com/user-attachments/assets/108dcf93-549e-4a76-9b11-7d85032040ef)

```
data.shape
```

![Screenshot 2024-11-11 082200](https://github.com/user-attachments/assets/961574df-1b00-4e90-ab3c-d7b8a67aa491)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![Screenshot 2024-11-11 082228](https://github.com/user-attachments/assets/3ac97016-0e3c-4642-87ff-8ab11a461248)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![Screenshot 2024-11-11 082257](https://github.com/user-attachments/assets/443f3ac9-2253-4046-a1d8-7a5572846549)

```
tips.time.unique()
```

![Screenshot 2024-11-11 082329](https://github.com/user-attachments/assets/cfaf0cfe-f47c-451b-8125-37d0de7ea4ba)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![Screenshot 2024-11-11 082409](https://github.com/user-attachments/assets/675b1fc8-095c-4c56-b773-b3bc67494821)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![Screenshot 2024-11-11 082415](https://github.com/user-attachments/assets/65d3d6ee-42d9-4ba2-89df-694635f6a928)

# RESULT:
Thus, feature selection and feature scaling has been used on the given dataset.
