
# 1. Credit card applications

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do!

Credit card being held in hand

We'll use the Credit Card Approval dataset from the UCI Machine Learning Repository. The structure of this notebook is as follows:

First, we will start off by loading and viewing the dataset.
We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.
First, loading and viewing the dataset. We find that since this data is confidential, the contributor of the dataset has anonymized the feature names.




```python
# Import pandas
import pandas as pd

# Load dataset
cc_apps = pd.read_csv("credit-approval.csv", header=None)

# Inspect data
print(cc_apps.head())
```

       0      1     2   3   4   5   6     7   8    9    10   11   12   13   14  \
    0  A1     A2    A3  A4  A5  A6  A7    A8  A9  A10  A11  A12  A13  A14  A15   
    1   b  30.83     0   u   g   w   v  1.25   t    t    1    f    g  202    0   
    2   a  58.67  4.46   u   g   q   h  3.04   t    t    6    f    g   43  560   
    3   a  24.50   0.5   u   g   q   h   1.5   t    f    0    f    g  280  824   
    4   b  27.83  1.54   u   g   w   v  3.75   t    t    5    t    g  100    3   
    
          15  
    0  class  
    1      +  
    2      +  
    3      +  
    4      +  
    


# 2. Inspecting the applications
The output may appear a bit confusing at its first sight, but let's try to figure out the most important features of a credit card application. The features of this dataset have been anonymized to protect the privacy, but this blog gives us a pretty good overview of the probable features. The probable features in a typical credit card application are Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus. This gives us a pretty good starting point, and we can map these features with respect to the columns in the output.

As we can see from our first glance at the data, the dataset has a mixture of numerical and non-numerical features. This can be fixed with some preprocessing, but before we do that, let's learn a bit more about the dataset a bit more to see if there are other dataset issues that need to be fixed.



```python
# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")

# Inspect missing values in the dataset
cc_apps.tail(17)
```

             0      1    2    3    4    5    6    7    8    9    10   11   12  \
    count   679    679  691  685  685  682  682  691  691  691  691  691  691   
    unique    3    350  216    4    4   15   10  133    3    3   24    3    4   
    top       b  22.67  1.5    u    g    c    v    0    t    f    0    f    g   
    freq    468      9   21  519  519  137  399   70  361  395  395  374  625   
    
             13   14   15  
    count   678  691  691  
    unique  171  241    3  
    top       0    0    -  
    freq    132  295  383  
    
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 691 entries, 0 to 690
    Data columns (total 16 columns):
    0     679 non-null object
    1     679 non-null object
    2     691 non-null object
    3     685 non-null object
    4     685 non-null object
    5     682 non-null object
    6     682 non-null object
    7     691 non-null object
    8     691 non-null object
    9     691 non-null object
    10    691 non-null object
    11    691 non-null object
    12    691 non-null object
    13    678 non-null object
    14    691 non-null object
    15    691 non-null object
    dtypes: object(16)
    memory usage: 86.5+ KB
    None
    
    
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>674</th>
      <td>NaN</td>
      <td>29.50</td>
      <td>2</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>2</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>256</td>
      <td>17</td>
      <td>-</td>
    </tr>
    <tr>
      <th>675</th>
      <td>a</td>
      <td>37.33</td>
      <td>2.5</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>h</td>
      <td>0.21</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>260</td>
      <td>246</td>
      <td>-</td>
    </tr>
    <tr>
      <th>676</th>
      <td>a</td>
      <td>41.58</td>
      <td>1.04</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.665</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>240</td>
      <td>237</td>
      <td>-</td>
    </tr>
    <tr>
      <th>677</th>
      <td>a</td>
      <td>30.58</td>
      <td>10.665</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>0.085</td>
      <td>f</td>
      <td>t</td>
      <td>12</td>
      <td>t</td>
      <td>g</td>
      <td>129</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>678</th>
      <td>b</td>
      <td>19.42</td>
      <td>7.25</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>0.04</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>100</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>679</th>
      <td>a</td>
      <td>17.92</td>
      <td>10.21</td>
      <td>u</td>
      <td>g</td>
      <td>ff</td>
      <td>ff</td>
      <td>0</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>0</td>
      <td>50</td>
      <td>-</td>
    </tr>
    <tr>
      <th>680</th>
      <td>a</td>
      <td>20.08</td>
      <td>1.25</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>0</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>681</th>
      <td>b</td>
      <td>19.50</td>
      <td>0.29</td>
      <td>u</td>
      <td>g</td>
      <td>k</td>
      <td>v</td>
      <td>0.29</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280</td>
      <td>364</td>
      <td>-</td>
    </tr>
    <tr>
      <th>682</th>
      <td>b</td>
      <td>27.83</td>
      <td>1</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>h</td>
      <td>3</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>176</td>
      <td>537</td>
      <td>-</td>
    </tr>
    <tr>
      <th>683</th>
      <td>b</td>
      <td>17.08</td>
      <td>3.29</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>v</td>
      <td>0.335</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>140</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <th>684</th>
      <td>b</td>
      <td>36.42</td>
      <td>0.75</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>v</td>
      <td>0.585</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>240</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>685</th>
      <td>b</td>
      <td>40.58</td>
      <td>3.29</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>3.5</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>s</td>
      <td>400</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>686</th>
      <td>b</td>
      <td>21.08</td>
      <td>10.085</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>1.25</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>260</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>687</th>
      <td>a</td>
      <td>22.67</td>
      <td>0.75</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>2</td>
      <td>f</td>
      <td>t</td>
      <td>2</td>
      <td>t</td>
      <td>g</td>
      <td>200</td>
      <td>394</td>
      <td>-</td>
    </tr>
    <tr>
      <th>688</th>
      <td>a</td>
      <td>25.25</td>
      <td>13.5</td>
      <td>y</td>
      <td>p</td>
      <td>ff</td>
      <td>ff</td>
      <td>2</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>t</td>
      <td>g</td>
      <td>200</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>689</th>
      <td>b</td>
      <td>17.92</td>
      <td>0.205</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.04</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280</td>
      <td>750</td>
      <td>-</td>
    </tr>
    <tr>
      <th>690</th>
      <td>b</td>
      <td>35.00</td>
      <td>3.375</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>8.29</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Handling the missing values (part i)

We've uncovered some issues that will affect the performance of our machine learning model(s) if they go unchanged:

Our dataset contains both numeric and non-numeric data (specifically data that are of float64, int64 and object types). Specifically, the features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values.
The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. Apart from these, we can get useful statistical information (like mean, max, and min) about the features that have numerical values.
Finally, the dataset has missing values, which we'll take care of in this task. The missing values in the dataset are labeled with '?', which can be seen in the last cell's output.
Now, let's temporarily replace these missing value question marks with NaN.



```python
# Import numpy
import numpy as np

# Inspect missing values in the dataset
print(cc_apps.isnull().values.sum())

# Replace the '?'s with NaN
cc_apps = cc_apps.replace("?",np.NaN)

# Inspect the missing values again
cc_apps.tail(17)
```

    67
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>674</th>
      <td>NaN</td>
      <td>29.50</td>
      <td>2</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>2</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>256</td>
      <td>17</td>
      <td>-</td>
    </tr>
    <tr>
      <th>675</th>
      <td>a</td>
      <td>37.33</td>
      <td>2.5</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>h</td>
      <td>0.21</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>260</td>
      <td>246</td>
      <td>-</td>
    </tr>
    <tr>
      <th>676</th>
      <td>a</td>
      <td>41.58</td>
      <td>1.04</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.665</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>240</td>
      <td>237</td>
      <td>-</td>
    </tr>
    <tr>
      <th>677</th>
      <td>a</td>
      <td>30.58</td>
      <td>10.665</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>0.085</td>
      <td>f</td>
      <td>t</td>
      <td>12</td>
      <td>t</td>
      <td>g</td>
      <td>129</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>678</th>
      <td>b</td>
      <td>19.42</td>
      <td>7.25</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>0.04</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>100</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>679</th>
      <td>a</td>
      <td>17.92</td>
      <td>10.21</td>
      <td>u</td>
      <td>g</td>
      <td>ff</td>
      <td>ff</td>
      <td>0</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>0</td>
      <td>50</td>
      <td>-</td>
    </tr>
    <tr>
      <th>680</th>
      <td>a</td>
      <td>20.08</td>
      <td>1.25</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>0</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>681</th>
      <td>b</td>
      <td>19.50</td>
      <td>0.29</td>
      <td>u</td>
      <td>g</td>
      <td>k</td>
      <td>v</td>
      <td>0.29</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280</td>
      <td>364</td>
      <td>-</td>
    </tr>
    <tr>
      <th>682</th>
      <td>b</td>
      <td>27.83</td>
      <td>1</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>h</td>
      <td>3</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>176</td>
      <td>537</td>
      <td>-</td>
    </tr>
    <tr>
      <th>683</th>
      <td>b</td>
      <td>17.08</td>
      <td>3.29</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>v</td>
      <td>0.335</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>140</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <th>684</th>
      <td>b</td>
      <td>36.42</td>
      <td>0.75</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>v</td>
      <td>0.585</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>240</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>685</th>
      <td>b</td>
      <td>40.58</td>
      <td>3.29</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>3.5</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>s</td>
      <td>400</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>686</th>
      <td>b</td>
      <td>21.08</td>
      <td>10.085</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>1.25</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>260</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>687</th>
      <td>a</td>
      <td>22.67</td>
      <td>0.75</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>2</td>
      <td>f</td>
      <td>t</td>
      <td>2</td>
      <td>t</td>
      <td>g</td>
      <td>200</td>
      <td>394</td>
      <td>-</td>
    </tr>
    <tr>
      <th>688</th>
      <td>a</td>
      <td>25.25</td>
      <td>13.5</td>
      <td>y</td>
      <td>p</td>
      <td>ff</td>
      <td>ff</td>
      <td>2</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>t</td>
      <td>g</td>
      <td>200</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>689</th>
      <td>b</td>
      <td>17.92</td>
      <td>0.205</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.04</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>280</td>
      <td>750</td>
      <td>-</td>
    </tr>
    <tr>
      <th>690</th>
      <td>b</td>
      <td>35.00</td>
      <td>3.375</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>8.29</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>0</td>
      <td>0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Handling the missing values (part ii)

We replaced all the question marks with NaNs. This is going to help us in the next missing value treatment that we are going to perform.

An important question that gets raised here is why are we giving so much importance to missing values? Can't they be just ignored? Ignoring missing values can affect the performance of a machine learning model heavily. While ignoring the missing values our machine learning model may miss out on information about the dataset that may be useful for its training. Then, there are many models which cannot handle missing values implicitly such as LDA.

So, to avoid this problem, we are going to impute the missing values with a strategy called mean imputation.



```python

# Impute the missing values with mean imputation
cc_apps = cc_apps.fillna(cc_apps.mean())

# Count the number of NaNs in the dataset to verify
print(cc_apps.isnull().values.sum())
```

    67
    

# 5. Handling the missing values (part iii)
We have successfully taken care of the missing values present in the numeric columns. There are still some missing values to be imputed for columns 0, 1, 3, 4, 5, 6 and 13. All of these columns contain non-numeric data and this why the mean imputation strategy would not work here. This needs a different treatment.

We are going to impute these missing values with the most frequent values as present in the respective columns. This is good practice when it comes to imputing missing values for categorical data in general.


```python
# Iterate over each column of cc_apps
print(cc_apps.info())
for col in cc_apps.columns:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps[col] = cc_apps[col].fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(cc_apps.isnull().values.sum())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 691 entries, 0 to 690
    Data columns (total 16 columns):
    0     679 non-null object
    1     679 non-null object
    2     691 non-null object
    3     685 non-null object
    4     685 non-null object
    5     682 non-null object
    6     682 non-null object
    7     691 non-null object
    8     691 non-null object
    9     691 non-null object
    10    691 non-null object
    11    691 non-null object
    12    691 non-null object
    13    678 non-null object
    14    691 non-null object
    15    691 non-null object
    dtypes: object(16)
    memory usage: 86.5+ KB
    None
    0
    

# 6. Preprocessing the data (part i)
The missing values are now successfully handled.

There is still some minor but essential data preprocessing needed before we proceed towards building our machine learning model. We are going to divide these remaining preprocessing steps into two main tasks:

Convert the non-numeric data into numeric.
Scale the feature values to a uniform range.
First, we will be converting all the non-numeric values into numeric ones. We do this because not only it results in a faster computation but also many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn) require the data to be in a strictly numeric format. We will do this by using a technique called label encoding.



```python
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col]=le.fit_transform(cc_apps[col])
```

# 7. Preprocessing the data (part ii)
We have successfully converted all the non-numeric values to numeric ones.

Now, let's try to understand what these scaled values mean in the real world. Let's use CreditScore as an example. The creidt score of a person is their creditworthiness based on their credit history. The higher this number, the more financially trustworthy a person is considered to be. So, a CreditScore of 1 is the highest since we're rescaling all the values to the range of 0-1.

Also, features like DriversLicense and ZipCode are not as important as the other features in the dataset for predicting credit card approvals. We should drop them to design our machine learning model with the best set of features. This is often called feature engineering or, more specifically, feature selection.



```python

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Drop features 10 and 13 and convert the DataFrame to a NumPy array
cc_apps

cc_apps = cc_apps.drop(cc_apps.columns[[10, 13]], axis=1) 
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X,y = cc_apps[:,0:13], cc_apps[:,13]


# Instantiate MinMaxScaler and use it to rescale
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
```

    C:\Users\nv\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\utils\validation.py:475: DataConversionWarning: Data with input dtype int64 was converted to float64 by MinMaxScaler.
      warnings.warn(msg, DataConversionWarning)
    

# 8. Splitting the dataset into train and test sets
Now that we have our data in a machine learning modeling-friendly shape, we are really ready to proceed towards creating a machine learning model to predict which credit card applications will be accepted and which will be rejected.

First, we will split our data into train set and test set to prepare our data for two different phases of machine learning modeling: training and testing.



```python
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(rescaledX,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)
```

# 9. Fitting a logistic regression model to the train set¶
Essentially, predicting if a credit card application will be approved or not is a classification task. According to UCI, our dataset contains more instances that correspond to "Denied" status than instances corresponding to "Approved" status. Specifically, out of 690 instances, there are 383 (55.5%) applications that got denied and 307 (44.5%) applications that got approved.

This gives us a benchmark. A good machine learning model should be able to accurately predict the status of the applications with respect to these statistics.

Which model should we pick? A question to ask is: are the features that affect the credit card approval decision process correlated with each other? Although we can measure correlation, that is outside the scope of this notebook, so we'll rely on our intuition that they indeed are correlated for now. Because of this correlation, we'll take advantage of the fact that generalized linear models perform well in these cases. Let's start our machine learning modeling with a Logistic Regression model (a generalized linear model).




```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(X_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



# 10. Making predictions and evaluating performance
But how well does our model perform?

We will now evaluate our model on the test set with respect to classification accuracy. But we will also take a look the model's confusion matrix. In the case of predicting credit card applications, it is equally important to see if our machine learning model is able to predict the approval status of the applications as denied that originally got denied. If our model is not performing well in this aspect, then it might end up approving the application that should have been approved. The confusion matrix helps us to view our model's performance from these aspects.




```python
# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(X_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(X_test, y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test, y_pred)
```

    Accuracy of logistic regression classifier:  0.825327510917
    




    array([[90, 13],
           [27, 99]], dtype=int64)



# 11. Grid searching and making the model perform better
Our model was pretty good! It was able to yield an accuracy score of almost 84%.

For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative instances (denied applications) predicted by the model correctly. And the last element of the second row of the confusion matrix denotes the true positives meaning the number of positive instances (approved applications) predicted by the model correctly.

Let's see if we can do better. We can perform a grid search of the model parameters to improve the model's ability to predict credit card approvals.

scikit-learn's implementation of logistic regression consists of different hyperparameters but we will grid search over the following two:

tol
max_iter




```python
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)
print(param_grid)
```

    {'tol': [0.01, 0.001, 0.0001], 'max_iter': [100, 150, 200]}
    

# 12. Finding the best performing model
We have defined the grid of hyperparameter values and converted them into a single dictionary format which GridSearchCV() expects as one of its parameters. Now, we will begin the grid search to see which values perform best.

We will instantiate GridSearchCV() with our earlier logreg model with all the data we have. Instead of passing train and test sets, we will supply rescaledX and y. We will also instruct GridSearchCV() to perform a cross-validation of five folds.

We'll end the notebook by storing the best-achieved score and the respective best parameters.

While building this credit card predictor, we tackled some of the most widely-known preprocessing steps such as scaling, label encoding, and missing value imputation. We finished with some machine learning to predict if a person's application for a credit card would get approved or not given some information about that person.




```python
# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
```

    C:\Users\nv\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\model_selection\_split.py:605: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)
    

    Best: 0.837916 using {'max_iter': 100, 'tol': 0.01}
    
