# Air Quality Index Prediction and EDA

 ## 1. Read and Clean Data


```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
```

### 1.1 Read and investigate data


```python
# read the csv file
data = pd.read_csv('station_hour.csv', low_memory=False).sort_values(by = ['Datetime', 'StationId'])
df = data.copy()

```

Let's look at the data


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>O3</th>
      <th>Benzene</th>
      <th>Toluene</th>
      <th>Xylene</th>
      <th>AQI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.941394e+06</td>
      <td>1.469831e+06</td>
      <td>2.035372e+06</td>
      <td>2.060110e+06</td>
      <td>2.098275e+06</td>
      <td>1.352465e+06</td>
      <td>2.089781e+06</td>
      <td>1.846346e+06</td>
      <td>1.863110e+06</td>
      <td>1.727504e+06</td>
      <td>1.546717e+06</td>
      <td>513979.000000</td>
      <td>2.018893e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.086481e+01</td>
      <td>1.584839e+02</td>
      <td>2.278825e+01</td>
      <td>3.523689e+01</td>
      <td>4.055115e+01</td>
      <td>2.870856e+01</td>
      <td>1.502366e+00</td>
      <td>1.211602e+01</td>
      <td>3.806408e+01</td>
      <td>3.305493e+00</td>
      <td>1.490266e+01</td>
      <td>2.448881</td>
      <td>1.801730e+02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.947618e+01</td>
      <td>1.397883e+02</td>
      <td>4.846146e+01</td>
      <td>3.497508e+01</td>
      <td>5.590894e+01</td>
      <td>2.753244e+01</td>
      <td>6.292445e+00</td>
      <td>1.467385e+01</td>
      <td>4.710653e+01</td>
      <td>1.214053e+01</td>
      <td>3.329729e+01</td>
      <td>8.973470</td>
      <td>1.404095e+02</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>0.000000e+00</td>
      <td>1.000000e-02</td>
      <td>0.000000e+00</td>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.816000e+01</td>
      <td>6.400000e+01</td>
      <td>3.050000e+00</td>
      <td>1.310000e+01</td>
      <td>1.135000e+01</td>
      <td>1.123000e+01</td>
      <td>4.100000e-01</td>
      <td>4.250000e+00</td>
      <td>1.102000e+01</td>
      <td>8.000000e-02</td>
      <td>3.400000e-01</td>
      <td>0.000000</td>
      <td>8.400000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.259000e+01</td>
      <td>1.162500e+02</td>
      <td>7.150000e+00</td>
      <td>2.479000e+01</td>
      <td>2.286000e+01</td>
      <td>2.235000e+01</td>
      <td>8.000000e-01</td>
      <td>8.250000e+00</td>
      <td>2.475000e+01</td>
      <td>9.600000e-01</td>
      <td>3.400000e+00</td>
      <td>0.200000</td>
      <td>1.310000e+02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.774000e+01</td>
      <td>2.040000e+02</td>
      <td>1.858000e+01</td>
      <td>4.548000e+01</td>
      <td>4.570000e+01</td>
      <td>3.778000e+01</td>
      <td>1.380000e+00</td>
      <td>1.453000e+01</td>
      <td>4.953000e+01</td>
      <td>3.230000e+00</td>
      <td>1.510000e+01</td>
      <td>1.830000</td>
      <td>2.590000e+02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+03</td>
      <td>1.000000e+03</td>
      <td>5.000000e+02</td>
      <td>4.999900e+02</td>
      <td>5.000000e+02</td>
      <td>4.999700e+02</td>
      <td>4.985700e+02</td>
      <td>1.999600e+02</td>
      <td>9.970000e+02</td>
      <td>4.980700e+02</td>
      <td>4.999900e+02</td>
      <td>499.990000</td>
      <td>3.133000e+03</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.boxplot(data=df)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_7_0.png)
    


From the description of the AQI in kaggle, the value of AQI is rarely above 1000. We can see that there are date enteries upto 3000.


```python
# Percentage of AQI entries above 1000
df[df['AQI']> 1000].shape[0]/df.shape[0] * 100
```




    0.1053654904072214




```python
df.drop(df[df.AQI > 1000].index, inplace=True)
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>O3</th>
      <th>Benzene</th>
      <th>Toluene</th>
      <th>Xylene</th>
      <th>AQI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.938719e+06</td>
      <td>1.469442e+06</td>
      <td>2.032749e+06</td>
      <td>2.057479e+06</td>
      <td>2.095615e+06</td>
      <td>1.352378e+06</td>
      <td>2.087161e+06</td>
      <td>1.844117e+06</td>
      <td>1.860660e+06</td>
      <td>1.724854e+06</td>
      <td>1.544114e+06</td>
      <td>511398.000000</td>
      <td>2.016165e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.079691e+01</td>
      <td>1.584340e+02</td>
      <td>2.269648e+01</td>
      <td>3.513759e+01</td>
      <td>4.045004e+01</td>
      <td>2.870922e+01</td>
      <td>1.386579e+00</td>
      <td>1.203381e+01</td>
      <td>3.808618e+01</td>
      <td>3.288109e+00</td>
      <td>1.483126e+01</td>
      <td>2.392345</td>
      <td>1.781195e+02</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.947585e+01</td>
      <td>1.396873e+02</td>
      <td>4.826406e+01</td>
      <td>3.478986e+01</td>
      <td>5.574789e+01</td>
      <td>2.753246e+01</td>
      <td>3.704627e+00</td>
      <td>1.438668e+01</td>
      <td>4.712477e+01</td>
      <td>1.209839e+01</td>
      <td>3.313170e+01</td>
      <td>8.826517</td>
      <td>1.271949e+02</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>0.000000e+00</td>
      <td>1.000000e-02</td>
      <td>0.000000e+00</td>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.811000e+01</td>
      <td>6.398000e+01</td>
      <td>3.050000e+00</td>
      <td>1.309000e+01</td>
      <td>1.134000e+01</td>
      <td>1.123000e+01</td>
      <td>4.100000e-01</td>
      <td>4.250000e+00</td>
      <td>1.103000e+01</td>
      <td>8.000000e-02</td>
      <td>3.300000e-01</td>
      <td>0.000000</td>
      <td>8.400000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.250000e+01</td>
      <td>1.162500e+02</td>
      <td>7.140000e+00</td>
      <td>2.475000e+01</td>
      <td>2.283000e+01</td>
      <td>2.235000e+01</td>
      <td>8.000000e-01</td>
      <td>8.230000e+00</td>
      <td>2.477000e+01</td>
      <td>9.500000e-01</td>
      <td>3.380000e+00</td>
      <td>0.200000</td>
      <td>1.310000e+02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.750000e+01</td>
      <td>2.040000e+02</td>
      <td>1.853000e+01</td>
      <td>4.540000e+01</td>
      <td>4.562000e+01</td>
      <td>3.778000e+01</td>
      <td>1.380000e+00</td>
      <td>1.450000e+01</td>
      <td>4.957000e+01</td>
      <td>3.220000e+00</td>
      <td>1.500000e+01</td>
      <td>1.780000</td>
      <td>2.580000e+02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+03</td>
      <td>1.000000e+03</td>
      <td>5.000000e+02</td>
      <td>4.999900e+02</td>
      <td>5.000000e+02</td>
      <td>4.999700e+02</td>
      <td>3.880400e+02</td>
      <td>1.999600e+02</td>
      <td>9.970000e+02</td>
      <td>4.980700e+02</td>
      <td>4.999900e+02</td>
      <td>499.990000</td>
      <td>1.000000e+03</td>
    </tr>
  </tbody>
</table>
</div>



Convert Datetime into pandas datetime format. 


```python
df['Datetime'] = pd.to_datetime(df['Datetime'])
```


```python
print('Date starts from {}, and ends in {}'.format(df.Datetime.min().strftime('%Y-%m-%d'), df.Datetime.max().strftime('%Y-%m-%d')))
```

    Date starts from 2015-01-01, and ends in 2020-07-01



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2586355 entries, 285144 to 2589082
    Data columns (total 16 columns):
     #   Column      Dtype         
    ---  ------      -----         
     0   StationId   object        
     1   Datetime    datetime64[ns]
     2   PM2.5       float64       
     3   PM10        float64       
     4   NO          float64       
     5   NO2         float64       
     6   NOx         float64       
     7   NH3         float64       
     8   CO          float64       
     9   SO2         float64       
     10  O3          float64       
     11  Benzene     float64       
     12  Toluene     float64       
     13  Xylene      float64       
     14  AQI         float64       
     15  AQI_Bucket  object        
    dtypes: datetime64[ns](1), float64(13), object(2)
    memory usage: 335.4+ MB


Correlation between pollutants and AQI


```python
Cor = df.corr(numeric_only = True).AQI.sort_values(ascending = False)
Cor
```




    AQI        1.000000
    PM10       0.767608
    PM2.5      0.719336
    NO2        0.408338
    NH3        0.390222
    NOx        0.379202
    NO         0.326048
    CO         0.325609
    SO2        0.273505
    Toluene    0.262648
    O3         0.152683
    Benzene    0.102557
    Xylene     0.065374
    Name: AQI, dtype: float64



### 1.2. Handing missing Values

We can see there are missig (NaN) values from the above table. To see the percentage of missing values for each column, we use the following function.


```python
df3 = df.copy()
```


```python
print(df3.isnull().sum())
```

    StationId           0
    Datetime            0
    PM2.5          647636
    PM10          1116913
    NO             553606
    NO2            528876
    NOx            490740
    NH3           1233977
    CO             499194
    SO2            742238
    O3             725695
    Benzene        861501
    Toluene       1042241
    Xylene        2074957
    AQI            570190
    AQI_Bucket     570190
    dtype: int64



```python
# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction.

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
```


```python
missing_values= missing_values_table(df3)
missing_values
```

    Your selected dataframe has 16 columns.
    There are 14 columns that have missing values.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Xylene</th>
      <td>2074957</td>
      <td>80.2</td>
    </tr>
    <tr>
      <th>NH3</th>
      <td>1233977</td>
      <td>47.7</td>
    </tr>
    <tr>
      <th>PM10</th>
      <td>1116913</td>
      <td>43.2</td>
    </tr>
    <tr>
      <th>Toluene</th>
      <td>1042241</td>
      <td>40.3</td>
    </tr>
    <tr>
      <th>Benzene</th>
      <td>861501</td>
      <td>33.3</td>
    </tr>
    <tr>
      <th>SO2</th>
      <td>742238</td>
      <td>28.7</td>
    </tr>
    <tr>
      <th>O3</th>
      <td>725695</td>
      <td>28.1</td>
    </tr>
    <tr>
      <th>PM2.5</th>
      <td>647636</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>AQI</th>
      <td>570190</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>AQI_Bucket</th>
      <td>570190</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>NO</th>
      <td>553606</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>NO2</th>
      <td>528876</td>
      <td>20.4</td>
    </tr>
    <tr>
      <th>CO</th>
      <td>499194</td>
      <td>19.3</td>
    </tr>
    <tr>
      <th>NOx</th>
      <td>490740</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>



We can see that 80% and close to 50% of data is missing for Xylene and NH3 respectively. If the majority of data is missing, usually it is safe to drop the column all together.

To hanlde missing values for the other columns, we can use:
 - with mean, median and mode imputation
 - with forward and backward filling
 - linear interpolation

#### Mean, median and mode imputation

Let's use SimpleImputer from sklearn:


```python
df3 = df3.set_index('Datetime').sort_values(by = ['StationId','Datetime'])
```


```python
from sklearn.impute import SimpleImputer

for method in ["mean", "median", "most_frequent"]:
    df3[method] = SimpleImputer(strategy=method).fit_transform(
        df3["NH3"].values.reshape(-1, 1)
    )
```

This will calculate mean, median and mode for column NH3 to fill the missing values.

#### Forward and Backward filling

Still for the same NH3 column, we can fill nan values:


```python
# fill NaN of NH3 column with forward and backward fill
df3["ffill"] = df3["NH3"].ffill()
df3["bfill"] = df3["NH3"].bfill()
```

#### Linear interpolation

Linear interpolation is an imputation technique that assumes a linear relationship between data points and utilises non-missing values from adjacent data points to compute a value for a missing data point.


```python
# interpolate for NH3 column
df3["interpolated"] = df3["NH3"].interpolate(limit_direction="both")
```

We will use the following function to plot the original distribution before and after an imputation(s) is performed:


```python
def compare_dists(original_dist, imputed_dists: dict):
    """
    Plot original_dist and imputed_dists on top of each other
    to see the difference in distributions.
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    # Plot the original
    sns.kdeplot(
        original_dist, linewidth=3, ax=ax, color="black", label="Original dist."
    )
    for key, value in imputed_dists.items():
        sns.kdeplot(value, linewidth=1, label=key, ax=ax)

    plt.legend()
    plt.show();
```


```python
compare_dists(
    df3["NH3"],
    {"mean": df3["mean"], "median": df3["median"], "mode": df3["most_frequent"], "ffill":df3["ffill"], 
     "bfill":df3["bfill"], "Linear":df3["interpolated"]},)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_35_0.png)
    


We can see that forward filling, backward filling and Linear interpolation provide a plot close to the original distribution, as is the case for time series data. We can repeat the same for the other colums to decide which method to use. 

Let's drop mean, median and mode and repeat the plot for visiblity.


```python
df3.drop(["mean", "median", "most_frequent"], axis=1, inplace=True)
```


```python
compare_dists(
    df3["NH3"],
    {"ffill":df3["ffill"], "bfill":df3["bfill"], "Linear":df3["interpolated"]},)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_38_0.png)
    


For NH3 all appear closer to the original disttribution. 

#### Repeat for PM10


```python
# fill NaN of PM10 column with forward and backward fill
df3["PM10_ffill"] = df3["PM10"].ffill()
df3["PM10_bfill"] = df3["PM10"].bfill()
```


```python
# interpolate for PM10 column
df3["PM10_interpolated"] = df3["PM10"].interpolate(limit_direction="both")
```


```python
compare_dists(
    df3["PM10"],
    {"ffill":df3["PM10_ffill"], "bfill":df3["PM10_bfill"], "Linear":df3["PM10_interpolated"]},)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_42_0.png)
    



```python
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene','Xylene']
inter = df3.loc[:, pollutants].interpolate(method = 'time', limit_direction="both"); 
df3.loc[:, pollutants] = inter

```


```python
df3.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>O3</th>
      <th>...</th>
      <th>Toluene</th>
      <th>Xylene</th>
      <th>AQI</th>
      <th>AQI_Bucket</th>
      <th>ffill</th>
      <th>bfill</th>
      <th>interpolated</th>
      <th>PM10_ffill</th>
      <th>PM10_bfill</th>
      <th>PM10_interpolated</th>
    </tr>
    <tr>
      <th>Datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-24 17:00:00</th>
      <td>AP001</td>
      <td>60.50</td>
      <td>98.00</td>
      <td>2.35</td>
      <td>30.80</td>
      <td>18.25</td>
      <td>8.50</td>
      <td>0.10</td>
      <td>11.85</td>
      <td>126.40</td>
      <td>...</td>
      <td>6.10</td>
      <td>0.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.50</td>
      <td>8.50</td>
      <td>8.50</td>
      <td>98.00</td>
      <td>98.00</td>
      <td>98.00</td>
    </tr>
    <tr>
      <th>2017-11-24 18:00:00</th>
      <td>AP001</td>
      <td>65.50</td>
      <td>111.25</td>
      <td>2.70</td>
      <td>24.20</td>
      <td>15.07</td>
      <td>9.77</td>
      <td>0.10</td>
      <td>13.17</td>
      <td>117.12</td>
      <td>...</td>
      <td>6.25</td>
      <td>0.15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.77</td>
      <td>9.77</td>
      <td>9.77</td>
      <td>111.25</td>
      <td>111.25</td>
      <td>111.25</td>
    </tr>
    <tr>
      <th>2017-11-24 19:00:00</th>
      <td>AP001</td>
      <td>80.00</td>
      <td>132.00</td>
      <td>2.10</td>
      <td>25.18</td>
      <td>15.15</td>
      <td>12.02</td>
      <td>0.10</td>
      <td>12.08</td>
      <td>98.98</td>
      <td>...</td>
      <td>5.98</td>
      <td>0.18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.02</td>
      <td>12.02</td>
      <td>12.02</td>
      <td>132.00</td>
      <td>132.00</td>
      <td>132.00</td>
    </tr>
    <tr>
      <th>2017-11-24 20:00:00</th>
      <td>AP001</td>
      <td>81.50</td>
      <td>133.25</td>
      <td>1.95</td>
      <td>16.25</td>
      <td>10.23</td>
      <td>11.58</td>
      <td>0.10</td>
      <td>10.47</td>
      <td>112.20</td>
      <td>...</td>
      <td>6.72</td>
      <td>0.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.58</td>
      <td>11.58</td>
      <td>11.58</td>
      <td>133.25</td>
      <td>133.25</td>
      <td>133.25</td>
    </tr>
    <tr>
      <th>2017-11-24 21:00:00</th>
      <td>AP001</td>
      <td>75.25</td>
      <td>116.00</td>
      <td>1.43</td>
      <td>17.48</td>
      <td>10.43</td>
      <td>12.03</td>
      <td>0.10</td>
      <td>9.12</td>
      <td>106.35</td>
      <td>...</td>
      <td>5.75</td>
      <td>0.08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.03</td>
      <td>12.03</td>
      <td>12.03</td>
      <td>116.00</td>
      <td>116.00</td>
      <td>116.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-06-30 20:00:00</th>
      <td>WB013</td>
      <td>15.55</td>
      <td>47.80</td>
      <td>7.27</td>
      <td>35.08</td>
      <td>42.38</td>
      <td>31.25</td>
      <td>0.80</td>
      <td>9.40</td>
      <td>17.24</td>
      <td>...</td>
      <td>11.57</td>
      <td>0.00</td>
      <td>59.0</td>
      <td>Satisfactory</td>
      <td>31.25</td>
      <td>31.25</td>
      <td>31.25</td>
      <td>47.80</td>
      <td>47.80</td>
      <td>47.80</td>
    </tr>
    <tr>
      <th>2020-06-30 21:00:00</th>
      <td>WB013</td>
      <td>15.23</td>
      <td>42.30</td>
      <td>6.10</td>
      <td>26.78</td>
      <td>32.85</td>
      <td>30.66</td>
      <td>0.56</td>
      <td>4.91</td>
      <td>17.46</td>
      <td>...</td>
      <td>12.29</td>
      <td>1.63</td>
      <td>59.0</td>
      <td>Satisfactory</td>
      <td>30.66</td>
      <td>30.66</td>
      <td>30.66</td>
      <td>42.30</td>
      <td>42.30</td>
      <td>42.30</td>
    </tr>
    <tr>
      <th>2020-06-30 22:00:00</th>
      <td>WB013</td>
      <td>11.40</td>
      <td>40.95</td>
      <td>6.58</td>
      <td>19.53</td>
      <td>26.12</td>
      <td>30.73</td>
      <td>0.61</td>
      <td>3.81</td>
      <td>17.24</td>
      <td>...</td>
      <td>8.88</td>
      <td>0.55</td>
      <td>59.0</td>
      <td>Satisfactory</td>
      <td>30.73</td>
      <td>30.73</td>
      <td>30.73</td>
      <td>40.95</td>
      <td>40.95</td>
      <td>40.95</td>
    </tr>
    <tr>
      <th>2020-06-30 23:00:00</th>
      <td>WB013</td>
      <td>9.25</td>
      <td>34.33</td>
      <td>9.17</td>
      <td>21.85</td>
      <td>31.00</td>
      <td>29.61</td>
      <td>0.65</td>
      <td>3.44</td>
      <td>12.74</td>
      <td>...</td>
      <td>8.43</td>
      <td>1.63</td>
      <td>59.0</td>
      <td>Satisfactory</td>
      <td>29.61</td>
      <td>29.61</td>
      <td>29.61</td>
      <td>34.33</td>
      <td>34.33</td>
      <td>34.33</td>
    </tr>
    <tr>
      <th>2020-07-01 00:00:00</th>
      <td>WB013</td>
      <td>10.50</td>
      <td>36.50</td>
      <td>7.78</td>
      <td>22.50</td>
      <td>30.25</td>
      <td>27.23</td>
      <td>0.58</td>
      <td>2.80</td>
      <td>13.10</td>
      <td>...</td>
      <td>7.39</td>
      <td>4.18</td>
      <td>59.0</td>
      <td>Satisfactory</td>
      <td>27.23</td>
      <td>27.23</td>
      <td>27.23</td>
      <td>36.50</td>
      <td>36.50</td>
      <td>36.50</td>
    </tr>
  </tbody>
</table>
<p>2586355 rows × 21 columns</p>
</div>



From the explanation for AQI index calculation, we can follow the same steps used to calculate AQI and fill the missing values of AQI_Bucket based on the the calcualted AQI.

#### AQI Calculation


```python
df3["PM10_24hr_avg"] = df3.groupby("StationId")["PM10"].rolling(window = 24, min_periods = 16).mean().values
df3["PM2.5_24hr_avg"] = df3.groupby("StationId")["PM2.5"].rolling(window = 24, min_periods = 16).mean().values
df3["SO2_24hr_avg"] = df3.groupby("StationId")["SO2"].rolling(window = 24, min_periods = 16).mean().values
df3["NOx_24hr_avg"] = df3.groupby("StationId")["NOx"].rolling(window = 24, min_periods = 16).mean().values
df3["NH3_24hr_avg"] = df3.groupby("StationId")["NH3"].rolling(window = 24, min_periods = 16).mean().values
df3["CO_8hr_max"] = df3.groupby("StationId")["CO"].rolling(window = 8, min_periods = 1).max().values
df3["O3_8hr_max"] = df3.groupby("StationId")["O3"].rolling(window = 8, min_periods = 1).max().values
```

#### PM2.5


```python
## PM2.5 Sub-Index calculation
def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0

df3["PM2.5_SubIndex"] = df3["PM2.5_24hr_avg"].apply(lambda x: get_PM25_subindex(x))
```

#### PM10


```python
def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0

df3["PM10_SubIndex"] = df3["PM10_24hr_avg"].apply(lambda x: get_PM10_subindex(x))
```

#### SO2


```python
## SO2 Sub-Index calculation
def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

df3["SO2_SubIndex"] = df3["SO2_24hr_avg"].apply(lambda x: get_SO2_subindex(x))
```

#### NOx (Any Nitric x-oxide)


```python
## NOx Sub-Index calculation
def get_NOx_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

df3["NOx_SubIndex"] = df3["NOx_24hr_avg"].apply(lambda x: get_NOx_subindex(x))
```

#### NH3 (Ammonia)


```python
## NH3 Sub-Index calculation
def get_NH3_subindex(x):
    if x <= 200:
        return x * 50 / 200
    elif x <= 400:
        return 50 + (x - 200) * 50 / 200
    elif x <= 800:
        return 100 + (x - 400) * 100 / 400
    elif x <= 1200:
        return 200 + (x - 800) * 100 / 400
    elif x <= 1800:
        return 300 + (x - 1200) * 100 / 600
    elif x > 1800:
        return 400 + (x - 1800) * 100 / 600
    else:
        return 0

df3["NH3_SubIndex"] = df3["NH3_24hr_avg"].apply(lambda x: get_NH3_subindex(x))
```

#### CO (Carbon Monoxide)


```python
## CO Sub-Index calculation
def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

df3["CO_SubIndex"] = df3["CO_8hr_max"].apply(lambda x: get_CO_subindex(x))
```

#### O3 (Ozone or Trioxygen)


```python
## O3 Sub-Index calculation
def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0

df3["O3_SubIndex"] = df3["O3_8hr_max"].apply(lambda x: get_O3_subindex(x))
```

#### AQI


```python
## AQI bucketing
def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return np.NaN

df3["Checks"] = (df3["PM2.5_SubIndex"] > 0).astype(int) + \
                (df3["PM10_SubIndex"] > 0).astype(int) + \
                (df3["SO2_SubIndex"] > 0).astype(int) + \
                (df3["NOx_SubIndex"] > 0).astype(int) + \
                (df3["NH3_SubIndex"] > 0).astype(int) + \
                (df3["CO_SubIndex"] > 0).astype(int) + \
                (df3["O3_SubIndex"] > 0).astype(int)

df3["AQI_calculated"] = round(df3[["PM2.5_SubIndex", "PM10_SubIndex", "SO2_SubIndex", "NOx_SubIndex",
                                 "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis = 1))
df3.loc[df3["PM2.5_SubIndex"] + df3["PM10_SubIndex"] <= 0, "AQI_calculated"] = np.NaN
df3.loc[df3.Checks < 3, "AQI_calculated"] = np.NaN

df3["AQI_bucket_calculated"] = df3["AQI_calculated"].apply(lambda x: get_AQI_bucket(x))

df3[~df3.AQI_calculated.isna()].head(3)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>O3</th>
      <th>...</th>
      <th>PM2.5_SubIndex</th>
      <th>PM10_SubIndex</th>
      <th>SO2_SubIndex</th>
      <th>NOx_SubIndex</th>
      <th>NH3_SubIndex</th>
      <th>CO_SubIndex</th>
      <th>O3_SubIndex</th>
      <th>Checks</th>
      <th>AQI_calculated</th>
      <th>AQI_bucket_calculated</th>
    </tr>
    <tr>
      <th>Datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-25 08:00:00</th>
      <td>AP001</td>
      <td>314.0</td>
      <td>218.36</td>
      <td>2.17</td>
      <td>12.20</td>
      <td>42.87</td>
      <td>22.52</td>
      <td>0.85</td>
      <td>21.78</td>
      <td>0.16</td>
      <td>...</td>
      <td>199.218750</td>
      <td>115.473333</td>
      <td>14.855469</td>
      <td>17.641406</td>
      <td>3.043594</td>
      <td>42.5</td>
      <td>131.617647</td>
      <td>7</td>
      <td>199.0</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>2017-11-25 09:00:00</th>
      <td>AP001</td>
      <td>104.0</td>
      <td>148.50</td>
      <td>1.93</td>
      <td>23.00</td>
      <td>13.75</td>
      <td>9.80</td>
      <td>0.10</td>
      <td>15.30</td>
      <td>117.62</td>
      <td>...</td>
      <td>202.009804</td>
      <td>116.465098</td>
      <td>15.106618</td>
      <td>17.614706</td>
      <td>3.008676</td>
      <td>42.5</td>
      <td>125.911765</td>
      <td>7</td>
      <td>202.0</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>2017-11-25 10:00:00</th>
      <td>AP001</td>
      <td>94.5</td>
      <td>142.00</td>
      <td>1.33</td>
      <td>16.25</td>
      <td>9.75</td>
      <td>9.65</td>
      <td>0.10</td>
      <td>17.00</td>
      <td>136.23</td>
      <td>...</td>
      <td>202.731481</td>
      <td>117.105926</td>
      <td>15.447917</td>
      <td>17.313194</td>
      <td>2.975556</td>
      <td>42.5</td>
      <td>153.279412</td>
      <td>7</td>
      <td>203.0</td>
      <td>Poor</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 38 columns</p>
</div>




```python
df3[~df3.AQI_calculated.isna()].AQI_bucket_calculated.value_counts()
```




    AQI_bucket_calculated
    Moderate        964797
    Satisfactory    513796
    Very Poor       473129
    Poor            390346
    Severe          150594
    Good             92043
    Name: count, dtype: int64




```python
print(df3.isnull().sum()/df3.shape[0] * 100)
```

    StationId                 0.000000
    PM2.5                     0.000000
    PM10                      0.000000
    NO                        0.000000
    NO2                       0.000000
    NOx                       0.000000
    NH3                       0.000000
    CO                        0.000000
    SO2                       0.000000
    O3                        0.000000
    Benzene                   0.000000
    Toluene                   0.000000
    Xylene                    0.000000
    AQI                      22.046084
    AQI_Bucket               22.046084
    ffill                     0.000000
    bfill                     0.000000
    interpolated              0.000000
    PM10_ffill                0.000000
    PM10_bfill                0.000000
    PM10_interpolated         0.000000
    PM10_24hr_avg             0.063796
    PM2.5_24hr_avg            0.063796
    SO2_24hr_avg              0.063796
    NOx_24hr_avg              0.063796
    NH3_24hr_avg              0.063796
    CO_8hr_max                0.000000
    O3_8hr_max                0.000000
    PM2.5_SubIndex            0.000000
    PM10_SubIndex             0.000000
    SO2_SubIndex              0.000000
    NOx_SubIndex              0.000000
    NH3_SubIndex              0.000000
    CO_SubIndex               0.000000
    O3_SubIndex               0.000000
    Checks                    0.000000
    AQI_calculated            0.063796
    AQI_bucket_calculated     0.063796
    dtype: float64


We can drop the remaining NAN values since they account only 0.06% of the data.


```python
df3 = df3[~df3.AQI_calculated.isna()]
```


```python
df3=df3.drop(['ffill', 'bfill', 'interpolated', 'PM10_ffill','PM10_bfill','PM10_interpolated', 'AQI', 'AQI_Bucket'], axis=1)
```


```python
df3 = df3.rename(columns={'AQI_calculated': 'AQI', 'AQI_bucket_calculated': 'AQI_Bucket'})
```


```python
df3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 2584705 entries, 2017-11-25 08:00:00 to 2020-07-01 00:00:00
    Data columns (total 30 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   StationId       object 
     1   PM2.5           float64
     2   PM10            float64
     3   NO              float64
     4   NO2             float64
     5   NOx             float64
     6   NH3             float64
     7   CO              float64
     8   SO2             float64
     9   O3              float64
     10  Benzene         float64
     11  Toluene         float64
     12  Xylene          float64
     13  PM10_24hr_avg   float64
     14  PM2.5_24hr_avg  float64
     15  SO2_24hr_avg    float64
     16  NOx_24hr_avg    float64
     17  NH3_24hr_avg    float64
     18  CO_8hr_max      float64
     19  O3_8hr_max      float64
     20  PM2.5_SubIndex  float64
     21  PM10_SubIndex   float64
     22  SO2_SubIndex    float64
     23  NOx_SubIndex    float64
     24  NH3_SubIndex    float64
     25  CO_SubIndex     float64
     26  O3_SubIndex     float64
     27  Checks          int64  
     28  AQI             float64
     29  AQI_Bucket      object 
    dtypes: float64(27), int64(1), object(2)
    memory usage: 611.3+ MB


## 3. Explanatory Data Analysis 


```python
df4 = df3.copy()
```

### Most polluted stations for highly correlated pollutants


```python
# correlation afer new features are added  
corr_ext = df4.corr(numeric_only = True).AQI.sort_values(ascending = False)
major_pollutants = (corr_ext[corr_ext>0.5]).index
major_pollutants.tolist()
```




    ['AQI',
     'PM2.5_SubIndex',
     'PM2.5_24hr_avg',
     'PM10_24hr_avg',
     'PM10_SubIndex',
     'PM2.5',
     'PM10',
     'CO_SubIndex']




```python
major_pollutants.insert(0, 'StationId')
```




    Index(['StationId', 'AQI', 'PM2.5_SubIndex', 'PM2.5_24hr_avg', 'PM10_24hr_avg',
           'PM10_SubIndex', 'PM2.5', 'PM10', 'CO_SubIndex'],
          dtype='object')




```python
most_polluted = df4.groupby(['StationId'])[major_pollutants].mean().sort_values(by = 'AQI', ascending = False).head(5)
most_polluted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AQI</th>
      <th>PM2.5_SubIndex</th>
      <th>PM2.5_24hr_avg</th>
      <th>PM10_24hr_avg</th>
      <th>PM10_SubIndex</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>CO_SubIndex</th>
    </tr>
    <tr>
      <th>StationId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GJ001</th>
      <td>319.378577</td>
      <td>150.512882</td>
      <td>76.004528</td>
      <td>165.744721</td>
      <td>145.177296</td>
      <td>75.927363</td>
      <td>165.605159</td>
      <td>263.945141</td>
    </tr>
    <tr>
      <th>DL002</th>
      <td>310.571699</td>
      <td>238.322695</td>
      <td>135.054147</td>
      <td>296.831982</td>
      <td>275.912778</td>
      <td>135.051371</td>
      <td>296.808978</td>
      <td>111.113549</td>
    </tr>
    <tr>
      <th>DL028</th>
      <td>274.632880</td>
      <td>208.158962</td>
      <td>112.758675</td>
      <td>134.091122</td>
      <td>120.015068</td>
      <td>112.745352</td>
      <td>134.093209</td>
      <td>85.765622</td>
    </tr>
    <tr>
      <th>DL038</th>
      <td>269.730716</td>
      <td>228.803371</td>
      <td>126.967274</td>
      <td>271.867159</td>
      <td>244.559199</td>
      <td>126.889543</td>
      <td>271.688092</td>
      <td>88.071505</td>
    </tr>
    <tr>
      <th>DL020</th>
      <td>265.610949</td>
      <td>216.114350</td>
      <td>121.987045</td>
      <td>263.574578</td>
      <td>239.971677</td>
      <td>121.987999</td>
      <td>263.553739</td>
      <td>78.847719</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.style.use('seaborn-whitegrid')
f, ax_ = plt.subplots((len(major_pollutants) + 1) // 2, 2, figsize = (12,16))

axes = ax_.flatten()

    
for i, pol in enumerate(major_pollutants):
    ax = axes[i]
    sns.barplot(x = most_polluted[pol],
                y = most_polluted.index,
                palette = 'RdBu', ax = ax);
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=14)
    ax.set_title(pol)
f.tight_layout()
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_77_0.png)
    


It appears that CO_SubIndex has a significant impact on AQI. Despite station GJ001 having relatively lower levels of other pollutants, its high AQI value could be attributed to the influence of CO_SubIndex or missing data.

### Looking at the pollutants and their distribution


```python
axes = df4[major_pollutants].plot(marker='.',alpha = 0.5, linestyle = 'None', figsize = (16,20), subplots = True)
for ax in axes:
    ax.set_xlabel('Years')
    ax.set_ylim(0,1000)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_80_0.png)
    


From the plots, we can make the following observations:
 - PM10 has high value
 - most exhibited seasonality though further observations are required
 - all pollutants exhibt low values between 2017 and 2018. Either some phenomenon that affected the values or because of missing data


```python
mask = (df['Datetime'] >= '2017-01-01') & (df['Datetime'] <= '2018-01-01')
filtered_df = df.loc[mask]
```


```python
filtered_df.isna().sum()
```




    StationId          0
    Datetime           0
    PM2.5         179893
    PM10          238813
    NO            168918
    NO2           165854
    NOx           141419
    NH3           247703
    CO            119957
    SO2           190278
    O3            197210
    Benzene       138467
    Toluene       147197
    Xylene        242330
    AQI           173173
    AQI_Bucket    173173
    dtype: int64



#### Let's see the correlation of the major pollutants.



```python
sns.pairplot(df4.sample(frac=0.1).reset_index(), vars=major_pollutants, hue ='AQI_Bucket')
```




    <seaborn.axisgrid.PairGrid at 0x7fc96d14de40>




    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_85_1.png)
    


Overall: Most variables show normal distribution skewed to the right 
 - AQI: There appears to be more data points for satisfactory and moderate AQI
 


```python
def plot_(df, pollutants, stations):
    # Calculate the number of rows and columns for the grid
    num_rows = int(len(pollutants) / 2) + len(pollutants) % 2
    num_cols = 2

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))

    # Flatten the axes array to easily iterate over the subplots
    axes = axes.flatten()

    # Plot each pollutant for all stations
    for i, pol in enumerate(pollutants):
        ax = axes[i]
        for st in stations:
            df.loc[st][pol].plot(ax=ax, label=st)
        #ax.set_xlabel('Hour')
        ax.set_title(f'{pol} Variation over Time')
        ax.legend()

    # Hide any unused subplots
    for j in range(len(pollutants), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
```

#### Seasonality in the data¶
Let's convert the data into daily, weekly, and monthly versions.


```python
df4['Year'] = df4.index.year
df4['Month'] = df4.index.month
df4['Week'] = df4.index.dayofweek
df4['Day'] = df4.index.day
df4['Hour'] = df4.index.hour
```


```python
df4.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>O3</th>
      <th>...</th>
      <th>CO_SubIndex</th>
      <th>O3_SubIndex</th>
      <th>Checks</th>
      <th>AQI</th>
      <th>AQI_Bucket</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Hour</th>
    </tr>
    <tr>
      <th>Datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-25 08:00:00</th>
      <td>AP001</td>
      <td>314.0</td>
      <td>218.36</td>
      <td>2.17</td>
      <td>12.20</td>
      <td>42.87</td>
      <td>22.52</td>
      <td>0.85</td>
      <td>21.78</td>
      <td>0.16</td>
      <td>...</td>
      <td>42.5</td>
      <td>131.617647</td>
      <td>7</td>
      <td>199.0</td>
      <td>Moderate</td>
      <td>2017</td>
      <td>11</td>
      <td>5</td>
      <td>25</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2017-11-25 09:00:00</th>
      <td>AP001</td>
      <td>104.0</td>
      <td>148.50</td>
      <td>1.93</td>
      <td>23.00</td>
      <td>13.75</td>
      <td>9.80</td>
      <td>0.10</td>
      <td>15.30</td>
      <td>117.62</td>
      <td>...</td>
      <td>42.5</td>
      <td>125.911765</td>
      <td>7</td>
      <td>202.0</td>
      <td>Poor</td>
      <td>2017</td>
      <td>11</td>
      <td>5</td>
      <td>25</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2017-11-25 10:00:00</th>
      <td>AP001</td>
      <td>94.5</td>
      <td>142.00</td>
      <td>1.33</td>
      <td>16.25</td>
      <td>9.75</td>
      <td>9.65</td>
      <td>0.10</td>
      <td>17.00</td>
      <td>136.23</td>
      <td>...</td>
      <td>42.5</td>
      <td>153.279412</td>
      <td>7</td>
      <td>203.0</td>
      <td>Poor</td>
      <td>2017</td>
      <td>11</td>
      <td>5</td>
      <td>25</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 35 columns</p>
</div>




```python
hour_day = df4.groupby(['StationId', 'Hour'])[major_pollutants].mean()
```


```python
hour_day.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>AQI</th>
      <th>PM2.5_SubIndex</th>
      <th>PM2.5_24hr_avg</th>
      <th>PM10_24hr_avg</th>
      <th>PM10_SubIndex</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>CO_SubIndex</th>
    </tr>
    <tr>
      <th>StationId</th>
      <th>Hour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">AP001</th>
      <th>0</th>
      <td>110.860906</td>
      <td>82.043054</td>
      <td>43.039829</td>
      <td>86.436156</td>
      <td>81.493065</td>
      <td>49.140832</td>
      <td>94.203751</td>
      <td>55.620610</td>
    </tr>
    <tr>
      <th>1</th>
      <td>108.764768</td>
      <td>82.066440</td>
      <td>43.060924</td>
      <td>86.488292</td>
      <td>81.519935</td>
      <td>47.375032</td>
      <td>92.259831</td>
      <td>56.452015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>107.398734</td>
      <td>81.980444</td>
      <td>43.058738</td>
      <td>86.484743</td>
      <td>81.497717</td>
      <td>44.973523</td>
      <td>90.178745</td>
      <td>56.634486</td>
    </tr>
  </tbody>
</table>
</div>




```python
day_week = df4.groupby(['StationId', 'Week'])[attr].mean()
month_year = df4.groupby(['StationId', 'Month'])[attr].mean()
```


```python
plot_(hour_day, major_pollutants, most_polluted.index)
#hour_day.plot(x="Hour", y=pollutants)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_94_0.png)
    


The concentration of pollutants exhibits a pattern throughout the day, with the low levels occurring around dawn and early evening; and the high levels around 10 o'clock. 
The average AQI remains constant, as expected based on its calculation method. Consequently, employing an **hourly prediction window is not suitable** in this scenario.


```python
plot_(day_week, major_pollutants, most_polluted.index)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_96_0.png)
    


The pollution levels are generally consistent throughout the week, with only a slight variation observed on weekends. This pattern is expected, as human activities are typically reduced during weekends. However, it is important to note that some stations experience an increase in pollution levels during weekends, indicating that certain factors may contribute to higher pollution levels in those areas.


```python
plot_(month_year, major_pollutants, most_polluted.index)
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_98_0.png)
    


There is evident seasonality observed for most pollutants. It is evident that pollution levels tend to decrease during the summer season and increase during colder seasons.

### Trends in the data


```python
daily = df4.groupby('StationId').resample('D')[major_pollutants].mean()
weekly = df4.groupby('StationId').resample('w')[major_pollutants].mean()
monthly = df4.groupby('StationId').resample('M')[major_pollutants].mean()
yearly = df4.groupby('StationId').resample('Y')[major_pollutants].mean()
```


```python
daily.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>AQI</th>
      <th>PM2.5_SubIndex</th>
      <th>PM2.5_24hr_avg</th>
      <th>PM10_24hr_avg</th>
      <th>PM10_SubIndex</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>CO_SubIndex</th>
    </tr>
    <tr>
      <th>StationId</th>
      <th>Datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">AP001</th>
      <th>2017-11-25</th>
      <td>196.437500</td>
      <td>196.326742</td>
      <td>88.898023</td>
      <td>127.171607</td>
      <td>118.114404</td>
      <td>97.083750</td>
      <td>136.788125</td>
      <td>34.375000</td>
    </tr>
    <tr>
      <th>2017-11-26</th>
      <td>205.416667</td>
      <td>183.630903</td>
      <td>85.089271</td>
      <td>129.422101</td>
      <td>119.614734</td>
      <td>78.322917</td>
      <td>129.062500</td>
      <td>9.125000</td>
    </tr>
    <tr>
      <th>2017-11-27</th>
      <td>204.375000</td>
      <td>188.850868</td>
      <td>86.655260</td>
      <td>137.114826</td>
      <td>124.743218</td>
      <td>111.780417</td>
      <td>154.661250</td>
      <td>38.723958</td>
    </tr>
  </tbody>
</table>
</div>



Let's plot the daily, weekly and monthly AQI time series together over a single 12-month period to compare them.


```python
###### Start and end of the date range to extract
start, end = '2018-01', '2018-12'
# Plot daily and weekly resampled time series together
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(daily['AQI']['AP001'].loc[start:end], marker='.', linestyle='-', linewidth=0.3, label='Daily')
ax.plot(weekly['AQI']['AP001'].loc[start:end], marker='.', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.plot(monthly['AQI']['AP001'].loc[start:end], marker='.', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.legend();
plt.suptitle('AQI of AP001')
```




    Text(0.5, 0.98, 'AQI of AP001')




    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_104_1.png)
    


We can observe that the weekly mean time series appears smoother compared to the daily time series. This smoothing is a result of averaging out the higher frequency variability during the resampling process. Additionally, there doesn't seem to be a noticeable trend in the data.


```python
fig, ax = plt.subplots(figsize=(12, 5))

for station in most_polluted.index:
    ax.plot(monthly['AQI'].loc[station], marker='.', linestyle='-', label=station)

# Additional code for labeling and styling the plot
ax.set_xlabel('Year')
ax.set_ylabel('AQI')
ax.set_title('Yearly AQI for Most Polluted Stations')
ax.legend()

plt.show()
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_106_0.png)
    


## 4. Feature Engineering

We will create some features based on our EDA above. Our earlier analysis has shown seasonality in during the time of the day, weekday/weekend, month and year.

Therefore, we will add new features Season, Year, Month, Part_of_Day, Is_Weekend to reflect the above.

Since we ruled out on a window less than a day(constant AQI throughout the day), we will use the daily resampled data.


```python
df5 = df4.copy()
```


```python
df5.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 2584705 entries, 2017-11-25 08:00:00 to 2020-07-01 00:00:00
    Data columns (total 35 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   StationId       object 
     1   PM2.5           float64
     2   PM10            float64
     3   NO              float64
     4   NO2             float64
     5   NOx             float64
     6   NH3             float64
     7   CO              float64
     8   SO2             float64
     9   O3              float64
     10  Benzene         float64
     11  Toluene         float64
     12  Xylene          float64
     13  PM10_24hr_avg   float64
     14  PM2.5_24hr_avg  float64
     15  SO2_24hr_avg    float64
     16  NOx_24hr_avg    float64
     17  NH3_24hr_avg    float64
     18  CO_8hr_max      float64
     19  O3_8hr_max      float64
     20  PM2.5_SubIndex  float64
     21  PM10_SubIndex   float64
     22  SO2_SubIndex    float64
     23  NOx_SubIndex    float64
     24  NH3_SubIndex    float64
     25  CO_SubIndex     float64
     26  O3_SubIndex     float64
     27  Checks          int64  
     28  AQI             float64
     29  AQI_Bucket      object 
     30  Year            int32  
     31  Month           int32  
     32  Week            int32  
     33  Day             int32  
     34  Hour            int32  
    dtypes: float64(27), int32(5), int64(1), object(2)
    memory usage: 660.6+ MB


We will remove AQI_Becket since we can infer it's value with get_AQI_Bucket() function once AQI prediction is done.


```python
df5 = df5.drop(['AQI_Bucket'], axis=1)
```


```python
df5 = df5.groupby('StationId').resample('D').mean().reset_index()
df5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>Datetime</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>...</th>
      <th>NH3_SubIndex</th>
      <th>CO_SubIndex</th>
      <th>O3_SubIndex</th>
      <th>Checks</th>
      <th>AQI</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AP001</td>
      <td>2017-11-25</td>
      <td>97.083750</td>
      <td>136.788125</td>
      <td>1.756875</td>
      <td>22.628125</td>
      <td>15.565000</td>
      <td>11.985000</td>
      <td>0.198125</td>
      <td>16.628750</td>
      <td>...</td>
      <td>2.965166</td>
      <td>34.375000</td>
      <td>178.391544</td>
      <td>7.0</td>
      <td>196.437500</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>25.0</td>
      <td>15.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AP001</td>
      <td>2017-11-26</td>
      <td>78.322917</td>
      <td>129.062500</td>
      <td>1.255417</td>
      <td>25.995833</td>
      <td>14.848333</td>
      <td>10.279167</td>
      <td>0.140417</td>
      <td>26.964583</td>
      <td>...</td>
      <td>2.755109</td>
      <td>9.125000</td>
      <td>193.332721</td>
      <td>7.0</td>
      <td>205.416667</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AP001</td>
      <td>2017-11-27</td>
      <td>111.780417</td>
      <td>154.661250</td>
      <td>9.397500</td>
      <td>33.808750</td>
      <td>41.526667</td>
      <td>14.015833</td>
      <td>0.657917</td>
      <td>25.658750</td>
      <td>...</td>
      <td>2.898368</td>
      <td>38.723958</td>
      <td>176.316213</td>
      <td>7.0</td>
      <td>204.375000</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AP001</td>
      <td>2017-11-28</td>
      <td>91.430000</td>
      <td>153.668333</td>
      <td>25.286250</td>
      <td>53.073750</td>
      <td>25.570417</td>
      <td>39.244583</td>
      <td>1.365000</td>
      <td>15.060833</td>
      <td>...</td>
      <td>8.227122</td>
      <td>156.537071</td>
      <td>155.996667</td>
      <td>7.0</td>
      <td>285.625000</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AP001</td>
      <td>2017-11-29</td>
      <td>71.177083</td>
      <td>118.970833</td>
      <td>5.228750</td>
      <td>23.205000</td>
      <td>16.586667</td>
      <td>12.965000</td>
      <td>0.162917</td>
      <td>10.565417</td>
      <td>...</td>
      <td>5.200356</td>
      <td>18.166667</td>
      <td>155.988725</td>
      <td>7.0</td>
      <td>192.166667</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>29.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107944</th>
      <td>WB013</td>
      <td>2020-06-27</td>
      <td>8.651250</td>
      <td>16.461667</td>
      <td>6.299583</td>
      <td>18.034583</td>
      <td>16.366250</td>
      <td>17.696667</td>
      <td>0.690000</td>
      <td>4.364583</td>
      <td>...</td>
      <td>5.014076</td>
      <td>40.270833</td>
      <td>46.651250</td>
      <td>7.0</td>
      <td>49.833333</td>
      <td>2020.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>107945</th>
      <td>WB013</td>
      <td>2020-06-28</td>
      <td>13.866667</td>
      <td>18.875833</td>
      <td>16.116250</td>
      <td>14.934167</td>
      <td>25.919583</td>
      <td>24.148333</td>
      <td>0.681250</td>
      <td>3.490833</td>
      <td>...</td>
      <td>5.228030</td>
      <td>38.833333</td>
      <td>63.438333</td>
      <td>7.0</td>
      <td>65.291667</td>
      <td>2020.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>28.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>107946</th>
      <td>WB013</td>
      <td>2020-06-29</td>
      <td>20.087083</td>
      <td>39.108750</td>
      <td>18.993333</td>
      <td>33.538333</td>
      <td>38.487500</td>
      <td>16.212083</td>
      <td>0.774583</td>
      <td>5.122500</td>
      <td>...</td>
      <td>5.261701</td>
      <td>42.020833</td>
      <td>62.919167</td>
      <td>7.0</td>
      <td>63.041667</td>
      <td>2020.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>107947</th>
      <td>WB013</td>
      <td>2020-06-30</td>
      <td>16.069167</td>
      <td>39.302917</td>
      <td>15.685000</td>
      <td>21.261667</td>
      <td>29.502500</td>
      <td>27.783750</td>
      <td>0.694167</td>
      <td>5.877083</td>
      <td>...</td>
      <td>5.315764</td>
      <td>43.937500</td>
      <td>49.443750</td>
      <td>7.0</td>
      <td>57.125000</td>
      <td>2020.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>107948</th>
      <td>WB013</td>
      <td>2020-07-01</td>
      <td>10.500000</td>
      <td>36.500000</td>
      <td>7.780000</td>
      <td>22.500000</td>
      <td>30.250000</td>
      <td>27.230000</td>
      <td>0.580000</td>
      <td>2.800000</td>
      <td>...</td>
      <td>7.139479</td>
      <td>40.500000</td>
      <td>58.960000</td>
      <td>7.0</td>
      <td>59.000000</td>
      <td>2020.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>107949 rows × 35 columns</p>
</div>



Following the [indian season names](https://en.wikipedia.org/wiki/Climate_of_India), the following code creates a column 'Season'


```python
season_dict = {1: 'Winter',
               2: 'Winter',
               3: 'pre-monsoon', 
               4: 'pre-monsoon',
               5: 'pre-monsoon',
               6: 'Monsoon',
               7: 'Monsoon',
               8: 'Monsoon',
               9: 'Monsoon',
               10: 'Post-monsoon',
               11: 'Post-monsoon',
               12: 'Post-monsoon'}
df5['Season'] = df5['Month'].apply(lambda x: season_dict[x])
```


```python
def part_of_day(hour):
    return (
        "morning" if 5 <= hour <= 11
        else
        "afternoon" if 12 <= hour <= 17
        else
        "evening" if 18 <= hour <= 22
        else
        "night"
    )
```


```python
df5['Part_of_Day'] = df5['Hour'].apply(lambda x: part_of_day(x))

```


```python
df5["is_weekend"] = df5['Datetime'].dt.dayofweek > 4
```

## 5. Data Preprocessing
We will use StandardScaler for numerical attributes and Onehot encoding to transform categorical attributes. We will also split the data into train and test. 

Using temporal variable is a more reliable way of splitting datasets whenever the dataset includes the date variable, and we want to predict something in the future that depends on date. 
Here we are going to split the dataframe into 85% train and 15%.


```python
# copy of dataset
df6 = df5.copy()
```


```python
df6.sort_values(by=['Datetime','StationId'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>Datetime</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>...</th>
      <th>Checks</th>
      <th>AQI</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Season</th>
      <th>Part_of_Day</th>
      <th>is_weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11891</th>
      <td>DL007</td>
      <td>2015-01-01</td>
      <td>357.2650</td>
      <td>558.18</td>
      <td>40.07250</td>
      <td>46.2100</td>
      <td>86.28375</td>
      <td>17.42125</td>
      <td>0.75125</td>
      <td>69.2375</td>
      <td>...</td>
      <td>7.0</td>
      <td>675.0</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>19.5</td>
      <td>Winter</td>
      <td>evening</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13900</th>
      <td>DL008</td>
      <td>2015-01-01</td>
      <td>360.6225</td>
      <td>558.18</td>
      <td>59.90625</td>
      <td>40.3125</td>
      <td>70.18750</td>
      <td>23.93750</td>
      <td>7.02125</td>
      <td>69.2375</td>
      <td>...</td>
      <td>7.0</td>
      <td>675.0</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>19.5</td>
      <td>Winter</td>
      <td>evening</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20260</th>
      <td>DL013</td>
      <td>2015-01-01</td>
      <td>203.6875</td>
      <td>558.18</td>
      <td>21.13000</td>
      <td>12.3125</td>
      <td>28.07125</td>
      <td>44.98125</td>
      <td>20.00000</td>
      <td>18.1250</td>
      <td>...</td>
      <td>7.0</td>
      <td>675.0</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>19.5</td>
      <td>Winter</td>
      <td>evening</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30586</th>
      <td>DL021</td>
      <td>2015-01-01</td>
      <td>360.6225</td>
      <td>558.18</td>
      <td>151.69875</td>
      <td>80.3125</td>
      <td>280.09625</td>
      <td>17.42125</td>
      <td>22.68750</td>
      <td>17.6875</td>
      <td>...</td>
      <td>7.0</td>
      <td>675.0</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>19.5</td>
      <td>Winter</td>
      <td>evening</td>
      <td>False</td>
    </tr>
    <tr>
      <th>44502</th>
      <td>DL033</td>
      <td>2015-01-01</td>
      <td>360.6225</td>
      <td>558.18</td>
      <td>151.78625</td>
      <td>38.2500</td>
      <td>223.38125</td>
      <td>17.42125</td>
      <td>10.37625</td>
      <td>10.8750</td>
      <td>...</td>
      <td>7.0</td>
      <td>675.0</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>19.5</td>
      <td>Winter</td>
      <td>evening</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>105808</th>
      <td>WB009</td>
      <td>2020-07-01</td>
      <td>10.5200</td>
      <td>34.55</td>
      <td>1.75000</td>
      <td>11.0200</td>
      <td>12.78000</td>
      <td>3.07000</td>
      <td>0.23000</td>
      <td>3.1200</td>
      <td>...</td>
      <td>7.0</td>
      <td>38.0</td>
      <td>2020.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monsoon</td>
      <td>night</td>
      <td>False</td>
    </tr>
    <tr>
      <th>106202</th>
      <td>WB010</td>
      <td>2020-07-01</td>
      <td>14.7000</td>
      <td>29.00</td>
      <td>3.10000</td>
      <td>6.7800</td>
      <td>9.88000</td>
      <td>6.78000</td>
      <td>0.33000</td>
      <td>3.2300</td>
      <td>...</td>
      <td>7.0</td>
      <td>35.0</td>
      <td>2020.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monsoon</td>
      <td>night</td>
      <td>False</td>
    </tr>
    <tr>
      <th>106805</th>
      <td>WB011</td>
      <td>2020-07-01</td>
      <td>12.5700</td>
      <td>41.40</td>
      <td>18.00000</td>
      <td>13.7000</td>
      <td>31.70000</td>
      <td>28.73000</td>
      <td>0.21000</td>
      <td>8.3600</td>
      <td>...</td>
      <td>7.0</td>
      <td>62.0</td>
      <td>2020.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monsoon</td>
      <td>night</td>
      <td>False</td>
    </tr>
    <tr>
      <th>107135</th>
      <td>WB012</td>
      <td>2020-07-01</td>
      <td>8.3800</td>
      <td>36.18</td>
      <td>4.53000</td>
      <td>6.8500</td>
      <td>11.38000</td>
      <td>7.95000</td>
      <td>0.16000</td>
      <td>5.5000</td>
      <td>...</td>
      <td>7.0</td>
      <td>39.0</td>
      <td>2020.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monsoon</td>
      <td>night</td>
      <td>False</td>
    </tr>
    <tr>
      <th>107948</th>
      <td>WB013</td>
      <td>2020-07-01</td>
      <td>10.5000</td>
      <td>36.50</td>
      <td>7.78000</td>
      <td>22.5000</td>
      <td>30.25000</td>
      <td>27.23000</td>
      <td>0.58000</td>
      <td>2.8000</td>
      <td>...</td>
      <td>7.0</td>
      <td>59.0</td>
      <td>2020.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Monsoon</td>
      <td>night</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>107949 rows × 38 columns</p>
</div>




```python
df6['Datetime'].max() - df6['Datetime'].min()
```




    Timedelta('2008 days 00:00:00')



We will also do the splitting before using transformations.


```python
split_tp = df6['Datetime'].max() - datetime.timedelta( days = 150)
```

Training dataset is:


```python
# Train dataset
X_train = df6[df6['Datetime'] < split_tp]
y_train = X_train['AQI']
```

Validation datset is:


```python
# Validation dataset
X_validation = df6[df6['Datetime'] >= split_tp]
y_validation = X_validation['AQI']

print( 'Training Min Date: {}'.format( X_train['Datetime'].min() ) )
print( 'Training Max Date: {}'.format( X_train['Datetime'].max() ) )

print( '\nValidation Min Date: {}'.format( X_validation['Datetime'].min() ) )
print( 'Validation Max Date: {}'.format( X_validation['Datetime'].max() ) )
X_train.head(2)
```

    Training Min Date: 2015-01-01 00:00:00
    Training Max Date: 2020-02-01 00:00:00
    
    Validation Min Date: 2020-02-02 00:00:00
    Validation Max Date: 2020-07-01 00:00:00





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>Datetime</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>...</th>
      <th>Checks</th>
      <th>AQI</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Season</th>
      <th>Part_of_Day</th>
      <th>is_weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AP001</td>
      <td>2017-11-25</td>
      <td>97.083750</td>
      <td>136.788125</td>
      <td>1.756875</td>
      <td>22.628125</td>
      <td>15.565000</td>
      <td>11.985000</td>
      <td>0.198125</td>
      <td>16.628750</td>
      <td>...</td>
      <td>7.0</td>
      <td>196.437500</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>25.0</td>
      <td>15.5</td>
      <td>Post-monsoon</td>
      <td>afternoon</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AP001</td>
      <td>2017-11-26</td>
      <td>78.322917</td>
      <td>129.062500</td>
      <td>1.255417</td>
      <td>25.995833</td>
      <td>14.848333</td>
      <td>10.279167</td>
      <td>0.140417</td>
      <td>26.964583</td>
      <td>...</td>
      <td>7.0</td>
      <td>205.416667</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>11.5</td>
      <td>Post-monsoon</td>
      <td>night</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 38 columns</p>
</div>




```python
# check the proportion of validation data datapoints
X_validation.shape[0]/df6.shape[0] * 100
```




    14.945946697051385




```python
X_validation[['StationId', 'Datetime']].shape[0]/df6[['StationId', 'Datetime']].shape[0]
```




    0.14945946697051385



Now let's normalize the features


```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
#X= df1.drop(['AQI', 'AQI_Bucket'], axis= 1)
#y= df1['AQI']

num_attributes = df6.select_dtypes( include = 'number')
cat_attributes = df6.select_dtypes( include = ['object', 'boolean'])

print(num_attributes.columns)
print(cat_attributes.columns)

```

    Index(['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
           'Benzene', 'Toluene', 'Xylene', 'PM10_24hr_avg', 'PM2.5_24hr_avg',
           'SO2_24hr_avg', 'NOx_24hr_avg', 'NH3_24hr_avg', 'CO_8hr_max',
           'O3_8hr_max', 'PM2.5_SubIndex', 'PM10_SubIndex', 'SO2_SubIndex',
           'NOx_SubIndex', 'NH3_SubIndex', 'CO_SubIndex', 'O3_SubIndex', 'Checks',
           'AQI', 'Year', 'Month', 'Week', 'Day', 'Hour'],
          dtype='object')
    Index(['StationId', 'Season', 'Part_of_Day', 'is_weekend'], dtype='object')



```python
Scaler = StandardScaler()
for att in num_attributes:
    X_train[att] = Scaler.fit_transform( X_train[[att]].values )

```

For categorical attributes:


```python
label_encoder = LabelEncoder()

# Iterate over each categorical attribute
for att in cat_attributes:
    X_train[att] = label_encoder.fit_transform( X_train[[att]].values)         

# is_weekend
#X_train['is_weekend'] = X_train['is_weekend'].apply(lambda x: 1 if x == True else 0)
```


```python
X_train.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>Datetime</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>...</th>
      <th>Checks</th>
      <th>AQI</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Season</th>
      <th>Part_of_Day</th>
      <th>is_weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-11-25</td>
      <td>97.083750</td>
      <td>136.788125</td>
      <td>1.756875</td>
      <td>22.628125</td>
      <td>15.565000</td>
      <td>11.985000</td>
      <td>0.198125</td>
      <td>16.628750</td>
      <td>...</td>
      <td>7.0</td>
      <td>196.437500</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>25.0</td>
      <td>15.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2017-11-26</td>
      <td>78.322917</td>
      <td>129.062500</td>
      <td>1.255417</td>
      <td>25.995833</td>
      <td>14.848333</td>
      <td>10.279167</td>
      <td>0.140417</td>
      <td>26.964583</td>
      <td>...</td>
      <td>7.0</td>
      <td>205.416667</td>
      <td>2017.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>11.5</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 38 columns</p>
</div>




```python
X_validation.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>Datetime</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>...</th>
      <th>day_sin</th>
      <th>day_cos</th>
      <th>month_sin</th>
      <th>month_cos</th>
      <th>week_sin</th>
      <th>week_cos</th>
      <th>Part_of_Day_sin</th>
      <th>Part_of_Day_cos</th>
      <th>season_sin</th>
      <th>season_cos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>799</th>
      <td>1</td>
      <td>2020-02-02</td>
      <td>40.599451</td>
      <td>152.056856</td>
      <td>-32.261781</td>
      <td>0.835245</td>
      <td>4.474435</td>
      <td>7.400284</td>
      <td>-43.424819</td>
      <td>6.113636</td>
      <td>...</td>
      <td>-0.999761</td>
      <td>-0.021868</td>
      <td>-0.744692</td>
      <td>0.667408</td>
      <td>-0.650484</td>
      <td>0.759520</td>
      <td>1.0</td>
      <td>6.123234e-17</td>
      <td>1.0</td>
      <td>6.123234e-17</td>
    </tr>
    <tr>
      <th>800</th>
      <td>1</td>
      <td>2020-02-03</td>
      <td>51.110104</td>
      <td>141.925113</td>
      <td>-32.144813</td>
      <td>2.996683</td>
      <td>-8.846747</td>
      <td>11.780822</td>
      <td>-43.375396</td>
      <td>14.418369</td>
      <td>...</td>
      <td>-0.692229</td>
      <td>0.721677</td>
      <td>-0.744692</td>
      <td>0.667408</td>
      <td>0.010881</td>
      <td>-0.999941</td>
      <td>1.0</td>
      <td>6.123234e-17</td>
      <td>1.0</td>
      <td>6.123234e-17</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 50 columns</p>
</div>



Apply transformation for validation dataset


```python
for att in num_attributes:
    X_validation[att] = Scaler.transform(X_validation[[att]].values)
    
for att in cat_attributes:
     X_validation[att] = label_encoder.transform(X_validation[[att]].values ) 

#X_validation['is_weekend'] = X_validation['is_weekend'].apply(lambda x: 1 if x == True else 0)
```

The target variable after transformation

Since the response variable has a skewed distribution, applying a log transformation can help to achieve uniformity in the data distribution.


```python
#tr = X_train.copy()
#tr['AQI'] = np.log1p( X_train['AQI'] )
X_train['AQI'] = np.log1p( X_train['AQI'] )
#sns.distplot(tr['AQI']);
sns.distplot(X_train['AQI']);
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_145_0.png)
    


Nature transformation


```python
# Hour 
X_train['hour_sin'] = X_train['Hour'].apply( lambda x: np.sin( x * ( 2. * np.pi/24 ) ) )
X_train['hour_cos'] = X_train['Hour'].apply( lambda x: np.cos( x * ( 2. * np.pi/24 ) ) )

# day 
X_train['day_sin'] = X_train['Day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
X_train['day_cos'] = X_train['Day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

# month
X_train['month_sin'] = X_train['Month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
X_train['month_cos'] = X_train['Month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

# day of week
X_train['week_sin'] = X_train['Week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
X_train['week_cos'] = X_train['Week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

# Part of day
X_train['Part_of_Day_sin'] = X_train['Part_of_Day'].apply( lambda x: np.sin( x * ( 2. * np.pi/4 ) ) )
X_train['Part_of_Day_cos'] = X_train['Part_of_Day'].apply( lambda x: np.cos( x * ( 2. * np.pi/4 ) ) )

# season
X_train['season_sin'] = X_train['Season'].apply( lambda x: np.sin( x * ( 2. * np.pi/4 ) ) )
X_train['season_cos'] = X_train['Season'].apply( lambda x: np.cos( x * ( 2. * np.pi/4 ) ) )
```

For validation data


```python
# Hour 
X_validation['hour_sin'] = X_validation['Hour'].apply( lambda x: np.sin( x * ( 2. * np.pi/24 ) ) )
X_validation['hour_cos'] = X_validation['Hour'].apply( lambda x: np.cos( x * ( 2. * np.pi/24 ) ) )

# day 
X_validation['day_sin'] = X_validation['Day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
X_validation['day_cos'] = X_validation['Day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

# month
X_validation['month_sin'] = X_validation['Month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
X_validation['month_cos'] = X_validation['Month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

# day of week
X_validation['week_sin'] = X_validation['Week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
X_validation['week_cos'] = X_validation['Week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

# Part of day
X_validation['Part_of_Day_sin'] = X_validation['Part_of_Day'].apply( lambda x: np.sin( x * ( 2. * np.pi/4 ) ) )
X_validation['Part_of_Day_cos'] = X_validation['Part_of_Day'].apply( lambda x: np.cos( x * ( 2. * np.pi/4 ) ) )

# season
X_validation['season_sin'] = X_validation['Season'].apply( lambda x: np.sin( x * ( 2. * np.pi/4 ) ) )
X_validation['season_cos'] = X_validation['Season'].apply( lambda x: np.cos( x * ( 2. * np.pi/4 ) ) )
```


```python
# target variable
X_validation['AQI'] = np.log1p(X_validation['AQI'] )
```


```python
# new y_train with target variable rescaled
y_validation = X_validation['AQI']

y_train = X_train['AQI'] 
```

## 6. Feature selection

Selecting the most relevant features that describe our dataset involves removing collinear features, which essentially convey the same information. For simplicity, we will use simple models at this stage.


```python
df7 = X_train.copy()
```


```python
# deleting features after feature engineering derivation and transformations. Deleting original variables.
cols_drop = ['Week', 'Day', 'Month', 'Year', 'Season', 'Part_of_Day']
df7 = df7.drop(cols_drop, axis = 1)
```


```python
nan_rows = df7['AQI'].isnull()

# Remove rows with NaN values from both X_train and y_train
df7 = df7[~nan_rows]
```


```python
df7.isna().sum()
```




    StationId          0
    Datetime           0
    PM2.5              0
    PM10               0
    NO                 0
    NO2                0
    NOx                0
    NH3                0
    CO                 0
    SO2                0
    O3                 0
    Benzene            0
    Toluene            0
    Xylene             0
    PM10_24hr_avg      0
    PM2.5_24hr_avg     0
    SO2_24hr_avg       0
    NOx_24hr_avg       0
    NH3_24hr_avg       0
    CO_8hr_max         0
    O3_8hr_max         0
    PM2.5_SubIndex     0
    PM10_SubIndex      0
    SO2_SubIndex       0
    NOx_SubIndex       0
    NH3_SubIndex       0
    CO_SubIndex        0
    O3_SubIndex        0
    Checks             0
    AQI                0
    Hour               0
    is_weekend         0
    hour_sin           0
    hour_cos           0
    day_sin            0
    day_cos            0
    month_sin          0
    month_cos          0
    week_sin           0
    week_cos           0
    Part_of_Day_sin    0
    Part_of_Day_cos    0
    season_sin         0
    season_cos         0
    dtype: int64



#### Best Features from Boruta


```python
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
# creating training and test dataset for Boruta, because it can't be a dataframe type

X_train_n = df7.drop( ['Datetime', 'AQI'], axis=1 ).values
y_train_n = y_train.values.ravel()

# Define RandomForestRegressor
rf = RandomForestRegressor( n_jobs=-1 )

# Define Boruta
boruta = BorutaPy( rf, n_estimators='auto', verbose=2, random_state=42 ).fit( X_train_n, y_train_n )
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	42
    Rejected: 	0
    Iteration: 	2 / 100
    Confirmed: 	0
    Tentative: 	42
    Rejected: 	0
    Iteration: 	3 / 100
    Confirmed: 	0
    Tentative: 	42
    Rejected: 	0
    Iteration: 	4 / 100
    Confirmed: 	0
    Tentative: 	42
    Rejected: 	0
    Iteration: 	5 / 100
    Confirmed: 	0
    Tentative: 	42
    Rejected: 	0
    Iteration: 	6 / 100
    Confirmed: 	0
    Tentative: 	42
    Rejected: 	0
    Iteration: 	7 / 100
    Confirmed: 	0
    Tentative: 	42
    Rejected: 	0
    Iteration: 	8 / 100
    Confirmed: 	24
    Tentative: 	3
    Rejected: 	15
    Iteration: 	9 / 100
    Confirmed: 	24
    Tentative: 	3
    Rejected: 	15
    Iteration: 	10 / 100
    Confirmed: 	24
    Tentative: 	3
    Rejected: 	15
    Iteration: 	11 / 100
    Confirmed: 	24
    Tentative: 	3
    Rejected: 	15
    Iteration: 	12 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	13 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	14 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	15 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	16 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	17 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	18 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	19 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	20 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	21 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	22 / 100
    Confirmed: 	24
    Tentative: 	1
    Rejected: 	17
    Iteration: 	23 / 100
    Confirmed: 	24
    Tentative: 	0
    Rejected: 	18
    
    
    BorutaPy finished running.
    
    Iteration: 	24 / 100
    Confirmed: 	24
    Tentative: 	0
    Rejected: 	18



```python
cols_selected = boruta.support_.tolist()

X_train_fs = df7.drop(['Datetime', 'AQI'], axis = 1)
cols_selected_boruta = X_train_fs.iloc[ :, cols_selected].columns.tolist()

# Not selected boruta features
cols_not_selected_boruta = np.setdiff1d(X_train_fs.columns, cols_selected_boruta)
pd.DataFrame(data = cols_selected_boruta, columns = ['feature_selected'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_selected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PM2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PM10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NO2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NOx</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NH3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CO</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SO2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>O3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Benzene</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Toluene</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Xylene</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PM10_24hr_avg</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PM2.5_24hr_avg</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NOx_24hr_avg</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO_8hr_max</td>
    </tr>
    <tr>
      <th>16</th>
      <td>O3_8hr_max</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PM2.5_SubIndex</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PM10_SubIndex</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NOx_SubIndex</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CO_SubIndex</td>
    </tr>
    <tr>
      <th>21</th>
      <td>O3_SubIndex</td>
    </tr>
    <tr>
      <th>22</th>
      <td>day_sin</td>
    </tr>
    <tr>
      <th>23</th>
      <td>day_cos</td>
    </tr>
  </tbody>
</table>
</div>



#### Best Features from Random Forest


```python
X_train_rf = df7.drop( ['Datetime', 'AQI'], axis=1 ).copy()
y_train_rf = df7['AQI'].copy()

# train random forest classifier
rf = RandomForestRegressor(n_estimators = 200, n_jobs =-1, random_state = 42)
rf.fit(X_train_rf, y_train_rf)

# feature importance data frame
feat_imp = pd.DataFrame({'feature': X_train_rf.columns,
                        'feature_importance': rf.feature_importances_})\
                        .sort_values('feature_importance', ascending=False)\
                        .reset_index(drop=True)
```


```python
# plot feature importance
plt.subplots(figsize=(12,10))
sns.barplot(x='feature_importance', y='feature', data=feat_imp, orient='h', color='royalblue')\
    .set_title('Feature Importance');
```


    
![png](aqi_prediction_allStations_files/aqi_prediction_allStations_163_0.png)
    



```python
feat_imp = feat_imp[:10]
```

The final features are selected combining the two results adding Benzene, Xylene, Toluene to the features selected by Random Forest since the other feature selected based on boruta are already represented by subIndex and averages. 

We can see that PM2.5_SubIndex has high importance than the remaining features. It is a good idea to do the **only based on one feature (PM2.5_SubIndex)** and compare the results. 


```python
feat_imp.set_index('feature')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_importance</th>
    </tr>
    <tr>
      <th>feature</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PM2.5_SubIndex</th>
      <td>0.601193</td>
    </tr>
    <tr>
      <th>PM10_24hr_avg</th>
      <td>0.188328</td>
    </tr>
    <tr>
      <th>CO_8hr_max</th>
      <td>0.098081</td>
    </tr>
    <tr>
      <th>O3_SubIndex</th>
      <td>0.050359</td>
    </tr>
    <tr>
      <th>PM10_SubIndex</th>
      <td>0.032976</td>
    </tr>
    <tr>
      <th>CO_SubIndex</th>
      <td>0.012207</td>
    </tr>
    <tr>
      <th>PM2.5_24hr_avg</th>
      <td>0.004498</td>
    </tr>
    <tr>
      <th>O3_8hr_max</th>
      <td>0.004010</td>
    </tr>
    <tr>
      <th>NOx_24hr_avg</th>
      <td>0.002974</td>
    </tr>
    <tr>
      <th>NOx_SubIndex</th>
      <td>0.002535</td>
    </tr>
  </tbody>
</table>
</div>




```python
# columns to add
feat_to_add = ['Benzene', 'Xylene', 'Toluene', 'StationId', 'Datetime', 'AQI']

# final features
final_features = pd.concat([pd.Series(feat_to_add), feat_imp['feature']])
#final_features.extend( feat_to_add )
print(final_features)
```

    0           Benzene
    1            Xylene
    2           Toluene
    3         StationId
    4          Datetime
    5               AQI
    0    PM2.5_SubIndex
    1     PM10_24hr_avg
    2        CO_8hr_max
    3       O3_SubIndex
    4     PM10_SubIndex
    5       CO_SubIndex
    6    PM2.5_24hr_avg
    7        O3_8hr_max
    8      NOx_24hr_avg
    9      NOx_SubIndex
    dtype: object



```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StationId</th>
      <th>Datetime</th>
      <th>PM2.5</th>
      <th>PM10</th>
      <th>NO</th>
      <th>NO2</th>
      <th>NOx</th>
      <th>NH3</th>
      <th>CO</th>
      <th>SO2</th>
      <th>...</th>
      <th>day_sin</th>
      <th>day_cos</th>
      <th>month_sin</th>
      <th>month_cos</th>
      <th>week_sin</th>
      <th>week_cos</th>
      <th>Part_of_Day_sin</th>
      <th>Part_of_Day_cos</th>
      <th>season_sin</th>
      <th>season_cos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2017-11-25</td>
      <td>97.083750</td>
      <td>136.788125</td>
      <td>1.756875</td>
      <td>22.628125</td>
      <td>15.565000</td>
      <td>11.985000</td>
      <td>0.198125</td>
      <td>16.628750</td>
      <td>...</td>
      <td>-8.660254e-01</td>
      <td>0.500000</td>
      <td>-0.500000</td>
      <td>0.866025</td>
      <td>-0.974928</td>
      <td>-0.222521</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>6.123234e-17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2017-11-26</td>
      <td>78.322917</td>
      <td>129.062500</td>
      <td>1.255417</td>
      <td>25.995833</td>
      <td>14.848333</td>
      <td>10.279167</td>
      <td>0.140417</td>
      <td>26.964583</td>
      <td>...</td>
      <td>-7.431448e-01</td>
      <td>0.669131</td>
      <td>-0.500000</td>
      <td>0.866025</td>
      <td>-0.781831</td>
      <td>0.623490</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.000000e+00</td>
      <td>6.123234e-17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2017-11-27</td>
      <td>111.780417</td>
      <td>154.661250</td>
      <td>9.397500</td>
      <td>33.808750</td>
      <td>41.526667</td>
      <td>14.015833</td>
      <td>0.657917</td>
      <td>25.658750</td>
      <td>...</td>
      <td>-5.877853e-01</td>
      <td>0.809017</td>
      <td>-0.500000</td>
      <td>0.866025</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.000000e+00</td>
      <td>6.123234e-17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2017-11-28</td>
      <td>91.430000</td>
      <td>153.668333</td>
      <td>25.286250</td>
      <td>53.073750</td>
      <td>25.570417</td>
      <td>39.244583</td>
      <td>1.365000</td>
      <td>15.060833</td>
      <td>...</td>
      <td>-4.067366e-01</td>
      <td>0.913545</td>
      <td>-0.500000</td>
      <td>0.866025</td>
      <td>0.781831</td>
      <td>0.623490</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.000000e+00</td>
      <td>6.123234e-17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2017-11-29</td>
      <td>71.177083</td>
      <td>118.970833</td>
      <td>5.228750</td>
      <td>23.205000</td>
      <td>16.586667</td>
      <td>12.965000</td>
      <td>0.162917</td>
      <td>10.565417</td>
      <td>...</td>
      <td>-2.079117e-01</td>
      <td>0.978148</td>
      <td>-0.500000</td>
      <td>0.866025</td>
      <td>0.974928</td>
      <td>-0.222521</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.000000e+00</td>
      <td>6.123234e-17</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107793</th>
      <td>103</td>
      <td>2020-01-28</td>
      <td>133.145000</td>
      <td>231.777083</td>
      <td>88.824583</td>
      <td>90.479167</td>
      <td>126.688750</td>
      <td>28.105417</td>
      <td>0.422500</td>
      <td>10.471667</td>
      <td>...</td>
      <td>-4.067366e-01</td>
      <td>0.913545</td>
      <td>0.500000</td>
      <td>0.866025</td>
      <td>0.781831</td>
      <td>0.623490</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.224647e-16</td>
      <td>-1.000000e+00</td>
    </tr>
    <tr>
      <th>107794</th>
      <td>103</td>
      <td>2020-01-29</td>
      <td>54.907917</td>
      <td>89.169583</td>
      <td>1.746667</td>
      <td>40.463750</td>
      <td>42.165833</td>
      <td>21.469583</td>
      <td>0.365000</td>
      <td>3.821667</td>
      <td>...</td>
      <td>-2.079117e-01</td>
      <td>0.978148</td>
      <td>0.500000</td>
      <td>0.866025</td>
      <td>0.974928</td>
      <td>-0.222521</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.224647e-16</td>
      <td>-1.000000e+00</td>
    </tr>
    <tr>
      <th>107795</th>
      <td>103</td>
      <td>2020-01-30</td>
      <td>60.808750</td>
      <td>107.190833</td>
      <td>32.665833</td>
      <td>47.236250</td>
      <td>79.896250</td>
      <td>17.534167</td>
      <td>0.677500</td>
      <td>4.210417</td>
      <td>...</td>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.866025</td>
      <td>0.433884</td>
      <td>-0.900969</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.224647e-16</td>
      <td>-1.000000e+00</td>
    </tr>
    <tr>
      <th>107796</th>
      <td>103</td>
      <td>2020-01-31</td>
      <td>77.214583</td>
      <td>154.479167</td>
      <td>81.585833</td>
      <td>59.868750</td>
      <td>138.255417</td>
      <td>19.615417</td>
      <td>0.472917</td>
      <td>7.978750</td>
      <td>...</td>
      <td>2.079117e-01</td>
      <td>0.978148</td>
      <td>0.500000</td>
      <td>0.866025</td>
      <td>-0.433884</td>
      <td>-0.900969</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.224647e-16</td>
      <td>-1.000000e+00</td>
    </tr>
    <tr>
      <th>107797</th>
      <td>103</td>
      <td>2020-02-01</td>
      <td>75.169583</td>
      <td>150.655000</td>
      <td>83.388333</td>
      <td>66.632917</td>
      <td>94.422083</td>
      <td>25.176250</td>
      <td>0.459583</td>
      <td>8.729583</td>
      <td>...</td>
      <td>2.079117e-01</td>
      <td>0.978148</td>
      <td>0.866025</td>
      <td>0.500000</td>
      <td>-0.974928</td>
      <td>-0.222521</td>
      <td>-1.0</td>
      <td>-1.836970e-16</td>
      <td>1.224647e-16</td>
      <td>-1.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>91815 rows × 50 columns</p>
</div>




```python
# Applying selected features by boruta on train and validation datasets
x_train = X_train[final_features].copy()
x_validation = X_validation[final_features].copy()

# Time Series Data Preparation for cross-validation
x_training = df7[final_features].copy()
```

## 7. Model Design and Training

Machine learning algorithms predict a single value and cannot be used directly for multi-step forecasting. We can use recursive method to predict multiple point at a time.

We will train several models and select the best performing one for deployment.

##### Five different algorithms are going to be used to predict the target variable:

-  **Average:** averaging model is the model we use most in everyday life, it will always predict the average. It is useful as it is a comparative basis for implementing other models

 - **Logistic Regression:** uses a complex cost function, which can be defined as the Sigmoid function. The output of the classification is based on the probability score between 0 and 1 of the input being in one class or another according to a threshold

 - **Random Forest:** it is a tree based model build with multiple ensamble decision trees created with the bagging method. Then, all the classifiers take a weighted vote on their predictions. Since the algorithm goal is not trying to find a linear function to describe the event, it works for problems with more complex behaviour

 - **XGBoost:** it is also a tree based model but they are built in a different way. While Random Forests builds each tree independently, XGBoost builds one tree at the time learning with its predecessor.
 - LightGBM: is a gradient boosting framework that uses tree based learning algorithms.

Let's define some helper functions and pipeline the training and evaluation.


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso,SGDRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgbm

def mean_absolute_percentage_error( y, yhat ):
    y, yhat = np.array(y), np.array(yhat)
    return np.mean( np.abs( ( y-yhat ) / y ))

def mean_percentage_error( y, yhat ):
    return np.mean( ( y - yhat ) / y )

# Define the function to evaluate the models
def weighted_mean_absolute_error(df, y, yhat):
    weights = 1
    return np.round(np.sum(weights*abs(y-yhat))/(np.sum(weights)), 2)

def ml_error( df, model_name, y, yhat):
    mae = mean_absolute_error( y,yhat )
    mape = mean_absolute_percentage_error( y,yhat )
    rmse = np.sqrt(mean_squared_error( y,yhat ))
    WMAE = weighted_mean_absolute_error(df, y, yhat)
    
    return pd.DataFrame( {'Model Name': model_name,
                          'MAE': mae,
                          'RMSE': rmse,
                          'WMAE': WMAE}, index=[0])

# time-series cross validation implementation
def cross_validation( x_training, kfold, model_name, model, sp, verbose=False ):
    mae_list = []
    mape_list = []
    rmse_list = []
    WMAE_list = []
     
    for k in reversed( range( 1, kfold+1 ) ): #k-fold implementation
        if verbose:
            print( '\nKFold Number: {}'.format( k ) )
        # start and end date for validation 
        start_date_validation = x_training['date'].max() - datetime.timedelta( days=k*sp) 
        end_date_validation = x_training['date'].max() - datetime.timedelta( days=(k-1)*sp)

        # filtering dataset
        training = x_training[x_training['date'] < start_date_validation]
        validation = x_training[(x_training['date'] >= start_date_validation) 
                                & (x_training['date'] <= end_date_validation)]

        # training and validation dataset
        # training
        xtraining = training.drop( ['Datetime', 'AQI'], axis=1 ) 
        ytraining = training['AQI']

        # validation
        xvalidation = validation.drop( ['Datetime', 'AQI'], axis=1 )
        yvalidation = validation['AQI']

        # model
        m = model.fit( xtraining, ytraining )

        # prediction
        yhat = m.predict(xvalidation)

        # performance
        m_result = ml_error( xvalidation, model_name, np.expm1( yvalidation ), np.expm1( yhat ) )

        # store performance of each kfold iteration
        mae_list.append(  m_result['MAE'] )
        rmse_list.append( m_result['RMSE'] )
        WMAE_list.append( m_result['WMAE'])

    return pd.DataFrame( {'Model Name': model_name,
                          'MAE CV':  np.round( np.mean( mae_list ), 2 ).astype( str )  
                          + ' +/- ' + np.round( np.std( mae_list ), 2 ).astype( str ),
                          'RMSE CV': np.round( np.mean( rmse_list ), 2 ).astype( str ) 
                          + ' +/- ' + np.round( np.std( rmse_list ), 2 ).astype( str ),
                          'WMAE CV': np.round( np.mean( WMAE_list ), 2 ).astype( str ) 
                          + ' +/- ' + np.round( np.std( WMAE_list ), 2 ).astype( str )}, index=[0] )




# prepare a list of ml models
def get_models(models=dict()):
    # linear models
    models['lr'] = LinearRegression()
    models['lasso'] = Lasso()
    models['rf'] = RandomForestRegressor( n_estimators = 100, n_jobs =-1, random_state=7 )
    models['xgb'] = xgb.XGBRegressor( objective='reg:squarederror', n_estimators = 100, random_state=7)
    models['lgbm'] = lgbm.LGBMRegressor(n_estimators = 100, n_jobs =-1, random_state=7)
    print('Defined %d models' % len(models))
    return models
```

#### Average model


```python
aux1 = x_validation.copy()
aux1['AQI'] = y_validation.copy()
```


```python
aux1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Benzene</th>
      <th>Xylene</th>
      <th>Toluene</th>
      <th>StationId</th>
      <th>Datetime</th>
      <th>AQI</th>
      <th>PM2.5_SubIndex</th>
      <th>PM10_24hr_avg</th>
      <th>CO_8hr_max</th>
      <th>O3_SubIndex</th>
      <th>PM10_SubIndex</th>
      <th>CO_SubIndex</th>
      <th>PM2.5_24hr_avg</th>
      <th>O3_8hr_max</th>
      <th>NOx_24hr_avg</th>
      <th>NOx_SubIndex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>799</th>
      <td>-219.148466</td>
      <td>-215.989310</td>
      <td>-193.184765</td>
      <td>1</td>
      <td>2020-02-02</td>
      <td>6.277460</td>
      <td>293.016268</td>
      <td>446.240297</td>
      <td>-214.269687</td>
      <td>470.146747</td>
      <td>446.240297</td>
      <td>334.068556</td>
      <td>85.625656</td>
      <td>470.146747</td>
      <td>-97.955761</td>
      <td>-66.079635</td>
    </tr>
    <tr>
      <th>800</th>
      <td>-219.910571</td>
      <td>-223.004590</td>
      <td>-210.563379</td>
      <td>1</td>
      <td>2020-02-03</td>
      <td>6.471456</td>
      <td>410.939536</td>
      <td>590.593902</td>
      <td>-212.680339</td>
      <td>507.691847</td>
      <td>590.593902</td>
      <td>413.535979</td>
      <td>156.379616</td>
      <td>507.691847</td>
      <td>-26.061899</td>
      <td>23.787693</td>
    </tr>
    <tr>
      <th>801</th>
      <td>-219.760756</td>
      <td>-210.719708</td>
      <td>-212.608688</td>
      <td>1</td>
      <td>2020-02-04</td>
      <td>6.430315</td>
      <td>474.685817</td>
      <td>548.356478</td>
      <td>-215.012252</td>
      <td>481.923558</td>
      <td>548.356478</td>
      <td>296.940334</td>
      <td>194.627385</td>
      <td>481.923558</td>
      <td>-62.342794</td>
      <td>-21.563426</td>
    </tr>
    <tr>
      <th>802</th>
      <td>-220.242771</td>
      <td>-211.957315</td>
      <td>-215.051334</td>
      <td>1</td>
      <td>2020-02-05</td>
      <td>6.418699</td>
      <td>467.165633</td>
      <td>517.965888</td>
      <td>-215.168581</td>
      <td>534.372057</td>
      <td>517.965888</td>
      <td>289.123866</td>
      <td>190.115274</td>
      <td>534.372057</td>
      <td>-92.654676</td>
      <td>-59.453279</td>
    </tr>
    <tr>
      <th>803</th>
      <td>-219.083329</td>
      <td>-220.900657</td>
      <td>-207.593121</td>
      <td>1</td>
      <td>2020-02-06</td>
      <td>6.565630</td>
      <td>460.911102</td>
      <td>582.010714</td>
      <td>-209.827328</td>
      <td>540.247436</td>
      <td>582.010714</td>
      <td>556.186516</td>
      <td>186.362556</td>
      <td>540.247436</td>
      <td>-26.048328</td>
      <td>23.804655</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107944</th>
      <td>-204.785706</td>
      <td>-218.412415</td>
      <td>-112.043316</td>
      <td>1</td>
      <td>2020-06-27</td>
      <td>6.318213</td>
      <td>60.004559</td>
      <td>68.485336</td>
      <td>-212.869237</td>
      <td>503.835723</td>
      <td>68.485336</td>
      <td>404.091080</td>
      <td>-54.181370</td>
      <td>503.835723</td>
      <td>64.822452</td>
      <td>137.393131</td>
    </tr>
    <tr>
      <th>107945</th>
      <td>-203.313605</td>
      <td>-215.793898</td>
      <td>-101.653927</td>
      <td>1</td>
      <td>2020-06-28</td>
      <td>6.679901</td>
      <td>88.832307</td>
      <td>60.111673</td>
      <td>-213.318684</td>
      <td>766.267116</td>
      <td>60.111673</td>
      <td>381.618735</td>
      <td>-36.884721</td>
      <td>766.267116</td>
      <td>92.089983</td>
      <td>171.477545</td>
    </tr>
    <tr>
      <th>107946</th>
      <td>-170.425817</td>
      <td>-166.771618</td>
      <td>-90.443810</td>
      <td>1</td>
      <td>2020-06-29</td>
      <td>6.634720</td>
      <td>151.578912</td>
      <td>169.121545</td>
      <td>-212.322084</td>
      <td>758.151017</td>
      <td>169.121545</td>
      <td>431.448718</td>
      <td>0.763242</td>
      <td>758.151017</td>
      <td>253.759506</td>
      <td>373.564449</td>
    </tr>
    <tr>
      <th>107947</th>
      <td>-196.448141</td>
      <td>-202.525444</td>
      <td>-94.156632</td>
      <td>1</td>
      <td>2020-06-30</td>
      <td>6.505143</td>
      <td>307.410239</td>
      <td>450.770049</td>
      <td>-211.722821</td>
      <td>547.490696</td>
      <td>450.770049</td>
      <td>461.411844</td>
      <td>94.262038</td>
      <td>547.490696</td>
      <td>322.456216</td>
      <td>459.435336</td>
    </tr>
    <tr>
      <th>107948</th>
      <td>-204.981118</td>
      <td>-160.114593</td>
      <td>-109.932870</td>
      <td>1</td>
      <td>2020-07-01</td>
      <td>6.548051</td>
      <td>180.626951</td>
      <td>377.886372</td>
      <td>-212.797586</td>
      <td>696.257619</td>
      <td>377.886372</td>
      <td>407.673628</td>
      <td>18.192065</td>
      <td>696.257619</td>
      <td>249.930794</td>
      <td>368.778559</td>
    </tr>
  </tbody>
</table>
<p>16134 rows × 16 columns</p>
</div>




```python
# Predictions
aux2 = aux1[['StationId', 'AQI']].groupby('StationId').mean().reset_index().rename(columns = {'AQI': 'predictions'})
aux1 = pd.merge( aux1, aux2, how= 'left', on='StationId')
yhat_baseline = aux1['predictions']
```


```python
# Performance
baseline_result = ml_error( aux1 ,'Average Model', np.expm1( y_validation ), np.expm1( yhat_baseline ))
baseline_result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model Name</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>WMAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Average Model</td>
      <td>729.445976</td>
      <td>1095.910434</td>
      <td>2226742.65</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_validation = x_validation.drop(['Datetime'], axis=1)
```


```python
x_train = x_train.drop(['Datetime'], axis=1)
```


```python
# Prepare, train and evaluate the ML models
models = get_models()

trained_models ={}
result = {}
for name, model in models.items():
    trained_models[name] = model.fit(x_train, y_train)
    
    # Prediction 
    yhat = model.predict(x_validation)
    
    #Evaluation
    result[name]= ml_error( x_validation, name, np.expm1(y_validation), np.expm1(yhat))

```

    Defined 5 models



```python
results=baseline_result.copy()
```


```python
for name, model in models.items():
    results = pd.concat([results, result[name]])
```


```python
results.set_index('Model Name').sort_values('RMSE')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>WMAE</th>
    </tr>
    <tr>
      <th>Model Name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lr</th>
      <td>3.581889e-10</td>
      <td>5.860661e-10</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>xgb</th>
      <td>5.508898e+02</td>
      <td>1.060549e+03</td>
      <td>8.888056e+06</td>
    </tr>
    <tr>
      <th>Average Model</th>
      <td>7.294460e+02</td>
      <td>1.095910e+03</td>
      <td>2.226743e+06</td>
    </tr>
    <tr>
      <th>rf</th>
      <td>7.473813e+02</td>
      <td>1.225239e+03</td>
      <td>1.205825e+07</td>
    </tr>
    <tr>
      <th>lgbm</th>
      <td>1.069358e+03</td>
      <td>1.495407e+03</td>
      <td>1.725301e+07</td>
    </tr>
    <tr>
      <th>lasso</th>
      <td>4.530090e+15</td>
      <td>4.654170e+17</td>
      <td>7.308848e+19</td>
    </tr>
  </tbody>
</table>
</div>




```python
import joblib
joblib.dump(model, "model/best_model.pkl")
```

## 8. Deployment

# Things To-Do
- **Check the training result and improve performance**
- **Tune best model**
- **Test deployed model**
- **Write documentation**



```python

```
