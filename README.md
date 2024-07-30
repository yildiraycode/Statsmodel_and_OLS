# statsmodel
* hangi özniteliklerin model için %95 güvenilirlikle ne kadar anlamlı olup olmadığı
* **veriyi kullanarak bir OLS (Ordinary Least Squares) regresyon modeli oluşturur ve özniteliklerin anlamlılığını test eder.**


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

#### California Housing veri setini yükleyelim


```python
housing = fetch_california_housing()
```


```python
X = pd.DataFrame(housing.data, columns=housing.feature_names)
```


```python
X
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
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
    </tr>
    <tr>
      <th>20635</th>
      <td>1.5603</td>
      <td>25.0</td>
      <td>5.045455</td>
      <td>1.133333</td>
      <td>845.0</td>
      <td>2.560606</td>
      <td>39.48</td>
      <td>-121.09</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>2.5568</td>
      <td>18.0</td>
      <td>6.114035</td>
      <td>1.315789</td>
      <td>356.0</td>
      <td>3.122807</td>
      <td>39.49</td>
      <td>-121.21</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>1.7000</td>
      <td>17.0</td>
      <td>5.205543</td>
      <td>1.120092</td>
      <td>1007.0</td>
      <td>2.325635</td>
      <td>39.43</td>
      <td>-121.22</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>1.8672</td>
      <td>18.0</td>
      <td>5.329513</td>
      <td>1.171920</td>
      <td>741.0</td>
      <td>2.123209</td>
      <td>39.43</td>
      <td>-121.32</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>2.3886</td>
      <td>16.0</td>
      <td>5.254717</td>
      <td>1.162264</td>
      <td>1387.0</td>
      <td>2.616981</td>
      <td>39.37</td>
      <td>-121.24</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 8 columns</p>
</div>




```python
y = pd.Series(housing.target, name='MedHouseVal')
```


```python
y
```




    0        4.526
    1        3.585
    2        3.521
    3        3.413
    4        3.422
             ...  
    20635    0.781
    20636    0.771
    20637    0.923
    20638    0.847
    20639    0.894
    Name: MedHouseVal, Length: 20640, dtype: float64



#### Eğitim ve test verilerine bölelim


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Statsmodels ile OLS (Ordinary Least Squares) regresyonu


```python
X_train_const = sm.add_constant(X_train)  # Sabit terim ekliyoruz
```


```python
model = sm.OLS(y_train, X_train_const).fit()

```


```python
# Model özetini yazdırma
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            MedHouseVal   R-squared:                       0.613
    Model:                            OLS   Adj. R-squared:                  0.612
    Method:                 Least Squares   F-statistic:                     3261.
    Date:                Tue, 30 Jul 2024   Prob (F-statistic):               0.00
    Time:                        16:49:55   Log-Likelihood:                -17998.
    No. Observations:               16512   AIC:                         3.601e+04
    Df Residuals:                   16503   BIC:                         3.608e+04
    Df Model:                           8                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -37.0233      0.728    -50.835      0.000     -38.451     -35.596
    MedInc         0.4487      0.005     95.697      0.000       0.439       0.458
    HouseAge       0.0097      0.000     19.665      0.000       0.009       0.011
    AveRooms      -0.1233      0.007    -18.677      0.000      -0.136      -0.110
    AveBedrms      0.7831      0.033     23.556      0.000       0.718       0.848
    Population  -2.03e-06   5.25e-06     -0.387      0.699   -1.23e-05    8.26e-06
    AveOccup      -0.0035      0.000     -7.253      0.000      -0.004      -0.003
    Latitude      -0.4198      0.008    -52.767      0.000      -0.435      -0.404
    Longitude     -0.4337      0.008    -52.117      0.000      -0.450      -0.417
    ==============================================================================
    Omnibus:                     3333.187   Durbin-Watson:                   1.962
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             9371.466
    Skew:                           1.071   Prob(JB):                         0.00
    Kurtosis:                       6.006   Cond. No.                     2.38e+05
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.38e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.
    


```python
# p-değeri 0.05'ten küçük olan öznitelikleri seçiyoruz
significant_features = model.pvalues[model.pvalues < 0.05].index.tolist()
significant_features.remove('const')  # Sabit terimi listeden çıkarıyoruz
```

#### Anlamlı özniteliklerle yeni bir model oluşturuyoruz


```python
X_train_sig = X_train[significant_features]
X_train_sig_const = sm.add_constant(X_train_sig)
model_sig = sm.OLS(y_train, X_train_sig_const).fit()
```


```python
# Yeni model özetini yazdırma
print(model_sig.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            MedHouseVal   R-squared:                       0.613
    Model:                            OLS   Adj. R-squared:                  0.612
    Method:                 Least Squares   F-statistic:                     3727.
    Date:                Tue, 30 Jul 2024   Prob (F-statistic):               0.00
    Time:                        16:55:42   Log-Likelihood:                -17998.
    No. Observations:               16512   AIC:                         3.601e+04
    Df Residuals:                   16504   BIC:                         3.607e+04
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -37.0114      0.728    -50.865      0.000     -38.438     -35.585
    MedInc         0.4487      0.005     95.743      0.000       0.440       0.458
    HouseAge       0.0098      0.000     20.864      0.000       0.009       0.011
    AveRooms      -0.1233      0.007    -18.674      0.000      -0.136      -0.110
    AveBedrms      0.7832      0.033     23.560      0.000       0.718       0.848
    AveOccup      -0.0035      0.000     -7.312      0.000      -0.004      -0.003
    Latitude      -0.4195      0.008    -52.993      0.000      -0.435      -0.404
    Longitude     -0.4335      0.008    -52.229      0.000      -0.450      -0.417
    ==============================================================================
    Omnibus:                     3338.428   Durbin-Watson:                   1.962
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             9390.846
    Skew:                           1.072   Prob(JB):                         0.00
    Kurtosis:                       6.009   Cond. No.                     1.67e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.67e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    

## Yeni Modeli Test Etme ve Karşılaştırma

#### test verisi üzerinde tahminler


```python
X_test_const = sm.add_constant(X_test)
y_pred = model.predict(X_test_const)

X_test_sig = X_test[significant_features]
X_test_sig_const = sm.add_constant(X_test_sig)
y_pred_sig = model_sig.predict(X_test_sig_const)
```

#### Performans karşılaştırması (Mean Squared Error)


```python
mse_original = mean_squared_error(y_test, y_pred)
mse_significant = mean_squared_error(y_test, y_pred_sig)

print(f'Original Model MSE: {mse_original}')
print(f'Significant Features Model MSE: {mse_significant}')
```

    Original Model MSE: 0.5558915986952462
    Significant Features Model MSE: 0.5559531347340669
    

* Model performansını karşılaştırmak için ortalama kare hatası (MSE) kullanılır. MSE, modelin tahminlerinin gerçek değerlerden ne kadar sapma gösterdiğini ölçer; daha düşük bir MSE değeri, daha iyi bir model performansını gösterir.
* Her iki modelin de MSE değerleri birbirine çok yakın olduğundan, performans açısından büyük bir fark yoktur. Orijinal model biraz daha iyi performans gösteriyor olsa da, anlamlı özniteliklerle oluşturulan modelin avantajları göz önünde bulundurularak (örneğin, modelin daha basit ve yorumlanabilir olması) tercih edilebilir. MSE'nin çok yakın olması, anlamlı özniteliklerle oluşturulan modelin performans kaybı olmadan daha sade bir model sunduğunu gösterir.


```python

```
