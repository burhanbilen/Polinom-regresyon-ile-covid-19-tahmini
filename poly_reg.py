import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv("covid_saglikbakanligi.csv")

imp = SimpleImputer(missing_values=np.nan, strategy='median')

X = [i for i in range(1, len(df)+1)]
X = np.array(X).reshape(len(X),1)

gunluk_hasta_sayisi = df.iloc[:,0].values.reshape(len(X), 1)
gunluk_test_sayisi = df.iloc[:,1].values.reshape(len(X), 1)
gunluk_vefat_sayisi = df.iloc[:,2].values.reshape(len(X), 1)
gunluk_iyilesen_sayisi = df.iloc[:,3].values.reshape(len(X), 1)
gunluk_iyilesen_sayisi = imp.fit_transform(gunluk_iyilesen_sayisi)

pf = PolynomialFeatures(degree = 11) 
X_Polynom = pf.fit_transform(X)

polynom = LinearRegression()
lr = LinearRegression()



polynom.fit(X_Polynom, gunluk_hasta_sayisi)
p = polynom.predict(X_Polynom)
lr.fit(X, gunluk_hasta_sayisi)
plt.plot(X, lr.predict(X), color="limegreen")

plt.scatter(X, gunluk_hasta_sayisi, marker = '.', s = 15, color = "grey")
plt.plot(X, p, color = "black")
plt.title('Günlere Göre Hastalık Sayısı')
plt.xlabel('Günler')
plt.ylabel('Hasta Sayısı')
plt.show()



polynom.fit(X_Polynom, gunluk_test_sayisi)
p = polynom.predict(X_Polynom)
lr.fit(X, gunluk_test_sayisi)
plt.plot(X, lr.predict(X), color="limegreen")

plt.scatter(X, gunluk_test_sayisi, marker='o', s=15, color = "grey")
plt.plot(X, p,  color="black") 
plt.title('Günlere Göre Test Sayısı')
plt.xlabel('Günler')
plt.ylabel('Test Sayısı')
plt.show()



polynom.fit(X_Polynom, gunluk_vefat_sayisi)
p = polynom.predict(X_Polynom)
lr.fit(X, gunluk_vefat_sayisi)
plt.plot(X, lr.predict(X), color="limegreen") 

plt.scatter(X, gunluk_vefat_sayisi, marker='o', s=15, color = "grey")
plt.plot(X, p,  color="black") 
plt.title('Günlere Göre Vefat Sayısı')
plt.xlabel('Günler')
plt.ylabel('Vefat Sayısı')
plt.show()



polynom.fit(X_Polynom, gunluk_iyilesen_sayisi)
p = polynom.predict(X_Polynom)
lr.fit(X, gunluk_iyilesen_sayisi)
plt.plot(X, lr.predict(X), color="limegreen") 

plt.scatter(X, gunluk_iyilesen_sayisi, marker='o', s=15, color = "grey")
plt.plot(X, p,  color="black") 
plt.title('Günlere Göre İyileşen Sayısı')
plt.xlabel('Günler')
plt.ylabel('İyileşen Sayısı')
plt.show()
