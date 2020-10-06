import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('ascombe.txt', sep = "\t")
x1 = data.x1.values
y1 = data.y1.values
mdl1 = LinearRegression().fit(x1.reshape(-1,1), y1)
x2 = data.x2.values
y2 = data.y2.values
mdl2 = LinearRegression().fit(x2.reshape(-1,1), y2)
x3 = data.x3.values
y3 = data.y3.values
mdl3 = LinearRegression().fit(x3.reshape(-1,1), y3)
x4 = data.x4.values
y4 = data.y4.values
mdl4 = LinearRegression().fit(x4.reshape(-1,1), y4)
plt.figure()
plt.subplot(2, 2, 1)
plt.scatter(x1, y1)
plt.plot(x1, mdl1.predict(x1.reshape(-1,1)))
plt.subplot(2, 2, 2)
plt.scatter(x2, y2)
plt.plot(x2, mdl2.predict(x2.reshape(-1,1)))
plt.subplot(2, 2, 3)
plt.scatter(x3, y3)
plt.plot(x3, mdl3.predict(x3.reshape(-1,1)))
plt.subplot(2, 2, 4)
plt.scatter(x4, y4)
plt.plot(x4, mdl4.predict(x4.reshape(-1,1)))
