import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('T-UK.csv', header=None, delimiter=',')
avgData = pd.read_csv('T-UK_avg.csv', header=None, delimiter=',')
year = data[0]
avgYear = avgData[0]
tennessee = data[2]
avgTennessee = avgData[1]
kentucky = data[4]
avgKentucky = avgData[2]


plt.plot(avgYear, avgTennessee, c='orange', linewidth = 3, label='10-Year Avg. Tennessee')
plt.plot(avgYear, avgKentucky, c='blue', linewidth = 3, label='10-Year Avg. Kentucky')
plt.scatter(year, tennessee, c='orange', label='Tennessee')
plt.scatter(year, kentucky, c='blue', label='Kentucky')
plt.title('Tennessee vs. Kentucky 1893-2017 (Football)')
plt.ylabel('Score')
plt.xlabel('Year')
plt.ylim([0, 70])
plt.legend()
plt.savefig('plot.png')
