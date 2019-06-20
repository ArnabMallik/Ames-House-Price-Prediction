# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:40:53 2018

@author: Ananda Mohon Ghosh
"""
#Finall Comparison
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)
memory = [41163.23764482, 47765.54137982, 44671.54111611, 31930.83344, 52250.7601211,41760.2730680]


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.4f' % (x)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(formatter)
plt.bar(x, memory)
plt.xticks(x, ('LR', 'KNN', 'GBRT', 'RF', 'SVM', 'NN'))
plt.title("Cost in different Models")
plt.xlabel("Models")
plt.ylabel("Cost / Loss (RMSE)")
plt.show()




#KNN Cost Comparison
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(5)
memory = [54552, 52269, 48229, 47765, 49136 ]


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.4f' % (x)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(formatter)
plt.bar(x, memory)
plt.xticks(x, ('K=1', 'K=5', 'K=10', 'K=13', 'K=15'))
plt.title("Cost for different value of K in KNN Model")
plt.xlabel("Models")
plt.ylabel("Cost / Loss (RMSE)")
plt.show()



#SVM Cost Comparison
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)
memory = [58873.3935, 58761.3016, 57827.734, 54317.3880, 49071.8554, 42906.938096 ]


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.4f' % (x)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(formatter)
plt.bar(x, memory)
plt.xticks(x, ('C=1', 'C=10', 'C=100', 'C=1000', 'C=10000', 'C=100000'))
plt.title("Cost for different value of C in SVM Model on RBF Kernel")
plt.xlabel("Models")
plt.ylabel("Cost / Loss (RMSE)")
plt.show()




#SVM Cost Comparison
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)
memory = [58873.3935, 58761.3016, 57827.734, 54317.3880, 49071.8554, 42906.938096 ]


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.4f' % (x)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(formatter)
plt.bar(x, memory)
plt.xticks(x, ('C=1', 'C=10', 'C=100', 'C=1000', 'C=10000', 'C=100000'))
plt.title("Cost for different value of C in SVM Model on RBF Kernel")
plt.xlabel("Models")
plt.ylabel("Cost / Loss (RMSE)")
plt.show()



#GBRT Cost Comparison
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(6)
memory = [36462.825,
  33601.0349,
  32642.973,
  32460.546,
  32467.199,
  32428.327]


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.4f' % (x)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(formatter)
plt.bar(x, memory)
plt.xticks(x, ('D=2', 'D=5', 'D=10', 'D=15', 'D=25', 'D=35'))
plt.title("Cost for different value of Depth in RF Model on 250 Trees")
plt.xlabel("Models")
plt.ylabel("Cost / Loss (RMSE)")
plt.show()

