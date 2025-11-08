from offloading import *
from plot_figure import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")




print('========================= running 1 =========================')
# ============= 
# Parameters
# ============= 
M = 5
K = 2
T = 20000
T_strict = 100000
mu = np.array([[0.3, 0.5],[0.5, 0.7], [0.6, 0.9] ,[0.1, 0.5], [0.2, 0.3]])
reliability = [0.99,0.97,0.95,0.96,0.94]
reliability_strict = [0.99,0.97,0.95,0.9,0.98]
deadline = 0.2
delay = [[0.015, 0.01],[0.019, 0.015], [0.021, 0.019] ,[0.03, 0.01], [0.015, 0.01]]
true_value = np.zeros((M,K,M))
print('========================= True Reliability =========================')
for i in range(M):
    for j in range(K):
        for n in range(M):
            true_value[i,j,n] = 1 - np.exp(-deadline/(delay[i][j]*(n+1)))
print(true_value)
test_times = 50
sym = "soft"
plot_figure(test_times, delay, K, M, T, mu, reliability, deadline, sym)

print('========================= running 2 =========================')
test_times_strict = 20
sym = "strict"
plot_figure(test_times_strict, delay, K, M, T_strict, mu, reliability_strict, deadline, sym)
