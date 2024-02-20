#%%
import numpy as np

def is_lipshitz(n):
    sum1 = lambda n: (n**2)/2 - (n/2)
    lipshitz_sum = lambda n: np.sum([np.ceil(np.sqrt(n))*m % n for m in range(1, n)])
    return sum1(n) == lipshitz_sum(n)

lipshitz_numbers = np.array([n for n in range(1000) if is_lipshitz(n)])
# %%

def sum_for_primeness(n):
    m_values = np.arange(1, n)
    k_values = np.arange(1, n)

    M = np.mod(np.outer(m_values, k_values), n)

    v = np.sum(M, axis=0)
    return M.T
# %%
sum_for