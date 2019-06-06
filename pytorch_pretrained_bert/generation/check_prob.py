import random
import numpy as np

from tqdm import tqdm

n = 100000
p = float(1/3)
probs = [p, 1-p]
results = []

for i in tqdm(range(n)):
    result = []
    while np.random.choice([0,1], size=1, p=probs) == 1:
        result.append("dummy")
    result.append("end")
    results.append(len(result)-1)

print(np.mean(results))
