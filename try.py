from collections import defaultdict

import numpy as np


q_table = defaultdict(lambda: np.zeros(6))

print(q_table.items())

q_table[0][0] = 1

print(q_table.items())
print(q_table[0][0])