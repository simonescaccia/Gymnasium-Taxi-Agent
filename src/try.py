import numpy as np
from sklearn.preprocessing import MinMaxScaler


scale_x = MinMaxScaler()
# create a random array with values between 0 and 499
x = np.array([[0, 499]]).reshape(-1, 1)
# fit the scaler
scale_x = scale_x.fit(x)


# scale the data
y = np.array([[0]])
y = scale_x.transform(y)
print(y)
