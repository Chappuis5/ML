import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

name_file = './data_lab1.txt' 

columns = ['x','y']
data_in = pd.read_csv(name_file, 
                      names=columns,
                      sep=' ')
                      
#data_in.plot(kind='scatter',x='x',y='y',color='red')

#x = np.asarray(data_in['x'])
#y = np.asarray(data_in['y'])

#plt.figure(5)
#plt.plot(x,y,'ro')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show() #plot the output as a function of the input data

#Split between training and testing data
index = int(len(data_in) * 0.7)
training_data = data_in[:index]
testing_data = data_in[index:]
x_train = training_data['x'].values
y_train = training_data['y'].values
x_test = testing_data['x'].values
y_test = testing_data['y'].values

#3) Using training data, let's fit the univariate linear regression parameters to the dataset using batch gradient descent (BGD).

#We define the model function

def linear_regression(theta_0,theta_1, x):
    return theta_0 + theta_1 * x

#We define the cost function using the mean squared error

def mean_cost(theta_0, theta_1, x, y):
    m = len(y)
    h = linear_regression(theta_0, theta_1 ,x) #hypothesis function
    return (1/(2*m)) * np.sum((h-y)**2)

learning_rate = 0.99
num_iterations = 1000

theta_0 = 0
theta_1 = 0

# We implement BGD

m = len(y_train)
for i in range(num_iterations):
    h = linear_regression(theta_0, theta_1 ,x_train)
    grad_0 = 1/m * np.sum((h-y_train))
    grad_1 = 1/m * np.sum((h-y_train)*x_train)
    theta_0 = theta_0 - learning_rate * grad_0 #update rule
    theta_1 = theta_1 - learning_rate * grad_1
    cost = mean_cost(theta_0, theta_1, x_train, y_train)
    print(f'Iteration {i+1}, cost={cost:.2f}, theta0={theta_0:.2f}, theta1={theta_1:.2f}')

#We plot the results
print(f'Final parameters: theta0={theta_0:.2f}, theta1={theta_1:.2f}')

#Let's test with the test data the trained model
y_pred = linear_regression(theta_0, theta_1, x_test)
for i in range(len(y_test)):
    print(f'x={x_test[i]:.2f}, y={y_test[i]:.2f}, y_pred={y_pred[i]:.2f}')
mse = np.mean((y_pred - y_test) ** 2)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
mae = np.mean(np.abs(y_pred - y_test))

print(f'MSE: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
print(f'MAE: {mae:.2f}')


# Step 7: Plot the linear regressors for training and test data
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_test, y_test, color='green', label='Test data')
plt.plot(x_train, linear_regression(theta_0, theta_1, x_train), color='red', label='Linear regression (training)')
plt.plot(x_test, linear_regression(theta_0, theta_1, x_test), color='orange', label='Linear regression (test)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
