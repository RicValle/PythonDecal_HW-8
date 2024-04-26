
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.optimize import curve_fit 

data = pd.read_csv("GlobalLandTemperaturesByState.csv")

Filtered_Data = data[['dt', 'AverageTemperature', 'State']]

Filtered_Data['dt'] = pd.to_datetime(Filtered_Data['dt'])

Filtered_Data = Filtered_Data[Filtered_Data['dt'].dt.year > 2000]

Filtered_Data = Filtered_Data[Filtered_Data['State'].isin(['Wyoming', 'Nebraska', 'South Dakota'])]

Average_Temperature = Filtered_Data.groupby('dt')['AverageTemperature'].mean().reset_index()

#plt.figure(figsize = (10, 6))
#plt.plot(Average_Temperature['dt'], Average_Temperature['AverageTemperature'], color = 'blue')
#plt.xlabel('Date')
#plt.ylabel('Average Temperature')
#plt.title('Average Temperature over all three States')
#plt.xticks(rotation = 45)
#plt.show()

Filtered_Data['Year'] = Filtered_Data['dt'].dt.year

Filtered_Data['Month'] = Filtered_Data['dt'].dt.month

Filtered_Data['Day'] = Filtered_Data['dt'].dt.day

Initial_Guess = np.array([10, 1 / 12, 0, Filtered_Data['AverageTemperature'].mean()])

def sinusiodal_model(x, A, f, phi, C):

    return A * np.sin(2 * np.pi * f * x + phi) + C

X_Data = Filtered_Data['dt'].astype('int64') // 10**9  
Y_Data = Filtered_Data['AverageTemperature']

params , cov_matrix = curve_fit(sinusiodal_model, X_Data, Y_Data, p0 = Initial_Guess)

plt.figure(figsize = (10, 6))
plt.plot(Filtered_Data['dt'], Filtered_Data['AverageTemperature'], label = 'Original Data', color = 'blue')
plt.plot(Filtered_Data['dt'], sinusiodal_model(X_Data, *params), label = 'Curve Fit', color = 'red')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.title('Average Temperature over all three States with fit line')
plt.xticks(rotation = 45)
plt.savefig('Curve_Fitting_Problem-1')
plt.legend()
plt.show()

parameter_errors = np.sqrt(np.diag(cov_matrix))

for i, (parameter, error) in enumerate(zip(params, parameter_errors), 1):
    
    print(f"Parameter {i}: {parameter:.2f} +/- {error:.2f}")

print(f"Final Equation: y(t) = {params[0]:.2f} * sin(2*pi*{params[1]:.2f}*t + {params[2]:.2f}) + {params[3]:.2f}")
