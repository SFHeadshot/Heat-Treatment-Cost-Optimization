import math
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Functions
def calculate_D(Do, Q, R, T):   # Function to calculate D
    D = Do * math.exp(-Q / (R * T))
    return D
def calculate_K(x, D, t):       # Function to calculate K
    k = (x/(2*math.sqrt(D*t)))
    return k
def calculate_Cs(cx, co, k):     # Function to calculate Cs
    try:
        if abs(1 - sp.erf(k)) != 0:
            Cs = ((cx - co) / (1 - sp.erf(k))) + co
        else:
            # Handle the case where 1 - erf(k) is very close to zero
            print("Warning: Division by a very small number encountered.")
            Cs = np.nan  # Or assign a default value
    except ZeroDivisionError:
        # Handle cases where erf(k) might exactly equal 1 (unlikely)
        print("Error: Division by zero encountered.")
        Cs = np.nan  # Or assign a default value
    return Cs
def calculate_P(Cs, R, T):  # Function to calculate P
    P = ((Cs) / ((7 * 10**-3) * (math.exp( (-20000) / (R*T) ))))**2
    return P
def cost_function(t, T, P_psi):
  cost = ((2.5/60) * t) + ( (0.125/3600) * (T - 1138.15) * t) + ((0.95/3600) * (P_psi - 14.696) * t)
  return cost

#Constant Parameters
Do = 2.3 * 10**-5  # Pre Exponential Factor
Q = 148000  # Activation energy (in Joules/mol)
R = 8.314  # Gas constant (in Joules/mol*K)
x = 0.2 * 10**-3 # Distance below the surface (m)
cx = 0.6 # wt%c at x below the surface
co = 0.3 # wt%c of the alloy


#Test Run validated through hand calcualtions

# Validation Run Parameters
T = 1138.15  # Temperature (in Kelvin)
t = 14400 # Time (s)

# Test Run Calculation
D = calculate_D(Do, Q, R, T)
K = calculate_K(x, D, t)
Cs = calculate_Cs(cx, co, K)
P = calculate_P(Cs, R, T)
P_psi = P * 0.0001450377
cost = cost_function(t, T, P_psi)

# Print out Test Run calculated values
print("D =", D, "m^2.s^-1")
print("K =", K)
print("erf(K)=",sp.erf(K))
print("Cs =", Cs, "wt%C")
print("P =", P, "Pascals")
print("P_psi =", P_psi, "Psi")
print("Cost = $", cost)


# Temperature range (K)
T_range = range(1138, 1223, 1)  # Temperature in Kelvin
# Time range (s)
t_range = range(1, 14400 + 1, 1)  # Time in seconds
# Pressure range (psi)
P_psi_min = 14.696
P_psi_max = 170


valid_data = [] # Vector to store valid Time, Temperature, Pressure, Cost combinations

# Calculating all possible combinations
for T in T_range:   # Loop through temperature range
    for t in t_range:   # Loop through time range
       
        # Calculate parameters needed for cost function in the iteration
        D = calculate_D(Do, Q, R, T)
        K = calculate_K(x, D, t)
        Cs = calculate_Cs(cx, co, K)

        # Validation Check for Cs=, due to division by 0 possibility by erf function
        if Cs is None:
            continue  # Skip to the next iteration if Cs is invalid
       
        # Caluclated Combinations pressure    
        P = calculate_P(Cs, R, T)
        # Converts Pascal to Psi
        P_psi = P * 0.0001450377    

        # Pressure range validation check
        if P_psi >= P_psi_min and P_psi <= P_psi_max:      # True if pressure satisfies constraints
            cost = cost_function(t, T, P_psi)              # Calcualtes cost of process
            valid_data.append([t, T, P_psi, cost])         # Adds valid combo to valid data vector
           
            # Printing the valid data combos (Random hand calculation validation checks)
            print(f"Temperature: {T} K, Time: {t/3600} hours, Pressure: {P_psi} psi")
            print(f"D: {D}, K: {K}, Cs: {Cs}, P: {P} Pa, Cost: ${cost}")
            print("--------------------------------------------------")
       
        else:   # For false the cost function is not calculated and combo is not strored as valid
            print(f"Temperature: {T} K, Time: {t/3600} hours, Pressure not within range")

# Prints outr all valid combos data set
print(valid_data)

# Finding lowest combination from the valid combination        
lowest_cost = float('inf')  # Sets temp variable to be able to hold float values
lowest_combo = None         # Sets temp variable to store nothing initially

for data in valid_data:     # Loop thorugh valid data set
    if data[3] < lowest_cost:   #Check if the current data point's cost is lower than the current lowest cost
        # Update the lowest cost and its corresponding data
        lowest_cost = data[3]
        lowest_combo = data
       
# Print the details of the lowest cost combination
print("Lowest Cost Combination:")
print("Time:", lowest_combo[0])
print("Temperature:", lowest_combo[1])
print("Pressure:", lowest_combo[2])
print("Cost:", lowest_combo[3])
       
# 4D Plot

# Prepare data for plotting
x = [data[0] for data in valid_data]
y = [data[2] for data in valid_data]
z = [data[1] for data in valid_data]
c = [data[3] for data in valid_data]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all points with a colormap
scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=50)

# Plot the lowest cost point in pink
ax.scatter(lowest_combo[0], lowest_combo[2], lowest_combo[1], c='pink', s=100, marker='*')

# Add labels and colorbar
ax.set_xlabel('Time')
ax.set_ylabel('Pressure')
ax.set_zlabel('Temperature')
cbar = fig.colorbar(scatter)
cbar.set_label('Cost')

plt.show()

print("Total number of data points:", len(valid_data))

