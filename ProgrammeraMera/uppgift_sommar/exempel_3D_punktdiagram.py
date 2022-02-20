# Exempel på källkoden för ett enkelt 3D punktdiagram


from mpl_toolkits import mplot3d  # Detta bibliotek är nödvändigt för att 3D punktdiagram ska fungera
import numpy as np
import matplotlib.pyplot as plt


# Låt oss skapa några koordinater att plotta för klass 1:
z1 = np.random.normal(200, 10, 50)
x1 = np.random.normal(200, 10, 50)
y1 = np.random.normal(200, 10, 50)
 
# Låt oss skapa några koordinater att plotta för klass 2:
z2 = np.random.normal(100, 10, 50)
x2 = np.random.normal(100, 10, 50)
y2 = np.random.normal(100, 10, 50)

# Låt oss skapa några okända data:

x3 = np.random.normal(150, 20, 20)
y3 = np.random.normal(150, 20, 20)
z3 = np.random.normal(150, 20, 20)

    
# Skapa figur
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ='3d')
 
ax.scatter3D(x1, y1, z1, color = 'r') # skapa punktdiagram för klass 1

ax.scatter3D(x2, y2, z2, color = 'b') # skapa punktdiagram för klass 2

ax.scatter3D(x3, y3, z3, color = 'k') # skapar punktdiagram för okända data:

plt.title('3D punktdiagram')
ax.set_xlabel('Egenskap 1')
ax.set_ylabel('Egenskap 2')
ax.set_zlabel('Egenskap 3')

plt.show() # Visa punktidiagrammet
