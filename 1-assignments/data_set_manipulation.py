#--------------------------------------------------------------------------------
#
# This Script manipulates the data set QM24 with some basic manipulations
#
#--------------------------------------------------------------------------------

## Libraries import

import numpy as np
import matplotlib.pyplot as plt



## Dataset manipulation

dataset = np.load("/home/a/Downloads/DFT_all.npz", allow_pickle = True) # The directory depends on where you downloaded the database, the allow_pickle is to

list_categories = dataset.files # List of data type

for category in list_categories: # Here we print the elements that compose the database
    print(category)
    print(dataset[category])


## 3D-print of an element
to_print = 100 # Number of the element to print in the dataset
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = []
y = []
z = []
for atom in dataset["coordinates"][to_print]:
    x.append(atom[0])
    y.append(atom[1])
    z.append(atom[2])


ax.scatter(x,y,z)
plt.show()
