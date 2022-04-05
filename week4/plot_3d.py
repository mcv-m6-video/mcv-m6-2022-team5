import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from matplotlib import cm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the inference script')
    parser.add_argument('-d', '--data', default='data.json', type=str, help='Relative path to json with grid search data')
    parser.add_argument('-c', '--compensation', default='backward', type=str, help='Compensation type: backward - forward')
    parser.add_argument('-f', '--function', default='ncc', type=str, help='Distance function: ncc - ssd - sad')
    parser.add_argument('-s', '--stride', default=1, type=int, help='Stride value used in grid seatch (data.json contains 1 and 2)')

    return parser.parse_args()

args = parse_args()

with open(args.data, 'rb') as op:
    data = json.load(op)

back_ssd = data[args.compensation][args.function]
X, Y, Z1, Z2 = [],[],[],[]
for res in back_ssd:
    if res[2] == args.stride: # only stride of 1
        X.append(res[0])
        Y.append(res[1])
        Z1.append(res[3])
        Z2.append(res[4])

ind_min = np.argmin(Z2)
ind_max = np.argmax(Z2)

print(f'Best result for {args.compensation} - {args.function}: MSEN: {Z1[ind_min]}, PEPN {Z2[ind_min]}, Block: {X[ind_min]}, Shift: {Y[ind_min]}')
print(f'Worst result for {args.compensation} - {args.function}: MSEN: {Z1[ind_max]}, PEPN {Z2[ind_max]}, Block: {X[ind_max]}, Shift: {Y[ind_max]}')

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(np.array(X), np.array(Y), np.array(Z1), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('Block')
ax.set_ylabel('Shift')
ax.set_zlabel('MSNE')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(np.array(X), np.array(Y), np.array(Z2), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('Block')
ax.set_ylabel('Shift')
ax.set_zlabel('PEPN')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()