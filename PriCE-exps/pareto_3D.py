# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

def remove_duplicates(lst):
    unique_set = set()
    unique_list = []
    for inner_list in lst:
        # Convert the inner list to a tuple to make it hashable
        tuple_inner_list = tuple(inner_list)
        if tuple_inner_list not in unique_set:
            unique_list.append(inner_list)
            unique_set.add(tuple_inner_list)
    return unique_list

# Define a custom formatting function
def y_formatter(x, pos):
    return f"{x*1000:.0f}"


def simple_cull(inputPoints, dominates, sMax=True):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            
            if dominates(candidateRow, row, sMax):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow, sMax):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow, sMax = True):
    if sMax:
        return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)  
    else: 
        return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)


# import random
# # inputPoints = [[random.randint(1,5) for i in range(3)] for j in range(500)]
# inputPoints = [[random.choice([1,2,3,4,5,6]), random.uniform(100.5, 750.5), random.uniform(10.5, 75.5)] for j in range(100)]
# print(inputPoints)
# inputPoints = remove_duplicates(inputPoints)


# # paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates, sMax=True) # find maximum
# paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates, sMax=False) # find minimum
# print(len(paretoPoints), len(dominatedPoints))

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# plt.rcParams['text.usetex'] = True

# # fig = plt.figure(layout="constrained")
# # (ax1, ax2) = fig.subplots(2, 2, squeeze=False)
# fig, axs = plt.subplots(2, 1, subplot_kw={'projection': '3d'})


# # axs = fig.add_subplot(111, projection='3d')
# dp = np.array(list(dominatedPoints))
# pp = np.array(list(paretoPoints))
# print(pp.shape,dp.shape)

# axs[0].scatter(dp[:,0],dp[:,1],dp[:,2],color='darkcyan')
# axs[0].scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
# # plt.show()

# import matplotlib.tri as mtri
# print(pp)
# # fig2= plt.figure()
# if len(dominatedPoints)>=2 and len(paretoPoints)>=2: 
#     axs[1].scatter(dp[:,0],dp[:,1],dp[:,2])
#     axs[1].scatter(pp[:,0],pp[:,1],pp[:,2],color='red', marker='^', s=70)
#     triang = mtri.Triangulation(pp[:,0],pp[:,1])
#     axs[1].plot_trisurf(triang,pp[:,2],color='red')
# elif len(paretoPoints)<=2 or len(dominatedPoints)<=2: 
#     axs[1].scatter(dp[:,0],dp[:,1],dp[:,2])
#     axs[1].scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
# elif len(paretoPoints)>=2 and len(dominatedPoints)<=2:
#     axs[1].scatter(dp[:,0],dp[:,1],dp[:,2])
#     axs[1].scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
#     triang = mtri.Triangulation(pp[:,0],pp[:,1])
#     axs[1].plot_trisurf(triang,pp[:,2],color='red')
# else: 
#     axs[1].scatter(dp[:,0],dp[:,1],dp[:,2])
#     axs[1].scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
# # axs.legend()
# axs[1].set_title("Pareto frontier 3D")

# bbox = dict(boxstyle="round", fc="0.8")
# # arrowprops = dict(
# #     arrowstyle="->",
# #     connectionstyle="angle,angleA=0,angleB=90,rad=10")
# axs[1].zaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
# # axs[1].text(min(dp[:,2]), max(dp[:,2]), '(x10$^3$)', transform=axs[1].transAxes, fontsize=16, va='bottom', ha='left', s=20)

# print(pp[:,0])
# xdata = pp[:,0][0]
# ydata = pp[:,1][0]
# zdata = pp[:,2][0]
# # axs[1].annotate(
# #     f'data = ({xdata:.1f}, {ydata:.1f}, {zdata:.1f})',
# #     (xdata, ydata, zdata),

# axs[1].text2D(0.05, 0.95, f'data = ({xdata:.1f}, {ydata:.1f}, {zdata:.1f})', transform=axs[1].transAxes)
# axs[1].set_xlabel('Number of GPU servers')
# axs[1].set_xticks(np.arange(min(dp[:,0]), max(dp[:,0]),1))
# axs[1].set_ylabel('Makespan')
# axs[1].set_zlabel(r'Total cost ($\times 10^{-3}$ \$)')
# # plt.savefig('/Volumes/T7/Research/WSI preprocessing project/Distributed Image Processing/image_transmission/benchmarks/res/'+"pareto3D_res.png")
# plt.show()



