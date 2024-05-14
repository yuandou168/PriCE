import os, os.path, shutil
import csv
import plotly.graph_objects as go
from base64 import b64encode

import networkx as nx
import pandas as pd
import random

# Evenly partitionly #
# * ``is_shuffle =  True``
# * ``is_shuffle =  False``
def shuffle_evenly_division(coordinates, num_sublists, is_shuffle = False):
    # calculate the length of each sublist
    
    if is_shuffle == True:
        random.shuffle(coordinates)
    sublist_length = len(coordinates) // num_sublists
    # calculate the number of leftover elements
    leftovers = len(coordinates) % num_sublists
    # initialize the starting index of each sublist
    start = 0
    # iterate over the number of sublists
    for i in range(num_sublists):
        # calculate the end index of the sublist
        end = start + sublist_length
        # if there are leftover elements, add one to the end index
        if leftovers > 0:
            end += 1
            leftovers -= 1
        # yield the sublist
        yield coordinates[start:end]
        # update the starting index for the next sublist
        start = end


# Uniform the format of the graph splitting
def unevenly_division(multi_list, num_sublists, is_shuffle = False):
    if is_shuffle == True:
        random.shuffle(multi_list)
    

# def constructD(file_folder, strategy,dataset_number):
#     # strategy = 'is_shuffled_True'
#     # dataset_number = 4
#     for i in range(dataset_number):
#         file_path = f'{file_folder}even_split/even_patchesnames_{i}_{strategy}.csv'
#         path_pd = pd.read_csv(file_path)

# visualization1
# def graph_coloring_based_splitting_visual1(folderpath, coordinates, patch_size, d, strategy="evenly split"):
#     '''Construct a graph'''
#     G = nx.Graph()
#     pos = {}
#     nodes = []

#     # add single node with position to the empty graph
#     m_list = []
#     n_list = []
#     for i, c in enumerate(coordinates): 
#         m_list.append(int(c[0])/patch_size)
#         n_list.append(int(c[1])/patch_size)
#     # print(max(m_list), max(n_list), )

#     for i, c in enumerate(coordinates): 
#         pos[i] = (int(c[0])/patch_size%(max(m_list)+1), -int(c[1])/patch_size)
#         node_info = (i, (c[0], c[1]))
#         nodes.append(node_info)
#     # print(pos.keys(), pos.items())
#     G.add_nodes_from(pos.keys())

#     for n, p in pos.items():
#         # print(n, p)
#         G.nodes[n]['pos'] = p
#     # print(G.nodes())

#     '''Add single edge to the graph'''
#     # print(len(nodes))
#     X = G.nodes()
#     Y = G.nodes()
#     for x in X:
#         # print(x, int(G.nodes[x]['pos'][0]))
#         for y in Y:
#         # for y in Y:
#             if x!=y: 
#                 # print(x, y)
#                 Mx,Nx = int(G.nodes[x]['pos'][0]), int(G.nodes[x]['pos'][1])
#                 My,Ny = int(G.nodes[y]['pos'][0]), int(G.nodes[y]['pos'][1])
#                 la = Ny-Nx
#                 lb = My-Mx

#                 xa = pow(la,2)+pow(lb,2)
#                 # print("before", la, lb, x, y)
#                 if (la, abs(lb)) == (0,1):
#                     # print("Horizatal", la, lb, x, y)
#                     G.add_edge(x,y)
#                 elif (abs(la), lb) == (1, 0):    
#                     # print("Vertical", la, lb,x, y)
#                     G.add_edge(x,y)
#                 elif (xa) == 2:
#                     # print("duijiaoxian", la, lb,x, y)
#                     G.add_edge(x,y)
#                 else: 
#                     # print("llaaa")
#                     continue

#     # d = nx.coloring.greedy_color(G, strategy="largest_first") 
#     # d = nx.coloring.greedy_color(G, strategy=strategy)
#     # print(d.keys(), d.values())
    
#     color_numbers = 1+max(d.values()) 
#     # print("Number of nodes: ", len(d.keys()), "Number of colors: ", 1+max(d.values()))
#     # return d, color_numbers

#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = G.nodes[edge[0]]['pos']
#         x1, y1 = G.nodes[edge[1]]['pos']
#         # print(x0,y0,x1,y1)
#         edge_x.append(x0)
#         edge_x.append(x1)
#         edge_x.append(None)
#         edge_y.append(y0)
#         edge_y.append(y1)
#         edge_y.append(None)

#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines')


#     node_x = []
#     node_y = []
#     for node in G.nodes():
#         x, y = G.nodes[node]['pos']
#         node_x.append(x)
#         node_y.append(y)


#     node_trace = go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers',
#         hoverinfo='text',
#         marker=dict(
#             showscale=True,
#             # colorscale options
#             #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
#             #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
#             #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#             colorscale='Viridis',
#             reversescale=True,
#             color=[],
#             size=10,
#             colorbar=dict(
#                 thickness=15,
#                 title='Patch Node Color',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line_width=2))


#     # 
#     node_adjacencies = []
#     node_colors = []
#     node_text = []
#     for node, adjacencies in enumerate(G.adjacency()):
#         node_adjacencies.append(len(adjacencies[1]))
#         node_colors.append(d[node])
#         node_text.append('# of connections: '+str(len(adjacencies[1]))+ ", "+str((node, G.nodes[node]['pos'], d[node])))

#     node_trace.marker.color = node_adjacencies # type: ignore
#     node_trace.marker.color = node_colors # type: ignore
#     node_trace.text = node_text

#     fig = go.Figure(data=[edge_trace, node_trace],
#                 layout=go.Layout(
#                     title='<br>Colored patch nodes for distributing to multiple GPU servers',
#                     titlefont_size=18,
#                     showlegend=False,
#                     hovermode='closest',
#                     margin=dict(b=20,l=5,r=5,t=40),
#                     annotations=[ dict(
#                         # text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
#                         showarrow=False,
#                         xref="paper", yref="paper",
#                         x=0.005, y=-0.002 ) ],
#                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                     )
#     fig.write_html(os.path.join(folderpath,f"graph_{strategy}.html"))
#     fig.show()
#     color_numbers = 1+max(d.values()) 
#     print("Number of nodes: ", len(d.keys()), "Number of colors: ", 1+max(d.values()))
#     return d, color_numbers

   