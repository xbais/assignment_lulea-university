import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import matplotlib.colors as mcolors
import random
from headers import *
from matplotlib.patches import Circle, Patch
from matplotlib.collections import LineCollection  # Correct import
from tests import sunburst_chart_test as sct

import networkx as nx
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio  # Import plotly.io for saving the plot

def last_index(_list:list, element):
    """
    returns the index of last occurence of the element in list
    """
    return len(_list) - 1 - _list[::-1].index(element)

def plot_sunburst_chart(G, save_path:str, merge_similar_elements:bool=True):
    # Prepare data for the Sunburst chart
    ids = []
    labels = []
    parents = []
    values = []

    def traverse_graph_1(node, parent=None, depth=0):
        # Ensure we are only considering up to depth 6
        if depth > 6:
            return

        # Add the current node
        node_id = len(ids)
        ids.append(node)
        labels.append(f'{G.nodes[node]["data"]["name"]}')
        parents.append(parent if parent is not None else '')
        values.append(G.nodes[node]['data']['time_taken'])

        # Traverse child nodes
        children = list(G.successors(node))
        if children:
            # Calculate total time_taken for siblings
            total_time = sum(G.nodes[child]["data"]['time_taken'] for child in children)
            for child in children:
                traverse_graph(child, node, depth + 1)

    def traverse_graph(node, parent=None, depth=0):
        # Ensure we are only considering up to depth 6
        if depth > 6:
            return

        # Add the current node
        #node_id = len(ids)
        label = G.nodes[node]['data']["name"]
        _parent = parent if parent is not None else ''
        _existing_id = None
        if merge_similar_elements:
            if label in labels and _parent in parents:
                if last_index(labels, label) == last_index(parents, _parent):
                    _existing_id = labels.index(label)
                    #print(f'Adding to existing sunburst segment')
                    #print(f'Current node = {label} | Current parent = {_parent}')        
        
        #values.append(G.nodes[node]['data']['time_taken'])
        # Traverse child nodes
        children = list(G.successors(node))
        if children:
            # Calculate total time_taken for siblings
            total_time = sum(G.nodes[child]['data']['time_taken'] for child in children)
            _value = total_time
            for child in children:
                traverse_graph(child, node, depth + 1)
        else:
            _value = G.nodes[node]['data']['time_taken']
        
        if isinstance(_existing_id, int):
            values[_existing_id] += _value
        else:
            ids.append(node)
            labels.append(label)
            parents.append(_parent)
            values.append(_value)

    # Start traversal from root nodes (nodes with no parents)
    for node in G.nodes():
        if G.in_degree(node) == 0:  # Root nodes
            traverse_graph(node)

    # Create a Plotly Sunburst chart
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        sort=False,
    ))

    # Update layout for better appearance
    fig.update_layout(
        title="NetworkX Graph as Plotly Sunburst Chart",
        margin=dict(t=0, l=0, r=0, b=0)
    )

    # Show the plot
    #fig.show()
    # Save the plot to an HTML file
    #output_file = "sunburst_chart.html"
    pio.write_html(fig, file=save_path, auto_open=False)

def plot_networkx_graph(graph, save_path:str, color_dict=None):
    node_name_colour_dict = color_dict
    # Compute positions
    def top_down_layout(G):
        pos = {}
        levels = nx.single_source_shortest_path_length(G, 0)
        
        # Determine vertical positions based on levels
        for node, level in levels.items():
            pos[node] = (node, -level)
        
        # Sort nodes by ID for horizontal positioning
        node_ids = sorted(G.nodes())
        for i, node in enumerate(node_ids):
            pos[node] = (i, pos[node][1])
        
        return pos
    
    # Compute rendering times based on neighbors
    def compute_rendering_times(G:nx.graph, total_nodes:int=None, main_call:bool=False, parent_graph=None): # Recursive fn
        # 1. Get all the node in the graph
        # 2. Start reading the node one after the other insequence (we will add time metric to each node)
        ## a. If the node has children, read the children in order, and do the same for the children first
        ## b. Once all children are read, take sum of the times taken by each of the children and add it to the current node's time_taken param, add the node to the visited nodes
        # Repeat this till all nodes have been added to the visited nodes list.
        # Next, Re-read the graph, and layer-wise normalise the child nodes of each node.
        # Return the times
        total_nodes = G.number_of_nodes() if not total_nodes else total_nodes
        parent_graph = G if not parent_graph else parent_graph

        #print(f'Total nodes = {total_nodes}')
        times = {}
        total_time = 0
        #node_counter = 0
        for node_id in G.nodes: #range(total_nodes):
            '''
            timestamp = G.nodes[node]['data']['time']
            neighbors = list(G.neighbors(node))
            last_timestamp = None
            if neighbors:
                neighbor_times = [G.nodes[neighbor]['data']['time'] for neighbor in neighbors]
                # Choose the minimum time difference for comparison
                time_diffs = [abs((timestamp - nt).total_seconds()) for nt in neighbor_times]
                times[node] = min(time_diffs) if time_diffs else 0
            else:
                times[node] = 0  # Isolated node
            '''
            descendants = nx.descendants(G, node_id)
            #print(f'Descendants = {descendants}')
            if descendants:
                # Find the times for the descendants first
                descendant_subgraph = G.subgraph(descendants)
                #print(f'Descendant subgraph = {descendant_subgraph}')
                #for descendant in descendants:
                compute_rendering_times(descendant_subgraph, total_nodes=total_nodes, parent_graph=parent_graph)
                G._node[node_id]['data']['time_taken'] = 0
                for desc_id in descendants:
                    G._node[node_id]['data']['time_taken'] += G._node[node_id]['data']['time_taken']
            else:
                # The node is either a terminal node or is lone node
                if node_id == total_nodes -1: # Terminal node (last node of graph)
                    G._node[node_id]['data']['time_taken'] = (datetime.datetime.now() - parent_graph._node[list(parent_graph.predecessors(node_id))[0]]['data']['time']).total_seconds()#0
                else: # Lone node
                    #print(f'Lone node? Nodes = {G.nodes} | Node id = {node_id}')
                    #print(f'Predecessor list for the node = {list(parent_graph.predecessors(node_id))}')
                    G._node[node_id]['data']['time_taken'] = (G._node[node_id]['data']['time'] - parent_graph._node[list(parent_graph.predecessors(node_id))[0]]['data']['time']).total_seconds()
                    #G._node[node_id]['data']['time_taken'] = (G._node[node_id]['data']['time'] - G._node[list(nx.edge_dfs(parent_graph,node_id, orientation='reverse'))[0]]['data']['time']).total_seconds()
        if main_call:    
            #times = [G._node[_]['data']['time_taken'] for _ in range(total_nodes)]
            times = {node: G.nodes[node]['data']['time_taken'] for node in G.nodes()}

            return times
    
    # Compute rendering times
    times = compute_rendering_times(graph, main_call=True) 

    '''
    # Normalize rendering times for coloring
    def normalize_times(times):
        values = list(times.values())
        norm = plt.Normalize(min(values), max(values))
        return {node: norm(time) for node, time in times.items()}, norm

    norm_times, norm = normalize_times(times)
    cmap = plt.get_cmap('coolwarm')  # Choose a colormap
    colors = {node: cmap(norm_times[node]) for node in norm_times}
    node_colors = [colors[node] for node in graph.nodes()]
    #node_size = 1000
    font_size = 12
    # Get the current axes
    fig, ax = plt.subplots()

    # Calculate the radius of the inner circle
    inner_radius = font_size*20 #(node_size / 10000) ** 0.5  # Convert node_size to data units
    outer_radius = inner_radius/2000 #* 1.2  # Adjust the outer circle radius (20% larger than inner)

    # Draw the graph
    pos = top_down_layout(graph)  # positions for all nodes
    # Add outer circles with lower zorder (e.g., zorder=1)
    outer_color = (1, 1, 0)  # RGB for yellow
    for node in graph.nodes():
        x, y = pos[node]
        outer_circle = Circle((x, y), outer_radius, color=outer_color, fill=True, zorder=1)
        ax.add_patch(outer_circle)

    # Draw edges manually using Matplotlib's LineCollection to set zorder
    edges = []
    for edge in graph.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        edges.append(((x1, y1), (x2, y2)))

    edge_collection = LineCollection(edges, color="black", zorder=2)
    ax.add_collection(edge_collection)

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=inner_radius)
    #nx.draw_networkx_edges(graph, pos, ax=ax, zorder=10)
    nx.draw_networkx_labels(graph, pos, font_size=font_size)
    
    node_name_colour_dict = {} if not node_name_colour_dict else node_name_colour_dict
    for node in graph.nodes():
        x, y = pos[node]
        node_name = graph.nodes[node]['data']['name']
        if node_name not in node_name_colour_dict.keys():
            #print(f'Node name {node_name} not in color dict : {node_name_colour_dict}')
            node_name_colour_dict[node_name] = (random.random(),random.random(), random.random())

        # Outer circle (yellow)
        outer_circle = Circle((x, y), outer_radius, color=node_name_colour_dict[node_name], fill=True, zorder=1)
        ax.add_patch(outer_circle)

    # Set the aspect ratio to be equal to ensure circles are not distorted
    ax.set_aspect('equal') 
    # Create legend patches
    patches = [Patch(color=list(color)+[1], label=label) for label, color in node_name_colour_dict.items()]

    # Add the legend to the plot
    ax.legend(handles=patches, loc='lower center')
    # Set the limits to ensure all nodes fit
    ax.set_xlim(min(x for x, y in pos.values()) - 0.15, max(x for x, y in pos.values()) + 0.15)
    ax.set_ylim(min(y for x, y in pos.values()) - 0.15, max(y for x, y in pos.values()) + 0.15)

    # Hide axes
    ax.axis('off')
    
    # Display the graph
    plt.title("Time Graph")
    plt.show()
    '''
    plot_sunburst_chart(graph, save_path=save_path)
    #plt.savefig(save_path)
    #plt.clf() # Clear the plot
    #plt.cla()
    return node_name_colour_dict

# Function to extract parameter data at each depth level
def extract_depth_data(G):
    depth_data = defaultdict(list)
    
    for node in G.nodes(data=True):
        #plot_networkx_graph(G)
        depth = nx.shortest_path_length(G, source=0, target=node[0])
        #print(node)
        depth_data[depth].append(node[1]['data']['time'])
    
    return depth_data

# Function to map values to colors, largest in red-orange and smallest in green
def get_colors_for_layer(sizes):
    norm = mcolors.Normalize(vmin=np.min(sizes), vmax=np.max(sizes))
    cmap = plt.colormaps['RdYlGn_r']  # Updated for the deprecation warning
    return [cmap(norm(size)) for size in sizes]

def get_random_colors_for_layer(sizes, alpha=1.0): # Test Function
    """
    Generates a list of random RGBA color lists.

    Parameters:
        n (int): Number of colors to generate.
        alpha (float): The alpha (transparency) value for all colors (default is 1.0).

    Returns:
        List[List[float]]: A list of RGBA color lists.
    """
    n = len(sizes)
    colors = []
    for _ in range(n):
        r = random.random()
        g = random.random()
        b = random.random()
        a = alpha  # Fixed alpha value for all colors or random.random() for random alpha
        colors.append([r, g, b, a])
    return colors

def get_color_from_colormap(percent, colormap_name='viridis'):
    """
    Returns a color from a Matplotlib colormap based on a percentage value.

    Parameters:
        percent (float): A percentage value (0-100) to pick the color.
        colormap_name (str): Name of the Matplotlib colormap (default is 'viridis').

    Returns:
        tuple: An RGBA color tuple from the colormap.
    """
    # Normalize the percentage to be between 0 and 1
    norm_percent = percent / 100.0

    # Get the colormap
    cmap = plt.get_cmap(colormap_name)

    # Get the color from the colormap
    color = cmap(norm_percent)

    return color

# Function to plot the multi-layer pie chart
def plot_multilayer_pie_chart(labels, sizes, save_path:str):
    n_layers = len(sizes)
    fig, ax = plt.subplots()
    ax.axis('equal')

    for i in range(n_layers):
        #colors = get_colors_for_layer((np.array(sizes[i])*255 / np.linalg.norm(sizes[i])).tolist())
        _sizes = (np.array(sizes[i])*100 / np.linalg.norm(sizes[i])).tolist()
        colors = [get_color_from_colormap(_) for _ in _sizes]
        #print(colors)
        wedges, texts = ax.pie(
            sizes[i], labels=None, radius=1 - i*0.3, colors=colors,
            wedgeprops=dict(width=0.3, edgecolor='w')
        )

        # Draw labels directly outside the wedges with a white background
        for j, (wedge, label) in enumerate(zip(wedges, labels[i])):
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            ax.text(
                1.1 * x, 1.1 * y, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
            )
    fig.tight_layout()
    #plt.show()
    plt.savefig(save_path)
    plt.clf() # Clear the plot
    plt.cla()

def core(G, save_path:str, color_dict=None):
    # Extract parameter distribution by depth
    depth_data = extract_depth_data(G)

    # Prepare data for plotting
    depth_labels = []
    size_data = []

    for depth, parameters in sorted(depth_data.items()):
        unique_params, counts = np.unique(parameters, return_counts=True)
        depth_labels.append([f"Depth {depth} - {param}" for param in unique_params])
        size_data.append(counts)

    # Plot the multi-layer pie chart
    #plot_multilayer_pie_chart(depth_labels, size_data, save_path)
    node_name_color_dict = plot_networkx_graph(G, color_dict=color_dict, save_path=save_path)
    return node_name_color_dict

if __name__ == '__main__':
    # Create a test graph
    G = nx.DiGraph()

    # Add nodes with a 'time' attribute
    G.add_node(0, parameter='A')
    G.add_node(1, parameter='B')
    G.add_node(2, parameter='A')
    G.add_node(3, parameter='C')
    G.add_node(4, parameter='B')
    G.add_node(5, parameter='A')

    # Add edges to create a graph structure
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)])

    core(G)
