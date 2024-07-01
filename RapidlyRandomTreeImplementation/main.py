'''
This file contains a full implementation of RRT algorithms. RRT algorithm is a widely used robotic path planning algorithms
The algorithms prerequisite:
1. Knowing the surrounding environment(obstacles)
2. Knowing the starting and ending locations
**Note: This is a 2-dimensional implementation
The algorithm proceed as follows:
1. initialize the map as every square is one coordinate/location and the starting/ending location
2. from the starting location, randomly generate a coordinate that is in the map, set this coordinate
    as our direction
3. Based on the direction generated from the previous steps, search 1 unit forward coordinate for obstacles
    based on the results of the search, there exists 2 cases:
        a. if there exists obstacles: go back to step 2
        b. if there exists non obstacles: add the coordinate that is 1 unit forward to our tree.
4. Repeat step 2 and 3 with the condition that in each direction we generate nodes from the closest node compare to the
    direction nodes
5. Plot the map as we proceed the algorithm

**Square is defines as one pixel in the image
**Unit is define as a distance of 10 pixel
'''
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation

from Node import Node
from Stack import Stack

# Position list to store clicked points, pos[0] is x values, pos[1] stores y values
pos = [Stack(), Stack()]  # this will record position for all the visualization part including direction node,
# median node and the source node
nodes = []  # this will record all the median node, source node and destination node
RADIUS = 5  # condition to determine destination reached

# Path to the image file
url = 'st_marry_hospital.png'

# Initialize the figure and scatter plot
fig, ax = plt.subplots()
pic = Image.open(url)
pixel_arr = np.zeros_like(np.arange(pic.size[0] * pic.size[1]).reshape(pic.size[1], pic.size[0]))  # create a zero
# like arr such that the size of the arr equals the size of image
ax.imshow(pic, origin='lower')
scatter = ax.scatter(pos[0].get_stack(), pos[1].get_stack(), c='b', s=10)
ax.axis('off')


def onclick(event):
    """
    Handle mouse click events, updating the graph with new points.
    """
    x = round(event.xdata)  # here we round the coordinates to the nearest integer thus we can use it in the pixel arr
    y = round(event.ydata)
    update_plot(x, y, None)


def update_plot(x, y, parent):
    '''
    Update the scatter plot with new data points.
    :param parent: parent of the current
    :param x: x coordinates
    :param y: y coordinates
    :return:
    '''
    pos[0].push(x)
    pos[1].push(y)
    if len(nodes) == 0:
        nodes.append(Node(x, y, 'source', 0))  # create a new object
    elif len(nodes) == 1:
        nodes.append(Node(x, y, 'destination', 1))
    else:
        nodes.append(Node(x, y, 'median', parent))
    scatter.set_offsets(np.c_[pos[0].get_stack(), pos[1].get_stack()])
    fig.canvas.draw_idle()


def generate_direction_node():
    '''
    Here we will generate the x and y coordinates of the 'direction' nodes, notice we will pop the last element after
    confirm next node
    :return: x, y coordinates of direction node
    '''
    y = np.random.randint(0, pic.size[1])
    x = np.random.randint(0, pic.size[0])
    return x, y


def compute_euclidean_distance(median_node: Node, direction_node_x, direction_node_y):
    '''
    This function will compute the euclidean distance between 2 nodes, specifically direction node and source node
    :param median_node: median node that in the graph
    :param direction_node_x: direction node x coordinate
    :param direction_node_y: direction node y coordinate
    :return:
    '''
    return math.sqrt(((direction_node_y - median_node.pos[1]) ** 2) + ((direction_node_x - median_node.pos[0]) ** 2))


def generate_median_node(closest_median_node: Node, direction_node_x, direction_node_y):
    '''
    This function is to calculate the coordinates for next median node, the new median node should have euclidean distance
    of 10 pixel compare to the closest_node
    :param closest_median_node:
    :param direction_node_x:
    :param direction_node_y:
    :return: x, y coordinates of the next median nodes
    '''
    # we begin with compute the angel between the closest median node and direction node
    dx = closest_median_node.pos[0] - direction_node_x
    dy = closest_median_node.pos[1] - direction_node_y
    angle = math.degrees(math.atan2(dy, dx))
    # we then compute the next node coordinate based on the angle
    x = closest_median_node.pos[0] + 10 * math.cos(angle)
    y = closest_median_node.pos[1] + 10 * math.sin(angle)
    print(x, y)
    update_plot(round(x), round(y), closest_median_node)
    return x, y


def is_reach(median_node: Node, radius):
    '''
    This function is for determine if the newly generated median node satisfied the ending condition, which is if the
    node is within the radius return True, otherwise False
    :param radius: a range that determine if the path has been found
    :param median_node: newly generated median node
    :return: True or False
    '''
    if radius >= compute_euclidean_distance(median_node, nodes[1].pos[0], nodes[1].pos[1]):
        return True  # if euclidean distance is smaller than radius means the median node is within the range,
        # thus return True
    return False


def path_logic():
    '''
    After the initialization we proceed to the actual path finding part of the algorithm
    :return:
    '''
    closest_median_node = float('inf')  # initial val of closest median node euclidean distance to infinity
    index_of_closest_median_node = 0
    x, y = generate_direction_node()
    for i in range(len(nodes)):  # loop through all the nodes except the destination nodes
        if nodes[i].identity == 'destination':
            pass
        eu_distance = compute_euclidean_distance(nodes[i], x, y)  # compute the eu distance
        if eu_distance < closest_median_node:  # update the distance iff the nodes are closer
            closest_median_node = eu_distance
            index_of_closest_median_node = i
    generate_median_node(nodes[index_of_closest_median_node], x, y)


def main():
    while True:  # initialize process such that will wait until we define the source and destination node and then
        # generate the first median node with source node being the parent
        if len(nodes) == 2:
            x, y = generate_direction_node()
            generate_median_node(nodes[0], x, y)
            break
    closest_median_node = nodes[-1]  # at this point the nodes list should have 3 element: source node, destination node
    # and the median node we just generated
    while not is_reach(closest_median_node, RADIUS):
        path_logic()

# ani = FuncAnimation(plt.gcf(), main, interval=3000)
#
# fig.canvas.mpl_connect('button_press_event', onclick)
#
# plt.show()
