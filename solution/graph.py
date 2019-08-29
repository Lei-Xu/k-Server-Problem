import random
import sys

class Node(object):
    '''
    Class represents the information related to the node of the graph
    '''
    def __init__(self, index):
        '''
        :param index: the index of the node, indicating as a digit
        '''
        self.index = index
        self.neighbors = []


class Graph(object):
    '''
    Class represents the information related to the graph
    '''

    FILE_PATH = 'data/'
    FILE_FORMAT = '.mtx'
    COMMENT_TAG = '%'

    def __init__(self):
        '''
        Represent the graph as an adjacency list
        '''
        self.__vertices = {}

    def generate_from_file(self, filename):
        '''
        Generate a graph from the given file
        :param filename: the name of the Matrix Market Coordinate (MMC) format file represents the undirected unweighted graph
        '''
        vertices_number, edges = self.__get_graph_data(filename)

        self.__vertices = [Node(i) for i in range(vertices_number)]

        for (u, v) in edges:
            self.__vertices[u-1].neighbors.append(self.__vertices[v-1])
            self.__vertices[v-1].neighbors.append(self.__vertices[u-1])

        if not self.__is_connected():
            print('\nYou are trying to solve the k-server problem on the graph, which is not a connected graph. Pls use another graph file!')
            sys.exit(1)

    def generate_randomly(self, n, p):
        '''
        Generate a random Erdos-Renyi graph denoted as G(n, p)
        :param n: the number of vertices
        :param p: the probability of including an edge
        '''
        self.__vertices = [Node(i) for i in range(n)]
        edges = [(i, j) for i in range(n) for j in range(i) if random.random() <= p]

        for (u, v) in edges:
            self.__vertices[u].neighbors.append(self.__vertices[v])
            self.__vertices[v].neighbors.append(self.__vertices[u])

        if not self.__is_connected():
            print('\nThe graph generated randomly is not a connected graph. Maybe try to increase the probability!')
            sys.exit(1)

    def computeDistanceFromFile(self):
        '''
        Compute the shortest path between vertex using BFS and store them in the array D
        :return: the array D storing pairwise shortest distances
        '''

        vertices_number = len(self.__vertices)
        D = [[0 for column in range(vertices_number)] for row in range(vertices_number)]
        for i in range(vertices_number):
            for j in range(i + 1, vertices_number):
                shortest_path = self._bfs(i, j)
                D[i][j] = shortest_path
                D[j][i] = shortest_path
        return D
        # print(D)
    
    def get_graph_info(self):
        '''
        Return the graph
        :return: the graph
        '''
        return self.__vertices

    def display_graph(self, graph_name):
        '''
        :param graph_name: the name of the graph
        Display the graph representing as an adjacency list
        '''
        print('-------- ', graph_name, ' --------')
        for node in self.__vertices:
            node_nerghbor_string = ''
            for node_nerghbor in node.neighbors:
                node_nerghbor_string = node_nerghbor_string + str(node_nerghbor.index+1) + ', '
            print(str(node.index+1), ' -> ', node_nerghbor_string)

    def __get_graph_data(self, filename):
        '''
        Get the graph information from the given file
        :return: the list containing all edges of the graph
        '''
        edges_list = []
        with open(self.FILE_PATH + filename + self.FILE_FORMAT) as mtx_file:
            for line in mtx_file:
                if not line.startswith(self.COMMENT_TAG):
                    basic_arguments = line.strip().split(' ')
                    if len(basic_arguments) == 3:
                        vertices_number = int(basic_arguments[0])
                        continue
                    vertex_1 = int(basic_arguments[0])
                    vertex_2 = int(basic_arguments[1])
                    edges_list.append([vertex_2, vertex_1])

        return vertices_number, edges_list

    def __is_connected(self):
        '''
        Check whether the given undirected unweighted is a connected graph
        :return: True -> connected, False -> not connected
        '''
        vertices_number = len(self.__vertices)
        components_set = [index for index in range(vertices_number)]

        for node in self.__vertices:
            for node_neighbor in node.neighbors:
                fu = self.__find(node.index, components_set)
                fv = self.__find(node_neighbor.index, components_set)
                components_set[fv] = fu
        count = 0
        for node in self.__vertices:
            if components_set[node.index] == node.index:
                count += 1
        if count == 1:
            # print('True')
            # print(components_set)
            return True
        else:
            # print('False')
            # print(components_set)
            return False

    def __find(self, node_index, components_set):
        '''
        Find the root of the given node
        :param node_index: the index of the node
        :param component_set: the set tracking the nodes' component
        :return: the index of the given node's root
        '''
        return node_index if node_index == components_set[node_index] else self.__find(components_set[node_index], components_set)

    def _bfs(self, start_point, end_point):
        '''
        Compute the shortest path between vertex using the BFS
        :param start_point: the starting point
        :param end_point: the end point
        :return: the shortest path between the two given points
        '''

        count = 0
        flag = False
        vertex_traverse_queue = []
        vertex_flag_queue = []

        distance = 0
        start_node = self.__vertices[start_point]
        vertex_traverse_queue.append(start_node)
        vertex_flag_queue.append(start_node)

        while vertex_traverse_queue:
            node = vertex_traverse_queue.pop()
            for node_nerghbor in node.neighbors:
                if node_nerghbor.index == end_point:
                    distance = count + 1
                    flag = True
                    break
                elif node_nerghbor not in vertex_flag_queue:
                    vertex_flag_queue.append(node_nerghbor)
                    vertex_traverse_queue.append(node_nerghbor)
            if flag:
                break
            count += 1

        return distance

# Testing
# if __name__ == '__main__':
#     graph_file = Graph()
#     graph_file.generate_from_file('test')
#     graph_file.computeDistanceFromFile()
#     graph_file.display_graph('Graph generated from the file')

#     random_graph = Graph()
#     random_graph.generate_randomly(100, 0.01)
#     random_graph.display_graph('Random Graph')
#     random_graph.computeDistanceFromFile()

