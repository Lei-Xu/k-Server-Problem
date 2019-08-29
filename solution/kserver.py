import random
from itertools import combinations
from itertools import permutations
import pprint

from graph import *

class KServer(object):
	'''
	Class represents the k-server problem.
	'''
	def __init__(self):
		self.__k = 4

	def generate_requests(self, n, requests_number, way_to_generate = 0):
		'''
		Generate the requests according to the given way
		:param n: the number of vertices
		:param requests_number: the length of requests
		:param way_to_generate: the way how to generate requests
		:return: the list of requests
		'''
		if way_to_generate == 0:
			# generate the input requests by sampling each new request uniformly at random from [n] 
			# and independently of all others
			if n >= requests_number:
				requests = random.sample(range(1, n), requests_number)
			else:
				print('\n      Since vertices is not more enough to generate requests different from each other!')
				requests = []
		elif way_to_generate == 1:
			# generate the input requests by picking 5 nodes from V uniformly at random
			# and always requesting one of the 5 nodes such that Greedy does not have a server present 
			# at the requested node at the time of the request
			seed = random.sample(range(1, n), 5)
			requests = [random.choice(seed) for _ in range(requests_number)]
		else:
			# generate the input requests randomly but with some dependence
			requests = [random.randint(1, n) for _ in range(requests_number)]
		return requests

	def generate_initial_configs(self, n, graph, initial_configs_number = 4, way_to_generate = 0):
		'''
		Generate the initial configurations according to the given way
		:param n: the number of vertices
		:param initial_configs_number: the length of initial configurations, known as the number of servers
		:param way_to_generate: the way how to generate requests
		:return: the list of requests
		'''
		if way_to_generate == 0:
			# generate the initial configurations by sampling each new initial configuration uniformly at random from [n] 
			# and independently of all others
			if n >= initial_configs_number:
				initial_configs = random.sample(range(1, n), initial_configs_number)
			else:
				print('\n      Since vertices is not more enough to generate the initial configuration different from each other!')
				initial_configs = []
		elif way_to_generate == 1:
			# generate the initial configurations whose each initial configuration is as far as from others
			m = int(n / initial_configs_number)
			initial_configs = [random.randint((1+m*i), (m+m*i)) for i in range(initial_configs_number)]
		elif way_to_generate == 2:
			# generate the initial configurations consisting with nodes with a higher degree
			initial_configs_with_degree = sorted(self.__get_degree_info(graph).items(), key=lambda x: x[1])[-4:]
			initial_configs = [x for x, _ in initial_configs_with_degree]
		else:
			# generate the initial configurations consisting with nodes with a lower degree
			initial_configs_with_degree = sorted(self.__get_degree_info(graph).items(), key=lambda x: x[1])[:4]
			initial_configs = [x for x, _ in initial_configs_with_degree]
		initial_configs.sort()
		return initial_configs

	def computeOPT(self, configuration_table):
		'''
		Compute the optimal solution of the k-server problem using the work function
		:param configuration_table: the table containing work function values of all configurations
		:return: the minimum total move distance
		'''
		return(min(configuration_table[-1].values()))

		
		
	def computeWFA(self, D, r, C0, configuration_table):
		'''
		Compute the optimal solution of the k-server problem using the Work Function algorithm
		on the sequence r starting in configuration C0
		:param D: the array of pairwise shortest distances that would be produced
		:param r: the sequence of requests
		:param C0: the initial pre-specified configuration of the k-server problem (servers)
		:param configuration_table: the table containing work function values of all configurations
		:return: the minimum total move distance
		'''
		total_distance = 0
		vertices_number = len(D)

		C = C0[:]
		for request_number, request in enumerate(r):
			if request not in C:
				values = {}
				for index, vertex in enumerate(C):
					C_request = C[:]
					C_request[index] = request
					C_request.sort()
					values[vertex] = configuration_table[request_number][tuple(C_request)] + D[request-1][vertex-1]
				min_value = min(values.items(), key=lambda x: x[1])[1]
				vertices_min_value = [key for key, value in values.items() if value == min_value]
				vertex_min_value = vertices_min_value[random.randint(0, len(vertices_min_value)-1)]
				total_distance += D[request-1][vertex_min_value-1]
				C[C.index(vertex_min_value)] = request
				C.sort()

		return total_distance

	def computeGreedy(self, D, r, C0):
		'''
		Compute the optimal solution of the k-server problem using the Greedy algorithm
		on the sequence r starting in configuration C0
		:param D: the array of pairwise shortest distances that would be produced
		:param r: the sequence of requests
		:param C0: the initial pre-specified configuration of the k-server problem (servers)
		:return: the minimum total move distance
		'''
		total_distance = 0
		for request in r:
			if request not in C0:
				shortest_distance, server_selected = self.__get_shortest_distance(D, request, C0)
				total_distance += shortest_distance
				C0[C0.index(server_selected)] = request

		return total_distance

	def __get_shortest_distance(self, D, request, servers):
		'''
		Get the shortest distance between request and servers
		:param request: the request representing as a number
		:param servers: the array represents positions of all servers
		:param D: the array of pairwise shortest distances that would be produced
		:return: the minimum distance between request and all servers
		'''
		shortest_distance = 0
		for server in servers:
			if shortest_distance == 0 or (shortest_distance != 0 and D[request-1][server-1] < shortest_distance):
				shortest_distance = D[request-1][server-1]
				servers_selected = []
				servers_selected.append(server)
			elif D[request-1][server-1] == shortest_distance:
				servers_selected.append(server)
		return shortest_distance, servers_selected[random.randint(0, len(servers_selected)-1)]

	def get_configuration_table(self, D, r, C0):
		'''
		Get the configuration table of all work function values
		:param D: the array of pairwise shortest distances that would be produced
		:param r: the sequence of requests
		:param C0: the initial pre-specified configuration of the k-server problem (servers)
		:return: the configuration table formatted as a 2D list
		'''
		configuration_table = []

		r.insert(0, '')
		vertices_number = len(D)
		requests_number = len(r)

		configurations = list(combinations(list(range(1, vertices_number+1)), self.__k))
		
		for request_number, request in enumerate(r):
			configuration_table_column = {}
			for configuration in configurations:
				if request == '':
					wf_value = self.__get_shortest_distance_between_configs(D, C0, configuration)
				else:
					wf_value = self.__compute_wf(D, request, configuration, configuration_table, request_number)				
				configuration_table_column[configuration] = wf_value
			configuration_table.append(configuration_table_column)
		r.remove('')
		return configuration_table

	def __get_shortest_distance_between_configs(self, D, C0, configuration):
		'''
		Compute the first row of the configuration table
		:param D: the array of pairwise shortest distances that would be produced
		:param C0: the initial pre-specified configuration of the k-server problem (servers)
		:param configuration: one possible configuration of the k-server problem
		:return: the shortest distance between C0 and the given configuration
		'''
		distances = []
		a, b, c, d = configuration
		C0_combinations = list(permutations(C0, self.__k))
		for e, f, g, h in C0_combinations:
			distances.append(D[a-1][e-1] + D[b-1][f-1] + D[c-1][g-1] + D[d-1][h-1])
		return min(distances)

		# for ppt - testing
		# a, b, c = configuration
		# C0_combinations = list(permutations(C0, self.__k))
		# for e, f, g in C0_combinations:
		# 	distances.append(D[a-1][e-1] + D[b-1][f-1] + D[c-1][g-1])
		# return min(distances)
		# 
		# for assignment2 - testing
		# a, b = configuration
		# C0_combinations = list(permutations(C0, self.__k))
		# for e, f in C0_combinations:
		# 	distances.append(D[a-1][e-1] + D[b-1][f-1])
		# return min(distances)


	def __compute_wf(self, D, request, C, configuration_table, request_number):
		'''
		Compute the work function value
		:param D: the array of pairwise shortest distances that would be produced
		:param request: the coming request
		:param C: the current configuration of the k-server problem
		:param configuration_table: the table containing all work function values of previous requests
		:param previous_request_number: the number of the previous request
		:return: the work function value of the current configuration
		'''
		wf_value = 0

		if request in C:
			wf_value = configuration_table[request_number-1][C]
		else:
			possible_wf_values = []
			
			for index, vertex in enumerate(C):
				C_list = list(C)
				C_list[index] = request
				C_list.sort()
				possible_configuration = tuple(C_list)
				possible_wf_value = configuration_table[request_number-1][possible_configuration] + D[vertex-1][request-1]
				possible_wf_values.append(possible_wf_value)

			wf_value = min(possible_wf_values)
		return wf_value

	def __get_degree_info(self, graph):
		'''
		Get the info of nodes' degree of the given graph
		:param graph: the graph where get the info of nodes' degree
		:return: the dictionary containing the info of nodes' degree of the given graph
		'''
		degree_statistics = {}
		for node in graph:
			degree_statistics[node.index+1] = len(node.neighbors)
		return degree_statistics


# Testing
# if __name__ == '__main__':
# 	graph_file = Graph()
# 	graph_file.generate_from_file('test')
# 	D = graph_file.computeDistanceFromFile()
# 	C0 = [1, 2, 3, 4]
# 	r = [5, 4, 6, 8, 1, 7]

# Case from the ppt for testing
	# D = [[0,1,1,1,2],[1,0,1,1,2],[1,1,0,1,2],[1,1,1,0,2],[2,2,2,2,0]]
	# C0 = [1, 2, 3]
	# r = [5, 4] #, 1, 2, 3, 1, 2, 1, 3, 5

# Case from the assignment2 for testing
	# D = [[0,1,2,4], [1,0,2,3],[2,2,0,3],[4,3,3,0]]
	# C0 = [1,2]
	# r = [3,4,2,3,1]

	# kserver = KServer()
	# table = kserver.get_configuration_table(D, r, C0)
	# pprint.pprint(table)
	# opt = kserver.computeOPT(table)
	# wfa = kserver.computeWFA(D, r, C0, table)
	# greedy = kserver.computeGreedy(D, r, C0)
	# print('opt -> ', opt)
	# print('wfa -> ', wfa)
	# print('greedy -> ', greedy)
# 	print(kserver.computeWFA(D, r, C0, table))
	# print(kserver.computeGreedy(D, r, C0))
	# print(table)
	# print(len(table))
	# print(min(table[-1].values()))
	



