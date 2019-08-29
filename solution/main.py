import os
from graph import *
from kserver import *

FILE_PATH = 'data/'
FILE_FORMAT = '.mtx'
MENU_OPTIONS = ('Graph loaded from files',
				'Graph generated randomly',
				'Exit')

def create_kserver(D, graph):
	'''
	Create the k-server problem
	:param D: the array of pairwise shortest distances that would be produced
	'''
	print('\nDone creating the graph. Next, solve the k-server problem on the graph.')

	print('\nSecondly, create the k-server problem.')

	n = len(D)
	kserver = KServer()

	while True:
		way_to_generate_ic = int(input('   1. Enter the way you want to generate initial configurations (0,1,2,3): '))
		C0 = kserver.generate_initial_configs(n, graph, way_to_generate=way_to_generate_ic)
		if C0:
			break
	 		
	# C0 = [4, 31, 40, 41]

	print('\n      The initial configuration is ', C0)

	while True:
		while True:
			requests_number = int(input('\n   2. Enter the number of requests (at most 1000): '))
			if requests_number <= 1000:
				break
		way_to_generate_r = int(input('   3. Enter the way you want to generate requests (0,1,2): '))

		r = kserver.generate_requests(n, requests_number, way_to_generate_r)
		if r:
			break

	# r = [28, 17, 34, 2, 31, 1, 22, 16, 28, 13, 44, 36, 35, 25, 44, 5, 31, 15, 26, 43, 22, 31, 15, 22, 26, 40, 8, 7, 16, 34, 28, 21, 6, 21, 2, 38, 10, 45, 43, 1, 21, 13, 33, 33, 44, 21, 39, 21, 34, 39, 43, 42, 25, 28, 27, 31, 4, 19, 22, 33, 33, 42, 23, 10, 43, 12, 18, 28, 43, 19, 14, 13, 38, 36, 14, 26, 20, 37, 35, 36, 35, 5, 30, 44, 33, 17, 21, 26, 16, 28, 8, 6, 35, 37, 21, 25, 26, 27, 24, 19]

	print('\n      The requests are ', r)

	print('\nCalculating the configuration table...')

	configuration_table = kserver.get_configuration_table(D, r, C0)

	print('\nDone calculating the configuration table and solving the k-server problem...')

	opt = kserver.computeOPT(configuration_table)
	wfa = kserver.computeWFA(D, r, C0, configuration_table)
	greedy = kserver.computeGreedy(D, r, C0)
	cr_wfa = wfa / opt
	cr_greedy = greedy / opt

	result = '''\nAccording to the k-server created, the results show as follows:\n   1. Optimum           Value: %d \n   2. Work Function Algorithm: %d \n   3. Competitive Ratio (WFA): %.2f \n   4. Greedy        Algorithm: %d \n   5. Competitive Ratio (Greedy): %.2f \n
	''' %(opt, wfa, cr_wfa, greedy, cr_greedy)

	print(result)

	main_menu()

def main_menu():
	'''
	Main Menu on the console
	'''
	# Get all graph data files
	graph_data_files = []
	for filename in os.listdir(FILE_PATH):
		if filename.endswith(FILE_FORMAT):
			graph_data_files.append(filename.split('.')[0])

	count = 1 # Initialize count to 1 to use for the loop

	for menu_option in MENU_OPTIONS:
		print(str(count) + '> ' + menu_option)
		count += 1

	while True:
		try:
			user_choice = int(input('\nChoose your option: '))

			if user_choice == 1:
				file_name = input('\nFirstly, create a graph from the file. Pls enter the name of file from following options: \n\n' + ', '.join(graph_data_files) + '\n\n')

				graph_from_file = Graph()
				graph_from_file.generate_from_file(file_name)
				graph_info = graph_from_file.get_graph_info()
				D = graph_from_file.computeDistanceFromFile()

				create_kserver(D, graph_info)

			elif user_choice == 2:
				random_graph = Graph()
				print('\nFirstly, create a random graph by following steps.')
				while True:
					n = int(input('   1. Enter the number of vertices (at most 50): '))
					if n <= 50:
						break
				while True:
					p = float(input('   2. Enter the probability (between 0 and 1): '))
					if p >= 0 and p <= 1:
						break
				random_graph.generate_randomly(n, p)
				graph_info = random_graph.get_graph_info()

				# random_graph.display_graph('Random Graph')

				D = random_graph.computeDistanceFromFile()

				create_kserver(D, graph_info)

			elif user_choice == 3:
				break
			else:
				print('\nInvalid choice! Pls renter a number (1 - ' + str(count-1) +')\n')
				main_menu()
		except ValueError:
			print('\nInvalid choice! Pls renter a number (1 - ' + str(count-1) +')\n')
	exit()

# Define the main method
if __name__ == '__main__':
	'''
	Main method
	'''
	print('\n----------- k-Server Problem on Graph Metrics -----------\n')
	main_menu()
