import matplotlib
import json
import argparse
from matplotlib import pyplot as plt

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--list', '-l')
	parser=parser.parse_args();
	input_file = parser.list

	points=json.load(open(input_file))
	fig=plt.figure()
	for i in range(len(points)):
		x=[x[0] for x in points[i]]
		y=[x[1] for x in points[i]]
		plt.scatter(x,y)
	plt.show()