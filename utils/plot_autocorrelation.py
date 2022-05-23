import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import numpy as np


def main(args):
	if args.replace == 0 and os.path.isfile(args.output_path):
		print(f"File {args.output_path} already exist and --replace is set to 0")
		return
		
	# Load dataset
	df = pd.read_csv(args.dataset)
	
	fig, ax = plt.subplots()
	
	if args.index == 'all':
		mean_ac = np.mean([plt.acorr(df.iloc[:,i], maxlags=args.maxlags)[1] for i in range(len(df.columns))], axis=0)
		x = plt.acorr(df.iloc[:,0], maxlags=args.maxlags)[0] 
		plt.close('all')

		lc = LineCollection([(x_c,0),(x_c,y_c)] for x_c, y_c in zip(x, mean_ac))
		fig, ax = plt.subplots()

		ax.add_collection(lc)
		ax.autoscale()
		# Keep only positive part
		plt.xlim([0, max(x)+1])
		plt.hlines(y=0, xmin=0, xmax=max(x)+1)
	elif args.index.isdigit():
		plt.acorr(df.iloc[:,int(args.index)], maxlags=args.maxlags)
		# Keep only positive part
		plt.xlim(0, args.maxlags + 1)
	else:
		print(f"--index must be an integer or 'all'. Received: {args.index}")
		return

	if args.title is not None:
		fig.suptitle(args.title)

	# Save result
	if args.output_path is None:
		folder = extract_folder(args.dataset)
		# Create folder if necessary
		os.system(f"mkdir -p {folder}/plot/density")
		# Save figure
		fig.savefig(f"{folder}plot/density/all.png")
	else:
		folder = extract_folder(args.output_path)
		# Create folder if necessary
		os.system(f"mkdir -p {folder}")
		# Save figure
		fig.savefig(args.output_path)
		
		
# Get folder	
def extract_folder(filename):
	b_spl = filename.split("/")
	if (len(b_spl) <= 1): 
		return ""
	# Return path witout file
	return filename.replace(b_spl[len(b_spl) - 1], "")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default="datasets/Original_Data/BeetleFly_TEST.csv", help='Path to the dataset')
	parser.add_argument('--output_path', type=str, default=None, help='Output path')
	parser.add_argument('--index', type=str, default='all', help='Index of the time series to plot, or \'all\' to plot the average acf')
	parser.add_argument('--title', type=str, default=None, help='Title of the figure')
	parser.add_argument('--maxlags', type=int, default=10, help='Number of lags to show')
	parser.add_argument('--replace', type=int, default=0, help='If 1, replace file at output_path (if existing)')
	
    
	args = parser.parse_args()
    
	main(args)
