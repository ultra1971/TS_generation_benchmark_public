import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os


def main(args):
	# Load dataset
	df = pd.read_csv(args.dataset)
	
	# Generate plot
	fig = plt.gcf()
	plt.plot(df)
	if args.show_plot == 1:
		plt.show()
	plt.draw()
	
	# Get folder
	if args.output_path is None:
		folder = extract_folder(args.dataset)
		# Create folder if necessary
		os.system(f"mkdir -p {folder}/plot")
		# Save figure
		fig.savefig(f"{folder}plot/all.png")
	else:
		folder = extract_folder(args.output_path)
		# Create folder if necessary
		os.system(f"mkdir -p {folder}")
		# Save figure
		fig.savefig(args.output_path)


def extract_folder(filename):
	b_spl = filename.split("/")
	if (len(b_spl) <= 1): 
		return ""
	# Return path witout file
	return filename.replace(b_spl[len(b_spl) - 1], "")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default="datasets/Original_Data/BeetleFly_TEST.csv", help='Path to the input dataset')
	parser.add_argument('--output_path', type=str, default=None, help='Output path')
	parser.add_argument('--show_plot', type=int, default=0, help='If 1, show plot (plot created anyway)')
	
    
	args = parser.parse_args()
    
	main(args)
