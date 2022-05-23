import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os


def main(args):

	if args.replace == 0 and os.path.isfile(args.output_path):
		print(f"File {args.output_path} already exist and --replace is set to 0")
		return
		 
	# Load dataset
	df = pd.read_csv(args.dataset)
	df.columns = [x for x in range(len(df.columns))]
	
	# Generate plot
#	fig = plt.gcf()
	fig, ax = plt.subplots()
	# Plot distributions
#	for _, values in df.iteritems():
#		plt.hist(values, alpha=.2, bins=args.nbins)

	kwargs ={"color":"darkcyan"}
	if args.individually:
		df.hist(grid=False, bins=args.nbins, ax=ax, **kwargs)
	else:
		plt.hist(df.to_numpy().reshape(-1,1), bins=args.nbins, **kwargs)
		
		
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
	parser.add_argument('--nbins', type=int, default=10, help='Number of bins for the histograms')
	parser.add_argument('--title', type=str, default=None, help='Title of the figure')
	parser.add_argument('--individually', type=int, default=0, help='Plot a different plot for each time sereis in the dataset')
	parser.add_argument('--replace', type=int, default=0, help='If 1, replace file at output_path (if existing)')
	
    
	args = parser.parse_args()
    
	main(args)
