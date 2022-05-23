import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import argparse


def main(args):
	# Create filter
	kf = KalmanFilter(transition_covariance=args.transition_covariance)
	# Extract dataset name
	name = extract_name(args.dataset)
	print(f"Applying Kalman filter to {name}")
	# Load dataset
	dataset = pd.read_csv(args.dataset)
	# Create new df
	filtered_df = pd.DataFrame()
	# Apply filter
	for col in dataset.columns:
		smo_m, _ = kf.smooth(dataset.loc[:,col])
		filtered_df.loc[:,col] = smo_m.flatten()[args.remove_initial:]
	# Ev create output path
	if args.output_path == "Default":
		savepath = f"{remove_ext(args.dataset)}_filtered.csv"
	else:
		savepath = f"{args.output_path}"
	# Save data
	filtered_df.to_csv(savepath, header=False, index=False)
	print(f"Filtered (smoothed) data saved at {savepath}")
	
	
def extract_name(path):
	# Remove ev. extension
	p_spl = path.split(".")
	if (len(p_spl) <= 1): 
		return p_spl[0]
	name = p_spl[len(p_spl) - 2]
	# Remove ev. folder path
	b_spl = name.split("/")
	name = b_spl[len(b_spl) - 1]
	return name
	
def remove_ext(path):
	p_spl = path.split(".")
	ext = p_spl[len(p_spl) - 1]
	return path.replace(f".{ext}",'')

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default="../datasets/Original_Data/BeetleFly_TEST.csv", help='Path to the input dataset')
	parser.add_argument('--output_path', type=str, default="Default", help='Output path')
	parser.add_argument('--transition_covariance', type=float, default=0.05, help='Value of the trainsition covariance to use. The bigger this value, the more the data is "flatten"')
	parser.add_argument('--remove_initial', type=int, default=0, help='Number of initial points to remove after applying filter. Usefult for ex. with TimeGAN')
	    
	args = parser.parse_args()
    
	main(args)
