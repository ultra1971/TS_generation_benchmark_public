from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
import numpy as np
import argparse
import os


def main(args):
	# Load dataset
	data = pd.read_csv(args.dataset)
	# Collect generated data
	generated = []
	# Choose lag window
	lag_win = int(len(data.iloc[:,0])/4) if args.lag_window == 0 else args.lag_window
	for s in range(len(data.columns)):
		X = tuple(data.iloc[:,s])   # AutoReg do not like dataframes.. cast it to tuple to remove warnings
		# train autoregression
		model = AutoReg(X, lags=lag_win, old_names=False)
		model_fit = model.fit()
		# Predict future data, and use it asnew data
		predictions = model_fit.predict(start=len(X)-1, end=len(X)*2-1, dynamic=False)
		# Save prediction
		generated.append(predictions)
	
	# Create folder if necessary
	folder = extract_folder(args.output_path)
	os.system(f"mkdir -p {folder}")
	
	pd.DataFrame(np.array(generated).T).to_csv(args.output_path, index=False)
	print("Autoregressive model generation terminated")



def extract_folder(filename):
	b_spl = filename.split("/")
	if (len(b_spl) <= 1): 
		return ""
	# Return path witout file
	return filename.replace(b_spl[len(b_spl) - 1], "")


if __name__ == "__main__":
	# Ev. extract parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default=None, help='Path to the input dataset')
	parser.add_argument('--lag_window', type=int, default=0, help='Batch size')
	parser.add_argument('--output_path', type=str, default=None, help='The file where to store the generated data')
	args = parser.parse_args()
	
	main(args)
