import argparse
import os
import pandas as pd

def create_dataframe(df, folder_name, original_col, generated_col):
	f = folder_name
	tmp = pd.read_csv(f"results/{f}/precision.csv")
	# Create row if necessary
	if f not in df.loc[:,"Dataset"]:
		df = df.append({"Dataset":f}, ignore_index=True)
	# Add original
	if original_col is not None:
		df.loc[df['Dataset'] == f, 'Original'] = float(tmp.loc[0,original_col])
	# For each algorithm
	for algo in tmp.loc[:,"Algorithm"]:
		# Add column if necessary
		if algo not in df.columns:
			df[algo] = ""
		# Select row
		row = tmp.loc[tmp['Algorithm'] == algo]
		# Add mean
		df.loc[df['Dataset'] == f, algo] = float(row[generated_col])          # !!!!!! Cast to value (float), otherwise problems (why?...)
	return df



def main(args):
	if args.dataset == []:													# By default, use all directories in "results"
		folders = os.listdir("results")
		folders = [f for f in folders if not '.' in f]						# Keep only element without '.' (directories)
	else:
		folders = [extract_name(d) for d in args.dataset if os.path.isdir(d)]		     	# Keep only directories
		
	
	
	# Print directories used
#	for folder in folders:
#		print(extract_name(folder))
		
	# Create dfs for temp files
	tmp_mean = pd.DataFrame()
	tmp_mean['Dataset'] = ""
	tmp_variance = pd.DataFrame()
	tmp_variance['Dataset'] = ""
	tmp_entropy = pd.DataFrame()
	tmp_entropy['Dataset'] = ""
	tmp_mi = pd.DataFrame()
	tmp_mi['Dataset'] = ""
	tmp_corr = pd.DataFrame()
	tmp_corr['Dataset'] = ""
	tmp_rmse = pd.DataFrame()
	tmp_rmse['Dataset'] = ""

	# For each dataset
	for f in folders:
		tmp_mean = create_dataframe(tmp_mean, f, 'Mean_original', 'Mean_generated')
		tmp_variance = create_dataframe(tmp_variance, f, 'Variance_original', 'Variance_generated')
		tmp_entropy = create_dataframe(tmp_entropy, f, 'Entropy_original', 'Entropy_generated')
		tmp_mi = create_dataframe(tmp_mi, f, None, 'Mutual_information')
		tmp_corr = create_dataframe(tmp_corr, f, None, 'Correlation')
		tmp_rmse = create_dataframe(tmp_rmse, f, None, 'RMSE')
		


	# Save temp files
	tmp_mean.to_csv('utils/temp_mean.csv', index=False)
	tmp_variance.to_csv('utils/temp_variance.csv', index=False)
	tmp_entropy.to_csv('utils/temp_entropy.csv', index=False)
	tmp_mi.to_csv('utils/temp_mi.csv', index=False)
	tmp_corr.to_csv('utils/temp_corr.csv', index=False)
	tmp_rmse.to_csv('utils/temp_rmse.csv', index=False)
		
	
	# Run gnuplot bash
	os.system("sh utils/plot_metrics.sh")
	
	
	# Remove temp files
	os.remove('utils/temp_mean.csv')
	os.remove('utils/temp_variance.csv')
	os.remove('utils/temp_entropy.csv')
	os.remove('utils/temp_mi.csv')
	os.remove('utils/temp_corr.csv')
	os.remove('utils/temp_rmse.csv')


def extract_name(path):
	if path[-1] == "/":
		path = path[:-1]
	# Remove ev. folder path
	b_spl = path.split("/")
	name = b_spl[len(b_spl) - 1]
	return name

if __name__ == "__main__":
	# Ev. extract parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default=[], nargs='*', help='Path to the folder(s) containing the "precision.csv" file. Use "results/*" for all')
	args = parser.parse_args()
	
	main(args)
			
	print("Finish!")
