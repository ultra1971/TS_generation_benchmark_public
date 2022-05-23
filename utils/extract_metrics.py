import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
#from dtaidistance import dtw
from sklearn.metrics import normalized_mutual_info_score as nmi, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import antropy as ant
from scipy.stats import pearsonr
import os


def main(args):
	# Load the original data
	original_df = pd.read_csv(args.original)
	# Load the generated data
	generated_df = pd.read_csv(args.generated, header=None)
	# Drop columns with NaN
	original_df = original_df.dropna(axis=1)
	generated_df_not_norm = generated_df.dropna(axis=1)
	# Some generation algorithms produce data between 0 and 1. Rescale original data in other case
	original_df_corrected = normalize_0_1(original_df)
	norm_gen = (args.normalized_gen == 0)
	if norm_gen:
		generated_df = normalize_0_1(generated_df_not_norm)
	else:
		generated_df = generated_df_not_norm
	# MI accepts only integers. 
	precision = 1e5
	original_df_int = (original_df * precision).astype('int64')
	generated_df_int = (generated_df * precision).astype('int64')
	original_df_corrected_int = (original_df_corrected * precision).astype('int64')
	# Extract values
	# Mean
	ori_means = np.mean(original_df_corrected)
	ori_not_norm_means = np.mean(original_df)
	gen_means = np.mean(generated_df)
	if norm_gen: gen_means_not_norm = np.mean(generated_df_not_norm)
	# Variance
	ori_vars = np.var(np.array(original_df_corrected))
	ori_not_norm_vars = np.var(np.array(original_df))
	gen_vars = np.var(generated_df)
	if norm_gen: gen_vars_not_norm = np.var(generated_df_not_norm)
	# Spectral entropy
	spec_func = lambda x: np.nan_to_num(ant.spectral_entropy(x, sf=100, normalize=True, method='fft'))            # nan_to_num to avoid nan (-> 0) or INF (-> very large number)
	ori_entropies = np.array([spec_func(original_df_corrected.loc[:,c]) for c in original_df_corrected.columns])
	gen_entropies = np.array([spec_func(generated_df.loc[:,c]) for c in generated_df.columns])
	# Correlation function
	corr_funct = lambda a,b: pearsonr(a,b)[0]
	min_len = min(len(original_df_corrected.iloc[:,0]), len(generated_df.iloc[:,0]))
	print(f"Considering first {min_len} points for metrics extraction")
	if args.is_dba == 0:
		# Mutual information (normalized)
		mutual_infos = [nmi(original_df_corrected_int.iloc[:min_len,c], generated_df_int.iloc[:min_len,c]) for c in range(min(len(original_df_corrected_int.columns),len(generated_df_int.columns)))]
		# Pearsons correlation
		correlations = [corr_funct(original_df_corrected.iloc[:min_len,c], generated_df.iloc[:min_len,c]) for c in range(min(len(original_df_corrected_int.columns),len(generated_df_int.columns)))]
		# RMSE
		RMSE = [np.sqrt(mean_squared_error(original_df_corrected.iloc[:min_len,c], generated_df.iloc[:min_len,c])) for c in range(min(len(original_df_corrected_int.columns),len(generated_df_int.columns)))]
	else:
		# Pearsons correlation
		correlations = []
		for col_name in generated_df:
			possibles = compare_ts_with_dataset(corr_funct, generated_df.iloc[:min_len,col_name], original_df_corrected.iloc[:min_len,:])
			correlations.append(max(possibles))
		# Mutual information (normalized)
		mutual_infos = []
		for col_name in generated_df:
			possibles = compare_ts_with_dataset(nmi, generated_df_int.iloc[:min_len,col_name], original_df_corrected_int.iloc[:min_len,:])
			mutual_infos.append(max(possibles))
		# RMSE
		RMSE = []
		for col_name in generated_df:
			possibles = compare_ts_with_dataset(mean_squared_error, generated_df.iloc[:min_len,col_name], original_df_corrected.iloc[:min_len,:])
			RMSE.append(np.sqrt(max(possibles)))
			
	# Write to the output file
	if args.output_path is not None:
		# Ev. create the file
		new_line = [
			args.algo,											# Algorithm
			'{:.6f}'.format(np.mean(ori_means)),				# Mean original
			'{:.6f}'.format(np.mean(ori_not_norm_means)),		# Mean original normalized
			'%.6f' % np.mean(gen_means),						# Mean generated
			'{:.6f}'.format(np.mean(ori_vars)),					# Variance original
			'{:.6f}'.format(np.mean(ori_not_norm_vars)),		# Variance original normalized
			'{:.6f}'.format(np.mean(gen_vars)),					# Variance generated
			'{:.6f}'.format(np.mean(ori_entropies)),			# Entropy original
			'{:.6f}'.format(np.mean(gen_entropies)),			# Entropy generated
			'{:.6f}'.format(np.mean(mutual_infos)),				# Mutual information
			'{:.6f}'.format(np.mean(correlations)),				# Correlation
			'{:.6f}'.format(np.mean(RMSE))						# RMSE
		]
		cols_name = ["Algorithm", "Mean_original", "Mean_original_not_normalized", "Mean_generated", "Variance_original", "Variance_original_not_normalized", "Variance_generated", "Entropy_original", "Entropy_generated", "Mutual_information", "Correlation", "RMSE"]
		if os.path.isfile(args.output_path):
			data = pd.read_csv(args.output_path, index_col=[0], float_precision='round_trip')
		else:
			data = pd.DataFrame(columns=cols_name)
		# Check if data already contains a line for the given algorithm. If yes, override it
		if len(data.loc[data["Algorithm"]==args.algo]) > 0: data = data.loc[data["Algorithm"]!=args.algo]
		# Add new data
		data = data.append({i:j for i,j in zip(cols_name, new_line)}, ignore_index=True)
		# Save data
		data.to_csv(args.output_path)
	
			
			
	# Save to file
#	foldername = extract_folder(args.generated)
	# Create folder if necessary
#	os.system(f"mkdir -p {foldername}metrics")
#	gen_means_not_norm_results = f" (not normalized: {'{:.6f}'.format(np.mean(gen_means_not_norm))})" if norm_gen else ""
#	gen_vars_not_norm_results = f" (not normalized: {'{:.6f}'.format(np.mean(gen_vars_not_norm))})" if norm_gen else ""
	# Open/create file
#	with open(f'{foldername}metrics/metrics.txt', 'w+') as f:
#		f.write(f"Mean original:         {'{:.6f}'.format(np.mean(ori_means))} (not normalized: {'{:.6f}'.format(np.mean(ori_not_norm_means))})\n")
#		f.write(f"Mean generated:        {'%.6f' % np.mean(gen_means)}{gen_means_not_norm_results}\n")
#		f.write("\n")
#		f.write(f"Variance original:     {'{:.6f}'.format(np.mean(ori_vars))} (not normalized: {'{:.6f}'.format(np.mean(ori_not_norm_vars))})\n")
#		f.write(f"Variance generated:    {'{:.6f}'.format(np.mean(gen_vars))}{gen_vars_not_norm_results}\n")
#		f.write("\n")
#		f.write(f"Entropy original:      {'{:.6f}'.format(np.mean(ori_entropies))}\n")
#		f.write(f"Entropy generated:     {'{:.6f}'.format(np.mean(gen_entropies))}\n")
#		f.write("\n")
#		f.write(f"Mutual information:    {'{:.6f}'.format(np.mean(mutual_infos))}\n")
#		f.write("\n")
#		f.write(f"Correlation:           {'{:.6f}'.format(np.mean(correlations))}\n")
#		f.write("\n")
#		f.write(f"DTW distance:          {'{:.6f}'.format(np.mean(dtw_distances))}\n")
#	f.close()


# Execute function with a single ts and a dataset
def compare_ts_with_dataset(funct, single_ts, dataset):
	results = []
	for col_name in dataset.columns:
		results.append(funct(dataset.loc[:,col_name], single_ts))
	return results


# Normalize a dataset to values between 0 and 1
def normalize_0_1(series):
    # prepare data for normalization
    normalized = pd.DataFrame()
    for col in series.columns:
        values = series.loc[:,col].values
        values = values.reshape((len(values), 1))
        # train the normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(values)
#         print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        # normalize the dataset and print the first 5 rows
        normalized.loc[:,col] = scaler.transform(values).tolist()
    # First columns appears to have elements in a single-value array...
    normalized.iloc[:,0] = [x[0] for x in normalized.iloc[:,0]]
    return normalized
    
    
def extract_folder(filename):
	b_spl = filename.split("/")
	if (len(b_spl) <= 1): 
		return ""
	# Return path witout file
	return filename.replace(b_spl[len(b_spl) - 1], "")
	

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--original', type=str, default="datasets/Original_Data/Coffee_TEST.csv", help='Path to the original dataset')
	parser.add_argument('--generated', type=str, default="results/DBA/Coffee_TEST/Coffee_TEST.csv", help='Path to the generated dataset')
	parser.add_argument('--normalized_gen', type=int, default=0, help='Wether or not the generated data is normalized between 0 and 1')
	parser.add_argument('--output_path', type=str, default=None, help='Path to the metrics file')
	parser.add_argument('--algo', type=str, default="NotSpecifiedAlgo", help='The algorithm considered')
	parser.add_argument('--is_dba', type=int, default=0, help='Wether or not the generating algorithm is DBA')

    
	args = parser.parse_args()
    
	main(args)
