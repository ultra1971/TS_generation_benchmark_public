import os
import json
import pandas as pd
import argparse
import time


############################################################################################################################################################
################################################################       DBA       ###########################################################################
############################################################################################################################################################
# Run the java implementation of the DBA algorithm	
def run_DBA(para_dict):
	print("Running DBA")
	# Change working directory
	os.chdir("Algorithms/DBA")
	p = para_dict["dataset"]
	name = extract_name(p)
	data_path = f'../../{p}'
	# Run java file Dba.class
	command = f"java Dba {data_path}"
	# Create folder if necessary
	os.system(f"mkdir -p ../../results/{name}/data")
	# Generate data
	start_time = time.time()
	os.system(command)
	stop_time = time.time()
	# Return to main working directory
	os.chdir("../..")
	# Cut data
	cut_data(f"results/{name}/data/{name}_DBA.csv", length=para_dict["length"], nb_series=para_dict["nb_series"])
	# Show plot
	show = para_dict["Show_plot"]
	# Create plots
	start_time_plot = time.time()
	os.system(f"python3.6 utils/create_plots.py --dataset results/{name}/data/{name}_DBA.csv --output_path results/{name}/plots/{name}_DBA.png --show_plot {show}")
	stop_time_plot = time.time()
	time_plot = stop_time_plot - start_time_plot
	dba_time = stop_time - start_time
	# Compute metrics
	metrics_time = None
	if para_dict["Compute_metrics"] == 1:
		start_time_metrics = time.time()
		os.system(f"python3.6 utils/extract_metrics.py --original datasets/Original_Data/{name}.csv --generated results/{name}/data/{name}_DBA.csv --output_path results/{name}/precision.csv --algo DBA --is_dba 1 --normalized_gen 0")
		stop_time_metrics = time.time()
		metrics_time = stop_time_metrics - start_time_metrics
	# Save runtime stats
	ori_data = pd.read_csv(para_dict["dataset"])
	nb_ts = len(ori_data.columns)
	s = len(ori_data.iloc[:,0])
	save_runtime(path=f"results/{name}/runtime.csv", algorithm="DBA", nb_ts=nb_ts, ts_len=s, ds_name=f"{name}", runtime=dba_time, plot_time=time_plot, filter_time=None, metrics_time=metrics_time)


############################################################################################################################################################
########################################################       Anomalies Injection       ###################################################################
############################################################################################################################################################
# Run the anomalies injection algorithm, based on agots
def run_anomaliesInjection(para_dict):
	print("Running AnomaliesInjection")
	# Change working directory
	os.chdir("Algorithms/anomalies_injection")
	# Extract parameters
	p = para_dict["dataset"]
	data_path = f'../../{p}'
	name = extract_name(p)
	nb_modifications = para_dict["AnomaliesInjection_nb_modifications"]
	multiple_mods = para_dict["AnomaliesInjection_multiple_modification_per_ts"]
	seed = para_dict["AnomaliesInjection_seed"]
	max_nb_extreme = para_dict["AnomaliesInjection_max_nb_extreme"]
	min_shift = para_dict["AnomaliesInjection_min_shift"]
	max_shift = para_dict["AnomaliesInjection_max_shift"]
	min_trend = para_dict["AnomaliesInjection_min_trend"]
	max_trend = para_dict["AnomaliesInjection_max_trend"]
	min_variance = para_dict["AnomaliesInjection_min_variance"]
	max_variance = para_dict["AnomaliesInjection_max_variance"]
	extreme_factor = para_dict["AnomaliesInjection_extreme_factor"]
	shift_factor = para_dict["AnomaliesInjection_shift_factor"]
	trend_factor = para_dict["AnomaliesInjection_trend_factor"]
	variance_factor = para_dict["AnomaliesInjection_variance_factor"]
	extreme_p = para_dict["AnomaliesInjection_probability_extreme"]
	shift_p = para_dict["AnomaliesInjection_probability_shift"]
	trend_p = para_dict["AnomaliesInjection_probability_trend"]
	variance_p = para_dict["AnomaliesInjection_probability_variance"]
	# Run the AnomaliesInjection generation
	start_time = time.time()
	os.system(f"python3.6 anomaliesInjection.py --dataset {data_path} --nb_modifications {nb_modifications} --multiple_modification_per_ts {multiple_mods} --seed {seed} --max_nb_extreme {max_nb_extreme} --min_shift {min_shift} --max_shift {max_shift} --min_trend {min_trend} --max_trend {max_trend} --min_variance {min_variance} --max_variance {max_variance} --extreme_factor {extreme_factor} --shift_factor {shift_factor} --trend_factor {trend_factor} --variance_factor {variance_factor} --extreme_probability {extreme_p} --shift_probability {shift_p} --trend_probability {trend_p} --variance_probability {variance_p}")
	stop_time = time.time()
	# Return to main working directory
	os.chdir("../..")
	# Cut data
	cut_data(f"results/{name}/data/{name}_AnomaliesInjection.csv", length=para_dict["length"], nb_series=para_dict["nb_series"])
	# Show plot
	show = para_dict["Show_plot"]
	start_time_plot = time.time()
	os.system(f"python3.6 utils/create_plots.py --dataset results/{name}/data/{name}_AnomaliesInjection.csv --output_path results/{name}/plots/{name}_AnomaliesInjection.png --show_plot {show}")
	stop_time_plot = time.time()
	anomalies_injection_time = stop_time - start_time
	time_plot = stop_time_plot - start_time_plot
	# Compute metrics
	metrics_time = None
	if para_dict["Compute_metrics"] == 1:
		start_time_metrics = time.time()
		os.system(f"python3.6 utils/extract_metrics.py --original datasets/Original_Data/{name}.csv --generated results/{name}/data/{name}_AnomaliesInjection.csv --output_path results/{name}/precision.csv --algo AnomaliesInjection --is_dba 0 --normalized_gen 0")
		stop_time_metrics = time.time()
		metrics_time = stop_time_metrics - start_time_metrics
	# Save runtime stats
	ori_data = pd.read_csv(para_dict["dataset"])
	nb_ts = len(ori_data.columns)
	s = len(ori_data.iloc[:,0])
	save_runtime(path=f"results/{name}/runtime.csv", algorithm="Anomalies Injection", nb_ts=nb_ts, ts_len=s, ds_name=f"{name}", runtime=anomalies_injection_time, plot_time=time_plot, filter_time=None, metrics_time=metrics_time)
	
	
############################################################################################################################################################
##############################################################       InfoGAN       #########################################################################
############################################################################################################################################################
# Run the tsgen InfoGAN algorithm
def run_InfoGAN(para_dict):
	print("Running InfoGAN")
	# Change working directory
	os.chdir("Algorithms/tsgen")
	# Update the tsgen parameters file (easy way not to change the code)
	update_tsgen_parameters(para_dict)
	# Extract name
	name = extract_name(para_dict["dataset"])
	# Run java file Dba.class
	start_time = time.time()
	os.system(". venv/bin/activate; python3.6 main_tsgen.py; deactivate")
	stop_time = time.time()
	# Return to main working directory
	os.chdir("../..")
	# Copy file to correct folder
	corr_f = f"results/{name}/data"
	os.system(f"mkdir -p {corr_f}")
	os.system(f"cp Algorithms/tsgen/results/{name}/fake_long_complete.csv {corr_f}/{name}_InfoGAN.csv")
	# Cut data
	cut_data(f"results/{name}/data/{name}_InfoGAN.csv", length=para_dict["length"], nb_series=para_dict["nb_series"])
	# Ev. apply filter to smooth data
	with open("./Algorithms/tsgen/parameters.json", "r") as f:
		tsgen_dict = json.load(f)
		# Algorithm to use
		win = tsgen_dict["window"]
		train_ep = tsgen_dict["train_ep"]
	s = len(pd.read_csv(para_dict["dataset"]).iloc[:,0])
	savepath = f"{corr_f}/{name}_InfoGAN.csv"
	show = para_dict["Show_plot"]
	filter_time = None
	if (para_dict["Kalman_filter"] == 1):
		print("Applying kalman filter to InfoGAN results")
		remove_init = para_dict["Kalman_remove_initial"]
		start_time_filter = time.time()
		os.system(f"python3.6 utils/KalmanFilter/apply_kalman.py --dataset {savepath} --transition_covariance 0.05 --output_path {savepath} --remove_initial {remove_init}") # Override
		stop_time_filter = time.time()
		filter_time = stop_time_filter - start_time_filter
		# Create plot
		start_time_plot = time.time()
		os.system(f"python3.6 utils/create_plots.py --dataset {corr_f}/{name}_InfoGAN.csv --output_path results/{name}/plots/{name}_InfoGAN.png --show_plot {show}")
		stop_time_plot = time.time()
	else:
		start_time_plot = time.time()
		os.system(f"python3.6 utils/create_plots.py --dataset {corr_f}/{name}_InfoGAN.csv --output_path results/{name}/plots/{name}_InfoGAN.png --show_plot {show}")
		stop_time_plot = time.time()
	time_plot = stop_time_plot - start_time_plot
	infogan_time = stop_time - start_time
	# Compute metrics
	metrics_time = None
	if para_dict["Compute_metrics"] == 1:
		start_time_metrics = time.time()
		if para_dict["Kalman_filter"] == 1:
			os.system(f"python3.6 utils/extract_metrics.py --original datasets/Original_Data/{name}.csv --generated {corr_f}/{name}_InfoGAN.csv --output_path results/{name}/precision.csv --algo InfoGAN --is_dba 0 --normalized_gen 0")
		else:
			os.system(f"python3.6 utils/extract_metrics.py --original datasets/Original_Data/{name}.csv --generated {corr_f}/{name}_InfoGAN.csv --output_path results/{name}/precision.csv --algo InfoGAN --is_dba 0 --normalized_gen 0")
		stop_time_metrics = time.time()
		metrics_time = stop_time_metrics - start_time_metrics
	# Save runtime stats
	ori_data = pd.read_csv(para_dict["dataset"])
	nb_ts = len(ori_data.columns)
	save_runtime(path=f"results/{name}/runtime.csv", algorithm="InfoGAN", nb_ts=nb_ts, ts_len=s, ds_name=f"{name}", runtime=infogan_time, plot_time=time_plot, filter_time=filter_time, metrics_time=metrics_time, eps=train_ep, win_size=win)
	


############################################################################################################################################################
##############################################################       TimeGAN       #########################################################################
############################################################################################################################################################
# Run the TimeGAN algorithm (inside a venv, as it requires a different version of tensorflow...)
def run_TimeGAN(para_dict):
	print("Running TimeGAN")
	# Change working directory
	os.chdir("Algorithms/TimeGAN")
	# Extract needed parameters
	path = para_dict["dataset"]
	eps = para_dict["nb_epochs"]
	seq_len = para_dict["TimeGAN_seq_len"]
	batch_size = para_dict["batch_size"]
	# Construct running command
	command = f"python3.6 main_timegan.py --iteration {eps} --seq_len {seq_len} --batch_size {batch_size} --data_path ../../{path}"
	print(command)
	# Run venv
	start_time = time.time()
	os.system(f". venv/bin/activate; {command}; deactivate")
	stop_time = time.time()
	# Return to main working directory
	os.chdir("../..")
	# Create results folder if necessary
	name = extract_name(para_dict["dataset"])
	os.system(f"mkdir -p results/{name}/data")
	# Reconstruct csvs
	out_folder = f"Algorithms/TimeGAN/results/pieces"
	if (True):
		os.system(f"python3.6 utils/npy_to_csv.py --npy_path Algorithms/TimeGAN/results/{name}.npy --output_folder {out_folder}")
	# Concatenate
	if (True):
		size = len(pd.read_csv(para_dict["dataset"]).columns)
		ts_length = len(pd.read_csv(para_dict["dataset"]).iloc[:,0])
		long_out = f"Algorithms/TimeGAN/results/long_pieces"
		# Make sure the folder for longXY exist
		if not os.path.exists(long_out):
			os.makedirs(long_out)
		# Run concatenation script
		command = f"python3.6 Algorithms/tsgen/concat.py --ori_data {para_dict['dataset']} --fakes_folder {out_folder}/ --output_folder {long_out}/ --seq_len {seq_len} --top_n 10 --ts_len {ts_length} --plot 0 --gen_ts_len 1 --gen_ts_dim {size}"
		os.system(f". Algorithms/tsgen/venv/bin/activate; {command}; deactivate")
		# Copy result
		os.system(f"cp fake_long_complete.csv results/{name}/data/{name}_TimeGAN.csv")
		os.system("rm fake_long_complete.csv")
		#print("Long ts reconstructed")
	# Cut data
	cut_data(f"results/{name}/data/{name}_TimeGAN.csv", length=para_dict["length"], nb_series=para_dict["nb_series"])
	# Ev. apply filter to smooth data
	show = para_dict["Show_plot"]
	filter_time = None
	if (para_dict["Kalman_filter"] == 1):
#		print("Can not apply Kalman Filter to TimeGAN. Should implement automatic ts reconstruction (lsh) before!")
		savepath = f"results/{name}/data/{name}_TimeGAN.csv"
		print("Applying kalman filter to TimeGAN results")
		remove_init = para_dict["Kalman_remove_initial"]
		start_time_filter = time.time()
		os.system(f"python3.6 utils/KalmanFilter/apply_kalman.py --dataset {savepath} --transition_covariance 0.05 --output_path {savepath} --remove_initial {remove_init}") # Override
		stop_time_filter = time.time()
		filter_time = stop_time_filter - start_time_filter
		# Show plot
		start_time_plot = time.time()
		os.system(f"python3.6 utils/create_plots.py --dataset results/{name}/data/{name}_TimeGAN.csv --output_path results/{name}/plots/{name}_TimeGAN.png --show_plot {show}")
		stop_time_plot = time.time()
	else:
		start_time_plot = time.time()
		os.system(f"python3.6 utils/create_plots.py --dataset results/{name}/data/{name}_TimeGAN.csv --output_path results/{name}/plots/{name}_TimeGAN.png --show_plot {show}")
		stop_time_plot = time.time()
	timegan_time = stop_time - start_time
	time_plot = stop_time_plot - start_time_plot
	# Compute metrics
	metrics_time = None
	if para_dict["Compute_metrics"] == 1:
		start_time_metrics = time.time()
		if para_dict["Kalman_filter"] == 1:
			os.system(f"python3.6 utils/extract_metrics.py --original datasets/Original_Data/{name}.csv --generated results/{name}/data/{name}_TimeGAN.csv --output_path results/{name}/precision.csv --algo TimeGAN --is_dba 0 --normalized_gen 0") # The generated data is normalized, but not in the range [0-1]!
		else:
			os.system(f"python3.6 utils/extract_metrics.py --original datasets/Original_Data/{name}.csv --generated results/{name}/data/{name}_TimeGAN.csv --output_path results/{name}/precision.csv --algo TimeGAN --is_dba 0 --normalized_gen 0")
		stop_time_metrics = time.time()
		metrics_time = stop_time_metrics - start_time_metrics
	# Save runtime stats
	ori_data = pd.read_csv(para_dict["dataset"])
	nb_ts = len(ori_data.columns)
	s = len(ori_data.iloc[:,0])
	save_runtime(path=f"results/{name}/runtime.csv", algorithm="TimeGAN", nb_ts=nb_ts, ts_len=s, ds_name=f"{name}", runtime=timegan_time, plot_time=time_plot, filter_time=filter_time, metrics_time=metrics_time, eps=eps, win_size=seq_len)
	# Delete temp folder in TimeGAN! Otherwise will be (partially) re-used next time
	os.system("rm -r Algorithms/TimeGAN/results")
	

############################################################################################################################################################
##########################################################       AutoregressiveModel     ###################################################################
############################################################################################################################################################
def run_AR(para_dict):
	print("Running AutoregressiveModel")
	# Change working directory
	os.chdir("Algorithms/AR")
	# Extract dataset
	path = para_dict["dataset"]
	# Extract name
	name = extract_name(para_dict["dataset"])
	# Extract lag window
	l_win = para_dict["AR_lag_window"] # if para_dict["AR_lag_window"] > 0 else None
	# Run model
	start_time = time.time()
	os.system(f"python3.6 AutoregressiveModel.py --dataset ../../{path} --output_path ../../results/{name}/data/{name}_AR.csv --lag_window {l_win}")
	stop_time = time.time()
	# Return to main working directory
	os.chdir("../..")
	# Cut data
	cut_data(f"results/{name}/data/{name}_AR.csv", length=para_dict["length"], nb_series=para_dict["nb_series"])
	# Show plot
	show = para_dict["Show_plot"]
	start_time_plot = time.time()
	os.system(f"python3.6 utils/create_plots.py --dataset results/{name}/data/{name}_AR.csv --output_path results/{name}/plots/{name}_AR.png --show_plot {show}")
	stop_time_plot = time.time()
	# Compute metrics
	start_time_metrics = time.time()
	os.system(f"python3.6 utils/extract_metrics.py --original datasets/Original_Data/{name}.csv --generated results/{name}/data/{name}_AR.csv --output_path results/{name}/precision.csv --algo AR --is_dba 0 --normalized_gen 0")
	stop_time_metrics = time.time()
	# Save runtime stats
	ori_data = pd.read_csv(para_dict["dataset"])
	nb_ts = len(ori_data.columns)
	s = len(ori_data.iloc[:,0])
	ar_time = stop_time - start_time
	time_plot = stop_time_plot - start_time_plot
	metrics_time = stop_time_metrics - start_time_metrics
	save_runtime(path=f"results/{name}/runtime.csv", algorithm="AR", nb_ts=nb_ts, ts_len=s, ds_name=f"{name}", runtime=ar_time, plot_time=time_plot, metrics_time=metrics_time)
	
	
############################################################################################################################################################
############################################################       BasicGAN3072       ######################################################################
############################################################################################################################################################
def run_BasicGAN3072(para_dict):
	print("hello")
	# Change working directory
	os.chdir("Algorithms/BasicGAN_3072")
	# Check shape
	d = para_dict["dataset"]
	df = pd.read_csv(f"../../{d}", header=None)
	if df.shape != (3072, 3072):
		print(f"Dataset has wrong shape! Expected (3072, 3072), found {df.shape}")
		return
	# Copy dataset in working folder (with correct name)
	os.system(f"cp ../../{d} column_23_3072_3072.txt")
	# Gen nb epochs
	num_epochs = para_dict["nb_epochs"]
	# Run the 3 needed script
#	os.system(f"python3.6 DCGAN.py --nb_epochs {num_epochs}")
#	os.system(f"python3.6 encoder_dc.py --nb_epochs {num_epochs}")
#	os.system(f"python3.6 test_dc.py")
	# Create results folder if necessary
	name = extract_name(para_dict["dataset"])
	out_path = f"../../results/BasicGAN3072/{name}/ep={num_epochs}"
	os.system(f"mkdir -p {out_path}")
	# Copy result to results folder
	out = pd.read_csv(f"fake_noise_23_raw_f.txt", header=None)
	out.to_csv(f"{out_path}/{name}.csv", header=False, index=False)
	# Return to main working directory
	os.chdir("..")
	os.chdir("..")





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
	
	
def update_tsgen_parameters(para_dict):
	# Open the parameters file
	jsonFile = open("parameters.json", "r")
	# Read the json into the buffer
	data = json.load(jsonFile)
	# Close the file
	jsonFile.close()
	
	# Change data
	inp_path = para_dict["dataset"]
	inp_path_correct = f"../../{inp_path}"
	data["input_data_path"] = [inp_path_correct]
	data["train_ep"] = para_dict["nb_epochs"]
	data["batch_size_train"] = para_dict["batch_size"]
	data["train"] = 1
	data["generate"] = 1
	
	# Reopen the file in writing mode
	jsonFile = open("parameters.json", "w+")
	# Override the file
	jsonFile.write(json.dumps(data, indent=4))
	# Close the file
	jsonFile.close()
	print("tsgen/paramenters.json correctly overridden!")


def write_params(args):
	# Open the parameters file
	jsonFile = open("parameters.json", "r")
	# Read the json into the buffer
	data = json.load(jsonFile)
	# Close the file
	jsonFile.close()
	
	# Change the input
	if args.dataset is not None:
		data["dataset"] = args.dataset
		print("Dataset set")
	# Change the algo
	if args.algorithm != []:
		data["algorithm"] = args.algorithm
		print("Algorithm set")
	if args.length > 0:
		data["length"] = args.length
		print("Length set")
	if args.nb_series > 0:
		data["nb_series"] = args.nb_series
		print("Number of series set")
	# Change the nb of epochs
	if args.nb_epochs > 0:
		data["nb_epochs"] = args.nb_epochs
		print("Nb epochs set")
	# Change the batch size
	if args.batch_size > 0:
		data["batch_size"] = args.batch_size
		print("Batch size set")
	if args.TimeGAN_seq_len > 0:
		data["TimeGAN_seq_len"] = args.TimeGAN_seq_len
		print("TimeGAN seq len set")
	# Change Kalman_filter
	if args.Kalman_filter is not None:
		data["Kalman_filter"] = args.Kalman_filter
		print("Kalman filter set")
	# Change Compute_metrics
	if args.Compute_metrics is not None:
		data["Compute_metrics"] = args.Compute_metrics
		print("Compute metrics set")
	# Change Show_plot
	if args.Show_plot is not None:
		data["Show_plot"] = args.Show_plot
		print("Show plot set")
	# Change AR_lag_window
	if args.lag_window is not None:
		data["AR_lag_window"] = args.lag_window
		print("Lag window set")
	
	# Reopen the file in writing mode
	jsonFile = open("parameters.json", "w+")
	# Override the file
	jsonFile.write(json.dumps(data, indent=4))
	# Close the file
	jsonFile.close()

def save_runtime(path, algorithm, nb_ts, ts_len, ds_name, runtime, plot_time=None, filter_time=None, metrics_time=None, eps=None, win_size=None):

	# Write to the output file
	if path is not None:
		# Ev. create the file
		f_t = None if filter_time is None else '{:.4f}'.format(filter_time)
		p_t = None if plot_time is None else '{:.4f}'.format(plot_time)
		m_t= None if metrics_time is None else '{:.4f}'.format(metrics_time)
		new_line = [
			algorithm,															# Algorithm
			ds_name,															# Dataset name
			nb_ts,																# Number of ts in the original dataset
			ts_len,																# Length of each ts in the original dataset
			eps,																# Number of epochs
			'{:.4f}'.format(runtime),											# Generation time
			f_t,																# Kalman filter time
			p_t,																# Plotting time
			m_t,																# Metrics time 
		]
		cols_name = ["Algorithm", "Dataset", "Number_of_ts_original", "Length_of_ts_original", "Epochs", "Generation_time", "Kalman_filter_time", "Plotting_time", "Metrics_time"]
		if os.path.isfile(path):
			data = pd.read_csv(path, index_col=[0], float_precision='round_trip')
		else:
			data = pd.DataFrame(columns=cols_name)
		# Check if data already contains a line for the given algorithm. If yes, override it
		if len(data.loc[data["Algorithm"]==algorithm]) > 0: 
			data = data.loc[data["Algorithm"]!=algorithm]
		# Add new data
		data = data.append({i:j for i,j in zip(cols_name, new_line)}, ignore_index=True)
		# Save data
		data.to_csv(path)



def cut_data(path, length, nb_series):
	df = pd.read_csv(path)
	# Ev. cut series
	if length < len(df) and length > 0: 
		df = df.iloc[:length,:]
		new_l = length
	else:
		new_l = len(df)
	# Ev. drop columns
	if nb_series < len(df.columns) and nb_series > 0: 
		df = df.iloc[:,:nb_series]
		new_nb = nb_series
	else:
		new_nb = len(df.columns)
	df.to_csv(path, index=False, header=False)
	print(f"Data reshaped into {new_nb}x{new_l}")

############################################################################################################################################################
################################################################       Main       ##########################################################################
############################################################################################################################################################
if __name__ == "__main__":
	# Ev. extract parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--algorithm', default=[], nargs='*', help='Algorithm(s) to use')
	parser.add_argument('--dataset', type=str, default=None, help='Path to the input dataset')
	parser.add_argument('--length', type=int, default=0, help='Max length of the generated time series')
	parser.add_argument('--nb_series', type=int, default=0, help='Number of time series to generate')
	parser.add_argument('--nb_epochs', type=int, default=0, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, default=0, help='Batch size')
	parser.add_argument('--TimeGAN_seq_len', type=int, default=0, help='Sequence length for TimeGAN generated data')
	parser.add_argument('--Kalman_filter', type=int, default=None, help='Whether or not apply Kalman filter')
	parser.add_argument('--Compute_metrics', type=int, default=None, help='Whether or not compute/extract metrics')
	parser.add_argument('--Show_plot', type=int, default=None, help='Whether or not show the plot of the generated data')
	parser.add_argument('--lag_window', type=int, default=None, help='Lag window for AR. 0 for default')
	args = parser.parse_args()
	write_params(args)
	
	# load parameters
	with open("./parameters.json", "r") as f:
		para_dict = json.load(f)
		# Algorithm to use
		algo = para_dict["algorithm"]
		print(f"Using algorithm {algo}")
		print(f"Using dataset {para_dict['dataset']}")
		if "DBA" in algo:
			run_DBA(para_dict)
		if "InfoGAN" in algo:	
			run_InfoGAN(para_dict)
		if "AnomaliesInjection" in algo:
			run_anomaliesInjection(para_dict)
		if "TimeGAN" in algo:
			run_TimeGAN(para_dict)
		if "AR" in algo:
			run_AR(para_dict)
		if "BasicGAN3072" in algo:
			run_BasicGAN3072(para_dict)
		# Check if there are wrong values
		if not set(algo).issubset(set(["DBA","InfoGAN","TimeGAN","AnomaliesInjection", "AR", "BasicGAN3072"])):
			print("At least one of the specified algorithm is not available!")
			print(f'Algorithms {set(algo).difference(set(["DBA","InfoGAN","TimeGAN","AnomaliesInjection","AR"]))} not available')
			
	print("Finish!")
			
		


