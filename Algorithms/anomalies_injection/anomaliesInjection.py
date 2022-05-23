import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import argparse
import os

import agots.multivariate_generators.multivariate_extreme_outlier_generator as meog
import agots.multivariate_generators.multivariate_shift_outlier_generator as msog
import agots.multivariate_generators.multivariate_trend_outlier_generator as mtog
import agots.multivariate_generators.multivariate_variance_outlier_generator as mvog

def addAnomaly(dataframe, ts_number=0, anomaly_type='extreme', timestamps=[], factor=8):
#     assert anomaly_type in ['extreme', 'shift', 'trend', 'variance']
    if anomaly_type == 'extreme':
        generator = meog.MultivariateExtremeOutlierGenerator(timestamps=timestamps,factor=factor)
    elif anomaly_type == 'shift':
        generator = msog.MultivariateShiftOutlierGenerator(timestamps=timestamps,factor=factor)
    elif anomaly_type == 'trend':
        generator = mtog.MultivariateTrendOutlierGenerator(timestamps=timestamps,factor=factor)
    elif anomaly_type == 'variance':
        generator = mvog.MultivariateVarianceOutlierGenerator(timestamps=timestamps,factor=factor)
    else:
        raise WrongAnomalyTypeError('Anomaly type should be one between [\'extreme\', \'shift\', \'trend\', \'variance\']')
        
    outliers = generator.add_outliers(dataframe.iloc[:,ts_number])
    dataframe.iloc[:,ts_number] += outliers
    
    
def main(args):
	print("Hello, world!")
	
	# Name of the dataset (for quicker change)
	name = extract_name(args.dataset)

	dataset = args.dataset
	
	# The original dataset
	df = pd.read_csv(dataset, header=None)

	# Ev. create save folder
	os.system(f"mkdir -p ../../results/{name}/data")
	
	# Where to save the results (None if nowhere)
	save_path_results = f'../../results/{name}/data/{name}_AnomaliesInjection.csv'
	# save_path_results = None

	# Where to save the modifications (None if nowhere)
#	save_path_parameters = f'../../results/AnomaliesInjection/{name}/{name}_parameters.txt'
	save_path_parameters = None

	# Where to save the plots (None if nowhere)
#	save_path_plots = f'../../results/AnomaliesInjection/{name}/{name}_plots.pdf'
	save_path_plots = None

	# Number of anomalies to inject
	nb_modification = args.nb_modifications if args.nb_modifications != -1 else len(df.columns)

	# If True, multiple modifications can be applied to the same ts. If False, nb_modifications must not exceed the nb of ts in the dataset!
	can_multiple_modification = True if args.multiple_modification_per_ts == 1 else False

	# Minimal and maximal value for the factor parameter (for each anomaly type)
	factors_boundaries = {'extreme':(-args.extreme_factor,args.extreme_factor), 'shift': (-args.shift_factor,args.shift_factor), 'trend': (-args.trend_factor,args.trend_factor), 'variance': (-args.variance_factor,args.variance_factor)}

	# Max number of extreme point to inject for each 'extreme' anomaly
	max_nb_extreme = args.max_nb_extreme

	# Minimal and maximal length of the shift for each 'shift' anomaly
	min_max_shift = (args.min_shift, args.max_shift)

	# Minimal and maximal length of the trend for each 'trend' anomaly
	min_max_trend = (args.min_trend, args.max_trend)

	# Minimal and maximal length of the variance modification for each 'variance' anomaly
	min_max_variance = (args.min_variance, args.max_variance)

	# Probability for each anomaly type ([extreme, shift, trend, variance]). MUST sum to 1
	anomalies_probabilities_temp = [args.extreme_probability, args.shift_probability, args.trend_probability, args.variance_probability]
	# Ensure they sum up to 1
	total = sum(anomalies_probabilities_temp)
	if total == 0:
		anomalies_probabilities = [0.25, 0.25, 0.25, 0.25]
		print(f"All the given probabilities are 0, using default probabilities {anomalies_probabilities}")
	elif total != 1:
		anomalies_probabilities = [p / total for p in anomalies_probabilities_temp]
		print(f"Given prob: {anomalies_probabilities_temp}")
		print(f"Used prob: {anomalies_probabilities}")
	else:
		anomalies_probabilities = anomalies_probabilities_temp

	# Set a seed for the random generator (for reproducibility)
	random_seed = args.seed


	##################################################################################################################
	############################################# DO NOT EDIT AFTER HERE #############################################
	############################################# (except for plotting) ##############################################
	##################################################################################################################

	# Use a copy of the original data
	copy = df.copy()

	# Number of ts in the dataset
	nb_of_ts = len(copy.iloc[0])

	# Group the set of min_max length
	length_boundaries = {'shift': min_max_shift, 'trend': min_max_trend, 'variance': min_max_variance}

	# If only 1 modification per ts is allowed, ensure nb_modification do not exceede nb_of_ts
	if not can_multiple_modification:
		if nb_modification > nb_of_ts:
		    print(f"Only 1 modification per ts is allowed! Nb. of modification reduced from {nb_modification} to {nb_of_ts}")
		    nb_modification = nb_of_ts
		    
	# Set the seed 
	np.random.seed(random_seed)

	# Choose randomly the ts to modify
	ts_to_modify = np.random.choice(nb_of_ts, nb_modification, replace=can_multiple_modification) 

	# The possible anomaly type
	anomalies_type = ['extreme', 'shift', 'trend', 'variance']

	# Keep track of the modifications
	modifications = []

	# For each modification
	for ts_nb in ts_to_modify:
		# Choose anomaly type
		anomaly_type = np.random.choice(anomalies_type, p=anomalies_probabilities)
		# Choose anomaly factor
		fact = np.random.uniform(factors_boundaries[anomaly_type][0], factors_boundaries[anomaly_type][1])
		# Get the length of the ts
		ts_length = len(copy.iloc[:,ts_nb])
		# Compute missing parameters
		if anomaly_type == 'extreme':
		    # Choose nb of extreme values to inject
		    nb_extreme = np.random.randint(max_nb_extreme) + 1
		    # Choose the points where to insert the anomalies (of the form [(value1,),(value2,),...])
		    tstamps = [(x,) for x in np.random.choice(ts_length, nb_extreme, replace=False)]
		elif anomaly_type == 'shift' or anomaly_type == 'trend' or anomaly_type == 'variance':
		    # Choose the length of the anomaly (in the given range)
		    anomaly_length = np.random.randint(length_boundaries[anomaly_type][0],length_boundaries[anomaly_type][1])
		    # Choose the beginning of the anomaly (random from 0 to len(ts) - anomaly_length)
		    anomaly_init = np.random.randint(ts_length - anomaly_length)
		    # Create the timestamps
		    tstamps = [(anomaly_init, anomaly_init + anomaly_length)]
		# Insert the anomalies
		addAnomaly(copy, ts_nb, anomaly_type, tstamps, fact)
		# Keep track of the modifications
		modifications.append({'ts_index': ts_nb, 'anomaly_type': anomaly_type, 'timestamps': tstamps, 'factor': fact})

	# Sort the list of modifications by ts and print it
	sorted_modifications = sorted(modifications, key=lambda k: k['ts_index'])
		
	# Write the output data
	if save_path_results != None:
		copy.to_csv(save_path_results, header=False, index=False)
		
	# Write the list of modifications into a file
	if save_path_parameters != None:
		f=open(save_path_parameters,'w')
		f.write('seed: '+repr(random_seed)+'\n')
		for el in sorted_modifications:
		    f.write(repr(el)+'\n')
		f.close()
		
	# Turn interactive plotting off
	plt.ioff()

	fig_list = []
	for i in set(ts_to_modify):
		fig = plt.figure(figsize=(4,2))
		plt.plot(copy.iloc[:,i])
		plt.plot(df.iloc[:,i])
		plt.legend([repr(i)])
		fig_list.append(fig)
		plt.close(fig)
		
	# Save the plots to a pdf file
	if save_path_plots != None:
		pdf = matplotlib.backends.backend_pdf.PdfPages(save_path_plots)
		for fig in fig_list:
		    pdf.savefig( fig )
		pdf.close()
		
	print(f"New data saved at {save_path_results}")
	
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
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', type=str, default="../../datasets/Original_Data/BeetleFly_TEST.csv", help='Path to the input dataset')
	parser.add_argument('--nb_modifications', type=int, default=-1, help='Number of modifications. If -1, an avarage of 1 modification per time series is used')
	parser.add_argument('--multiple_modification_per_ts', type=int, default=1, help='If 1 (true), multiple modifications can be injected in a single time series. If 0 (false), nb_modifications should not be greater than the number of time series')
	parser.add_argument('--seed', type=int, default=123456789, help='Seed determining the random generation')
	parser.add_argument('--max_nb_extreme', type=int, default=5, help='Max number of extreme values (spikes) per anomaly')
	parser.add_argument('--min_shift', type=int, default=20, help='Min length of each shift anomaly')
	parser.add_argument('--max_shift', type=int, default=80, help='Max length of each shift anomaly')
	parser.add_argument('--min_trend', type=int, default=20, help='Min length of each trend anomaly')
	parser.add_argument('--max_trend', type=int, default=80, help='Max length of each trend anomaly')
	parser.add_argument('--min_variance', type=int, default=20, help='Min length of each variance anomaly')
	parser.add_argument('--max_variance', type=int, default=80, help='Max length of each variance anomaly')
	parser.add_argument('--extreme_factor', type=int, default=5, help='Max intensity of each extreme anomaly')
	parser.add_argument('--shift_factor', type=int, default=5, help='Max intensity of each shift anomaly')
	parser.add_argument('--trend_factor', type=int, default=1, help='Max intensity of each trend anomaly')
	parser.add_argument('--variance_factor', type=int, default=15, help='Max intensity of each variance anomaly')
	parser.add_argument('--extreme_probability', type=float, default=0.25, help='Probability that an anomaly is of type extreme')
	parser.add_argument('--shift_probability', type=float, default=0.25, help='Probability that an anomaly is of type shift')
	parser.add_argument('--trend_probability', type=float, default=0.25, help='Probability that an anomaly is of type trend')
	parser.add_argument('--variance_probability', type=float, default=0.25, help='Probability that an anomaly is of type variance')
	
    
	args = parser.parse_args()
    
	main(args)
		
