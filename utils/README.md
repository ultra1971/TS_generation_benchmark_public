# Time Series Generation Benchmark - utils

This folder contains a set of utils used in the benchmark.

[**Kalman Filter**](#kalman-filter) | [**Create Plots**](#create-plots) | [**Extract Metrics**](#extract-metrics) | [**Npy to Csv**](#npy-to-csv) | [**Plot Metrics**](#plot-metrics) | [**Plot Distribution**](#plot-distribution) | [**Plot Autocorrelation**](#plot-autocorrelation) | [**Images**](#images)
___

## Kalman Filter


Apply Kalman filter on given dataset.

Example: 
```
$ python3.6 KalmanFilter/apply_kalman.py --dataset ../results/BeetleFly_TEST/data/BeetleFly_TEST_TimeGAN.csv --transition_covariance 0.05 --output_path ../results/BeetleFly_TEST/data/BeetleFly_TEST_TimeGAN_filtered.csv --remove_initial 20
```
&rarr; Apply filter to "BeetleFly\_TEST\_TimeGAN.csv", with a transition covariance of 0.05.  
&rarr; Then remove the first 20 points of each time series, and save result as "BeetleFly\_TEST\_TimeGAN\_filtered.csv".
___

## Create Plots

Plot given dataset.

Example:
```
$ python3.6 create_plots.py --dataset ../results/BeetleFly_TEST/data/BeetleFly_TEST_TimeGAN.csv --output_path ../results/BeetleFly_TEST/plots/BeetleFly_TEST_TimeGAN.png --show_plot 0
```
&rarr; Create plot of dataset "BeetleFly\_TEST\_TimeGAN.csv" and save it as "BeetleFly\_TEST\_TimeGAN.png".  
&rarr; Do not show it directly, only save it to the output file.
___

## Extract Metrics

Extract metrics from a given dataset and save them to a file. If this file do not exist yet, create it.

Example:
```
$ python3.6 extract_metrics.py --original ../datasets/Original_Data/BeetleFly_TEST.csv --generated ../results/BeetleFly_TEST/data/BeetleFly_TEST_TimeGAN.csv --output_path ../results/BeetleFly_TEST/precision.csv --algo TimeGAN --is_dba 0 --normalized_gen 0
```
&rarr; Extract metrics from file "BeetleFly\_TEST\_TimeGAN.csv" (and original file "BeetleFly\_TEST.csv") and save them to the file "precision.csv".  
&rarr; Specify that the algorithm considered is "TimeGAN" (used as "algorithm" value in the "precision.csv" file).  
&rarr; Metrics from data created with DBA are extracted differently (as there is not a 1-to-1 match between original and generated data).  
&rarr; Specify that the generated data has not yet been normalized between 0 and 1.
___

## Npy to Csv

TimeGAN implementation save the data as ".npy". Convert it to ".csv".

Example:
```
$ python3.6 npy_to_csv.py --npy_path ../Algorithms/TimeGAN/results/BeetleFly.npy --output_folder ../Algorithms/TimeGAN/results/pieces
```
&rarr; Extract data in "BeetleFly.npy" to multiple ".csv" files and save them in folder "TimeGAN/results/pieces".  
&rarr; It is now possible to apply a reconstruction algorithm to rebuild this pieces into longet time series.
___

## Plot Metrics

Plot metrics from a "precision.csv" file.

Example:
```
$ sh plot_metrics.sh
```
&rarr; Read files "temp\_mean.csv" "temp\_variance.csv" "temp\_entropy.csv" "temp\_mi.csv" and "temp\_corr.csv" and plot them using _GNUPlot_.  
&rarr; This script is not intended to be called directly. One should rather use the "plot\_metrics.py" in the main folder, which takes care of creating the necessary temporary files.
___

## Plot Distribution

Plot distribution of a given dataset

Example:
```
$ python3.6 plot_distribution.py --dataset ../results/BeetleFly_TEST/data/BeetleFly_TEST_InfoGAN.csv --output_path ./InfoGAN.png --nbins 30 --replace 1 --title "InfoGAN distribution" --individually 0
```
&rarr; Read file "BeetleFly\_TEST\_InfoGAN.csv" and plot the histograms with the density of the time series it contains.  
&rarr; Save the results to the file "InfoGAN.png". If it already exist, override it.  
&rarr; Use 30 bins for each histogram.  
&rarr; Set the title of the output image as "InfoGAN density".  
&rarr; Plot a single plot with the distribution of the entire dataset, instead of a plot for each time series.
___

## Plot Autocorrelation

Example:
```
$ python3.6 plot_autocorrelation.py --dataset ../datasets/Original_Data/BeetleFly_TEST.csv --output_path autocorrelation_BeetleFly.png --index all --title "Autocorrelation BeetleFly" --replace 0 --maxlags 100
```
&rarr; Plot the average autocorrelation function of all the time series in "BeetleFly\_TEST.csv".  
&rarr; Save the result as "autocorrelation\_BeetleFly.png". If this file already exists, do nothing.  
&rarr; Set the title of the output image as "Autocorrelation BeetleFly".  
&rarr; For the autocorrelation, consider 100 points.  
___

## Images

This folder contains images for the README file(s)
