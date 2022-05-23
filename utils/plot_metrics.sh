#!/bin/bash



for MET in "mean" "variance" "entropy" "mi" "corr" "rmse"
do
	TITLE=$(echo "$MET" | sed 's/.*/\u&/')

	# Count number of columns (= # of algorithms + 1) and store it
	NBCOLS=$(head -1 utils/temp_$MET.csv | sed 's/[^,]//g' | wc -c)
	
	# Plot - temp
	gnuplot -e "set style data histogram;    										`# Set plot type`				\
				set style fill solid;        										`# Set plot style`				\
				set key font ',19';							                        `# Set font size`				\
				set xtics font ',19';						                        `# Set font size`				\
				set xtics rotate by 12 right;										`# Rotate x labels`				\
				set tmargin at screen 0.85; 										`# Set top starting point`		\
				set bmargin at screen 0.15;  										`# Add space bottom`			\
				set rmargin at screen 0.0;											`# Set left starting point`		\
				set rmargin at screen 0.85;											`# Add space right`				\
				set xtics center offset 0.2,-2;										`# Move x labels position`		\
				set key noenhanced; set xtic noenhanced;         					`# Allow underscore in title`	\
				set key box autotitle columnhead; 									`# Set first column as key`		\
				set key right out opaque; 											`# Set legend position`			\
				set border back; 													`# Put border behind legend`	\
				set datafile sep ',';        										`# Indicate file separator`		\
				set title '$TITLE' font ',25';           							`# Set plot title`				\
				plot for [i=2:$NBCOLS] './utils/temp_$MET.csv' using i:xtic(1);		`# Plotting command`			\
				pause -1"                    										# Wait until user closes the image

done		

echo "Goodbye!"



			
			

# set yrange [0:1];                                              		`# Set y range`                  \
# set xtics center offset 0,-1; \
