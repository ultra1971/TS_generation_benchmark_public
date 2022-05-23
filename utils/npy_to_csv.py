import numpy as np
import argparse
import os


# npyPath = "C:/Users/jonas/OneDrive/Desktop/University/Informatique/7. Bachelor work/Generated_Data/TimeGAN/Npy"
# npyName = "Lighting7_TEST_TimeGAN.npy"
# npyName = "Currency2_TimeGAN.npy"

# outputFolderPath = "C:/Users/jonas/OneDrive/Desktop/University/Informatique/7. Bachelor work/Generated_Data/TimeGAN/Csv/Currency2_pieces"

# TimeGAN -> length 24
# InfoGAN -> length 384 (?)

# input_path = npyPath + "/" + npyName


def main(args):
	array = np.load(args.npy_path)
	
	# Create folder if not exist
	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder)
   

	# Create a fakeX.csv for each outer array
	for i in range(len(array)):
		# Insert array with headers
		headerArr = np.insert(array[i], 0, np.array([x for x in range(len(array[i][0]))]), 0)
		# Save csv
		np.savetxt(args.output_folder + "/fake" + str(i) + ".csv", headerArr, delimiter=",")

	print(f"{len(array)} csv files written to {args.output_folder}")
	
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--npy_path', type=str, default="results/TimeGAN/BeetleFly_TEST_ep=10_seqLen=24.npy", help='Path to the npy file')
	parser.add_argument('--output_folder', type=str, default="results/TimeGAN/BeetleFly_pieces", help='Path to the output folder')
	
	args = parser.parse_args()
    
	main(args)
