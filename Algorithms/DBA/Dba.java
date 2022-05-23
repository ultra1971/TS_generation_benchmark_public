import java.io.*;

public class Dba {

	public static void main(String[] args) {
		System.out.println("Hello world!");
		
		if (args.length < 1) {
			System.out.println("Error: csv path not specified for DBA!");
			System.exit(1);
		}
		
		String path = args[0];
		
		
		// Load data
		double[][] temp_data = readFile(path, ",");
		double[][] data = getTransposedMatrix(temp_data);
		//double[][] data = readFile(path, ";");	
			
		// Check if data was correctly loaded
		if (data == null) {
			System.out.println("Error: the path specied is not a valid csv file!");
			System.exit(1);
		}
		
		// Get the name
		String name = getName(path);
		
		// Print name and shape to console	
		System.out.println("Loaded data from " + name + ", size: " + data.length + "x" + data[0].length);
		
		// Randomly choosed seed
		long seed = -9177012694930770614L;

		// Generate synthetic data
        double[][] newData = TimeSeriesGenerator.generateFakeData(3, data, data.length, seed);

		// Save the results
		if (true) {
			printResults("../../results/" + name + "/data/" + name + "_DBA.csv", getTransposedMatrix(newData));
        }
	}
	
	
	private static double[][] readFile(String name, String separator) {
        try {
            // Open the file
            BufferedReader file = new BufferedReader(new FileReader(name));

            // Count the lines
            int lCount = 0;
            while ((file.readLine()) != null) {
                lCount++;
            }
            // The array will at most contain one element per line
            double[][] data = new double[lCount][];

            // Return at the beginning of the file
            file.close();
            file = new BufferedReader(new FileReader(name));

            String line;
            int lNum = 0;
 /*           while((line = file.readLine()) != null) {
                String[] parts = line.split(separator);
                // If the data is in the series we are interested in (-1 as parameters means all the series)
                if (true) {
                    data[lNum] = new double[parts.length - 1];
                    for (int i = 1; i < parts.length; i++) {
                        data[lNum][i - 1] = Double.parseDouble(parts[i]);
                    }
                    lNum++;
                }
            }
  */
  			file.readLine();  // Skip first line, with columns names
  			while((line = file.readLine()) != null) {
  				String[] parts = line.split(separator);
  				data[lNum] = new double[parts.length];
  				for (int i = 0; i < parts.length; i++) {
  					data[lNum][i] = Double.parseDouble(parts[i]);
  				}
  				lNum++;
  			}

            return reduceFirstArray(data);

        } catch (java.io.FileNotFoundException e) {
            System.err.println("File not found");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
    
    // Ev. remove empty rows
	private static double[][] reduceFirstArray(double[][] array) {
        for (int i = 0; i < array.length; i++) {
            if(array[i] == null) {
                double[][] reduced = new double[i][];
                for (int j = 0; j < i; j++) {
                    reduced[j] = array[j];
                }
                return reduced;
            }
        }
        return array;
    }
    
    
	private static void printResults(String filepath, double[][] results) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(filepath));
            StringBuilder sb = new StringBuilder();

            for (double[] ts : results) {
                // Insert class number (by default -1)
                //sb.append("-1");
                //sb.append(";");
                for (double value : ts) {
                    sb.append(String.valueOf(value));
                    sb.append(",");
                }
                // Remove last ","
                sb.setLength(sb.length() - 1);
                sb.append("\n");
            }

            bw.write(sb.toString());

            bw.flush();
            bw.close();
            System.out.println("Data printed to " + filepath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static String getName(String path) {
    
    	// Split on each point
    	String[] p_spl = path.split("\\.");
    	
    	// If there is no point, just return everything
    	if (p_spl.length == 1) {
    		return p_spl[0];
    	}
    	
    	// Take the last part
    	String last = p_spl[p_spl.length - 2];
    	
    	// Remove ev. folders name
    	if (last.contains("/")) {
    		// Take only the name
    		String[] s_spl = last.split("/");
    		last = s_spl[s_spl.length - 1];
    	}
    	
    	return last;
    }
    
    static double[][] getTransposedMatrix(double[][] matrix) {
        int maxLen = 0;
        for (double[] d : matrix) {
            if (d.length > maxLen) {
                maxLen = d.length;
            }
        }

        double[][] transposedMatrix = new double[maxLen][matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }
        return transposedMatrix;
    }
}
