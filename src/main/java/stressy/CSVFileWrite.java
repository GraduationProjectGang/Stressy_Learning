package stressy;

import au.com.bytecode.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import java.io.*;

public class CSVFileWrite {

    static double[][][] data_label_2;
    static int[] where_is_data;

    static double[][][] getData(){

        double[][][] data_all = new double[4014][5][6];

        try{
            File file = new File("src/main/resources/trainingData_all.csv");
            InputStreamReader is = new InputStreamReader(new FileInputStream(file));
            BufferedReader reader = new BufferedReader(is);
            CSVReader read = new CSVReader(reader);
            String[] record = null;

            for (int i =0; i < 4014 ; i++){ //i=4014
                record = read.readNext();
                double[][] coroutine_array = new double[5][6];
                for (int j = 0 ; j < 5; j++){//1,5,6
                    String[] attributes = record[j].replace("[","").replace("]","").split(",");
                    for(int k = 0 ; k < 6; k++) {
                        coroutine_array[j][k] = Double.parseDouble(attributes[k].trim());
                    }
                }
                data_all[i] = coroutine_array;

            }
        }catch (IOException e){
            e.printStackTrace();
        }

        return data_all;
    }

    static double[][] getLabel(){
        int numLabel = 1;
        int timestep = 5;
        double[][] data_label = new double[4014][4];
        data_label_2 = new double[4014][4][5];
        where_is_data = new int[4014];
        try{
            File file = new File("src/main/resources/stressData_all.csv");
            InputStreamReader is = new InputStreamReader(new FileInputStream(file));
            BufferedReader reader = new BufferedReader(is);
            CSVReader read = new CSVReader(reader);
            String[] record = null;
            record = read.readNext();
            for (int j = 0 ; j < 4014; j++){//one-hot encoding 4014, 4, 5
                double temp = Double.parseDouble(record[j].trim());
                for (int k = 0; k < 5; k++){
                    data_label_2[j][(int)temp][k] = 1;
                }
            }
            for (int j = 0 ; j < 4014; j++){//4014,1,4
                double temp = Double.parseDouble(record[j].trim());
                data_label[j][(int)temp] = 1;
                where_is_data[j] = (int)temp;
            }

        }catch (IOException e){
            e.printStackTrace();
        }
        return data_label;
    }

    public static void main(String[] args) throws IOException {

        double[][][] featureArr = getData();
        double[][] labelArr = getLabel();

        CSVWriter writer = new CSVWriter(new FileWriter("src/main/resources/stressy_data.csv"));

        for (int i = 0; i < 4014; i++) {
            String[] write_string = new String[31];
            int idx = 1;
            write_string[0] = String.valueOf(where_is_data[i]);
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 6; k++) {
                    write_string[idx++] = String.valueOf(featureArr[i][j][k]);
                }
            }
            writer.writeNext(write_string);
        }

        writer.close();

    }

}
