package stressy;

import au.com.bytecode.opencsv.CSVReader;

import java.io.*;

public class GetDataFixing {

    static double[][][] getData(){

        double[][][] data_all = new double[4014][5][6];

        try{
            System.out.println("fileread");
            File file = new File("src/main/resources/trainingData_all.csv");
            InputStreamReader is = new InputStreamReader(new FileInputStream(file));
            BufferedReader reader = new BufferedReader(is);
            CSVReader read = new CSVReader(reader);
            String[] record = null;

            for (int i =0; i < 4014 ; i++){ //i=4014
                record = read.readNext();
                double[][] coroutine_array = new double[5][6];
                for (int j = 0 ; j < 5; j++){//1,5,6
//                    System.out.println(record[j]);
                    String[] attributes = record[j].replace("[","").replace("]","").split(",");
//                    System.out.println(attributes[5]);
                    for(int k = 0 ; k < 6; k++) {
                        coroutine_array[j][k] = Double.parseDouble(attributes[k].trim());
                    }
                }
                data_all[i] = coroutine_array;

                //
                for(double[] item: coroutine_array){
                    for (double item2: item) {
                        System.out.print(item2 + " ");
                    }
                    System.out.println("");
                }
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
        try{
            File file = new File("src/main/resources/stressData_all.csv");
            InputStreamReader is = new InputStreamReader(new FileInputStream(file));
            BufferedReader reader = new BufferedReader(is);
            CSVReader read = new CSVReader(reader);
            String[] record = null;
            record = read.readNext();
            // for (int j = 0 ; j < 4014; j++){//one-hot encoding 4014, 4, 5
            // double temp = Double.parseDouble(record[j].trim());
            // for (int k = 0; k<timestep;k++){
            // data_label[j][(int)temp][k] = 1;
            // }
            // }
            for (int j = 0 ; j < 4014; j++){//4014,1,4
                double temp = Double.parseDouble(record[j].trim());
                data_label[j][(int)temp] = 1;
            }
        }catch (IOException e){
            e.printStackTrace();
        }
        return data_label;
    }

    public static void main(String[] args) {

        double[][][] data_arr = getData();
        double[][] label_arr = getLabel();

//        for (int i = 0; i < 4014; i++) {
//            for (int j = 0; j < 5; j++) {
//                for (int k = 0; k < 6; k++) {
//                    System.out.print(data_arr[i][j][k] + " ");
//                }
//                System.out.println();
//            }
//            System.out.println();
//            System.out.println();
//        }

    }

}
