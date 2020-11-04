package stressy;

import au.com.bytecode.opencsv.CSVReader;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

public class StressyModelTraining {

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
//                for(double[] item: coroutine_array){
//                    for (double item2: item) {
//                        System.out.print(item2 + " ");
//                    }
//                    System.out.println("");
//                }
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

    public static void main(String[] args) throws Exception {
        int seed = 1000;
        double learningRate = 0.001;
        int batchSize = 1;
        int epochs = 1000;
        double[][][] data_array = getData();
        double[][] label_array = getLabel();
        INDArray data_ind = Nd4j.create(data_array);
        INDArray data_reshape_ind = data_ind.reshape(4014, 30);
        INDArray label_ind = Nd4j.create(label_array);
        INDArray label_reshape_ind = label_ind.reshape(4014, 4);
//        System.out.println(data_ind);
        double min = 0;
        double max = 0;
        for (int i = 0; i < 4014; i++) {
            for (int j = 0; j < 30; j++) {
                max = Math.max(max, data_reshape_ind.getDouble(i, j));
                min = Math.min(min, data_reshape_ind.getDouble(i, j));
            }
        }
        System.out.println(max + " " + min);
        INDArray normalized_data = data_reshape_ind.add(min * (-1)).div(max - min);
//        System.out.println(normalized_data);

        DataSet allData = new DataSet(normalized_data, label_reshape_ind);

        File locationToLoad = new File("src/main/resources/stressy_model_nn.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad, false);
        model.init();
//        model.setListeners(new ScoreIterationListener(1000));

        Evaluation eval = new Evaluation(4);

        double last_accuracy = 0.3;

        while (true) {

            allData.shuffle();

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
            DataSet trainData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            DataSetIterator train_iter = new ExistingDataSetIterator(trainData);
            DataSetIterator test_iter = new ExistingDataSetIterator(testData);

            int train_idx = 0;
            int test_idx = 0;

            for (int i = 0; i < 1; i++) {
                while (train_iter.hasNext()) {
                    DataSet nextData = train_iter.next();
                    INDArray reshapedFeatures = nextData.getFeatures().reshape(1, 30);

                    model.fit(new DataSet(reshapedFeatures, nextData.getLabels()));
                    train_idx++;
                }

                while (test_iter.hasNext()) {

                    DataSet nextData = test_iter.next();
                    INDArray reshapedFeatures = nextData.getFeatures().reshape(1, 30);
//                INDArray reshapedLabels = nextData.getLabels();

                    INDArray output = model.output(nextData.getFeatures().reshape(1, 30));
                    eval.eval(nextData.getLabels(), output);
//                eval.eval(output, reshapedLabels);
                    test_idx++;
                }

                System.out.println(eval.stats());
                System.out.println(train_idx + " " + test_idx);

                train_iter.reset();
                test_iter.reset();
            }

            if (eval.accuracy() > last_accuracy) {
                boolean saveUpdate = true;
                File locationToSave = new File("src/main/resources/stressy_model_nn.zip");
                ModelSerializer.writeModel(model, locationToSave, saveUpdate);
                last_accuracy = eval.accuracy();

                SimpleDateFormat sdf = new SimpleDateFormat("MMM dd,yyyy HH:mm");
                Date result_date = new Date(System.currentTimeMillis());

                System.out.println("Accuracy : " + last_accuracy + ", Saved at " + sdf.format(result_date));
            }

        }

    }

}