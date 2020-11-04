package stressy;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;

public class StressyWithRR2 {

    static double[][][] data_label_2;
    static int[] where_is_data;
    static double[][][] featureArr;
    static double[][] labelArr;

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

    static void shuffleCSV() throws IOException {

        ArrayList<Integer> random_index = new ArrayList();
        for (int i = 0; i < 4014; i++) {
            random_index.add(i);
        }
        Collections.shuffle(random_index);

        CSVWriter train_writer = new CSVWriter(new FileWriter("src/main/resources/stressy_train_data.csv"));
        CSVWriter test_writer = new CSVWriter(new FileWriter("src/main/resources/stressy_test_data.csv"));

        int trainSplit = (int)(4014 * 0.8);

        for (int i = 0; i < trainSplit; i++) {
            String[] write_string = new String[31];
            int idx = 1;
            write_string[0] = String.valueOf(where_is_data[random_index.get(i)]);
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 6; k++) {
                    write_string[idx++] = String.valueOf(featureArr[random_index.get(i)][j][k]);
                }
            }
            train_writer.writeNext(write_string);
        }

        for (int i = trainSplit; i < 4014; i++) {
            String[] write_string = new String[31];
            int idx = 1;
            write_string[0] = String.valueOf(where_is_data[random_index.get(i)]);
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 6; k++) {
                    write_string[idx++] = String.valueOf(featureArr[random_index.get(i)][j][k]);
                }
            }
            test_writer.writeNext(write_string);
        }

        train_writer.close();
        test_writer.close();
    }

    public static void main(String[] args) throws IOException, InterruptedException {

        featureArr = getData();
        labelArr = getLabel();

        String TrainFilePath = "src/main/resources/stressy_train_data.csv";
        String TestFilePath = "src/main/resources/stressy_test_data.csv";

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(30).nOut(1024)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(1024).nOut(512)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(512).nOut(4).build())
                .build();

//        File locationToLoad = new File("src/main/resources/stressy_model_nn_rr.zip");
//        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad, false);

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        double last_accuracy = 0.33;

        while (true) {

            shuffleCSV();

            RecordReader rrTrain = new CSVRecordReader();
            rrTrain.initialize(new FileSplit(new File(TrainFilePath)));

            RecordReader rrTest = new CSVRecordReader();
            rrTest.initialize(new FileSplit(new File(TestFilePath)));

            DataSetIterator train_iter = new RecordReaderDataSetIterator(rrTrain, 1, 0, 4);
            DataSetIterator test_iter = new RecordReaderDataSetIterator(rrTest, 1, 0, 4);

            model.fit(train_iter, 10);

            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(4);
            while(test_iter.hasNext()){
                DataSet t = test_iter.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray predicted = model.output(features,false);

                eval.eval(labels, predicted);

            }

            System.out.println(eval.stats());

            if (last_accuracy < eval.accuracy()) {
                boolean saveUpdate = true;
                File locationToSave = new File("src/main/resources/stressy_model_rr2.zip");
                ModelSerializer.writeModel(model, locationToSave, saveUpdate);
                last_accuracy = eval.accuracy();
            }

        }

    }

}
