package stressy;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;

import static org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasModelAndWeights;

public class StressyFromKeras {

    final static double STD_MAX = 44014018.0;
    final static double STD_MIN = 0.0;

    final static int trainSplit = (int)(4014 * 0.8);
    final static int nEpochs = 1000;

    static double[][][] data_array;
    static double[][] label_array;

    static DataSet trainDataSet;
    static DataSet testDataSet;

    static DataSetIterator train_iter;
    static DataSetIterator test_iter;

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

    static ArrayList<Integer> getRandomIndex() {

        ArrayList<Integer> random_index = new ArrayList();
        for (int i = 0; i < 4014; i++) {
            random_index.add(i);
        }
        Collections.shuffle(random_index);

        return random_index;
    }

    static void ShuffleTrainTest() {

        double[][][] train_data = new double[trainSplit][5][6];
        double[][] train_label = new double[trainSplit][4];
        double[][][] test_data = new double[4014 - trainSplit][5][6];
        double[][] test_label = new double[4014 - trainSplit][4];

        ArrayList<Integer> newRandomIndex = getRandomIndex();

        for (int i = 0; i < trainSplit; i++) {
            train_data[i] = data_array[newRandomIndex.get(i)];
            train_label[i] = label_array[newRandomIndex.get(i)];
        }

        for (int i = 0; i < 4014 - trainSplit; i++) {
            test_data[i] = data_array[newRandomIndex.get(trainSplit + i)];
            test_label[i] = label_array[newRandomIndex.get(trainSplit + i)];
        }

        INDArray train_data_ind = Nd4j.create(train_data);
        INDArray train_label_ind = Nd4j.create(train_label);
        INDArray test_data_ind = Nd4j.create(test_data);
        INDArray test_label_ind = Nd4j.create(test_label);

        INDArray normalized_train_data = train_data_ind.div(STD_MAX);
        INDArray normalized_test_data = test_data_ind.div(STD_MAX);

        trainDataSet = new DataSet(normalized_train_data, train_label_ind);
        testDataSet = new DataSet(normalized_test_data, test_label_ind);

        train_iter = new ExistingDataSetIterator(trainDataSet);
        test_iter = new ExistingDataSetIterator(testDataSet);

    }

    public static void main(String[] args) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {

//        String stressyAbsolutePath = new File("src/main/resources/stressy_model.h5").getAbsolutePath();
//        ComputationGraph stressy_keras = KerasModelImport.importKerasModelAndWeights(stressyAbsolutePath);

        data_array = getData();
        label_array = getLabel();

        InputStream modelByteStream = new BufferedInputStream(new FileInputStream("src/main/resources/stressy_model.h5"));
        MultiLayerNetwork stressy_keras = KerasModelImport.importKerasSequentialModelAndWeights(modelByteStream, true);

        System.out.println(stressy_keras.summary());

        stressy_keras.init();
        stressy_keras.setListeners(new ScoreIterationListener(100));

        double last_accuracy = 0.33;

        Evaluation eval = new Evaluation(4);

        while (true) {

            ShuffleTrainTest();

            stressy_keras.fit(train_iter, nEpochs);

            while (test_iter.hasNext()) {
                DataSet nextData = test_iter.next();
                INDArray features = nextData.getFeatures();
                INDArray labels = nextData.getLabels();
                INDArray predicted = stressy_keras.output(features,false);

                eval.eval(labels, predicted);
            }

            System.out.println(eval.stats());

            if (last_accuracy < eval.accuracy()) {
                boolean saveUpdate = true;
                File locationToSave = new File("src/main/resources/stressy_from_keras.zip");
                ModelSerializer.writeModel(stressy_keras, locationToSave, saveUpdate);
                last_accuracy = eval.accuracy();
            }

            train_iter.reset();
            test_iter.reset();

        }

    }

}
