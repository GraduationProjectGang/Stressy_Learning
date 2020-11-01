package stressy;

import au.com.bytecode.opencsv.CSVReader;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;

public class StressyModel1 {

    static double[][][] getData(){
        int timeStep = 5;
        int feature = 6;
        double[][][] data_all = new double[4014][timeStep][feature];
        try{
            System.out.println("fileread");
            File file = new File("src/main/resources/trainingData_all.csv");
            InputStreamReader is = new InputStreamReader(new FileInputStream(file));
            BufferedReader reader = new BufferedReader(is);
            CSVReader read = new CSVReader(reader);
            String[] record = null;
            for (int i =0; i < 4014 ; i++){ //i=4014
                record = read.readNext();
                double[][] coroutine_array = new double[timeStep][feature];
                for (int j = 0 ; j < timeStep; j++){//1,5,6
                    String[] attributes = record[j].replace("[","").replace("]","").split(",");
                    double[] att_array = new double[feature];
                    for(int k = 0 ; k < feature; k++){
                        coroutine_array[j][k] = Double.parseDouble(attributes[j].trim());
                    }
                }
                data_all[i] = coroutine_array;
            }
        }catch (IOException e){
            e.printStackTrace();
        }
        System.out.println(data_all.length);
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

//        Normalize

        double min = 0;
        double max = 0;

        for (int i = 0; i < data_array.length; i++) {
            for (int j = 0; j < data_array[i].length; j++) {
                for (int k = 0; k < data_array[i][j].length; k++) {
                    max = Math.max(data_array[i][j][k], max);
                    min = Math.min(data_array[i][j][k], min);
                }
            }
        }

        for (int i = 0; i < data_array.length; i++) {
            for (int j = 0; j < data_array[i].length; j++) {
                for (int k = 0; k < data_array[i][j].length; k++) {
                    data_array[i][j][k] = (2 * (data_array[i][j][k] - min)) / (max - min) - 1;
                }
            }
        }

        int trainSplit = (int)(4014 * 0.8);

        double[][][] trainFeatures = new double[trainSplit][5][6];
        double[][] trainLabels = new double[trainSplit][4];
        double[][][] testFeatures = new double[4014-trainSplit][5][6];
        double[][] testLabels = new double[4014-trainSplit][4];

        for (int i = 0; i< trainSplit ; i++) {
            trainFeatures[i] = data_array[i];
            trainLabels[i] = label_array[i];
        }
        for (int i = 0; i < 4014-trainSplit; i++){
            testFeatures[i] = data_array[i+trainSplit];
            testLabels[i] = label_array[i+trainSplit];
        }

        INDArray x_train = Nd4j.create(trainFeatures);
        INDArray y_train = Nd4j.create(trainLabels);
        INDArray x_test = Nd4j.create(testFeatures);
        INDArray y_test = Nd4j.create(testLabels);

        DataSet trainDataSet = new DataSet(x_train,y_train);
        DataSet testDataSet = new DataSet(x_test,y_test);

        DataSetIterator train_iter = new ExistingDataSetIterator(trainDataSet);
        DataSetIterator test_iter = new ExistingDataSetIterator(testDataSet);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .list()
                .layer(0, new DenseLayer.Builder().nIn(30).nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(1000).nOut(1000)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(1000).nOut(4).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1000));

        Evaluation eval = new Evaluation(4);

        for (int i = 0; i < 1; i++) {

            while (train_iter.hasNext()) {
                DataSet nextData = train_iter.next();
                INDArray reshapedFeatures = nextData.getFeatures().reshape(1,30);
                INDArray reshapedLabels = nextData.getLabels();
                model.fit(new DataSet(reshapedFeatures, reshapedLabels));
            }

            while (test_iter.hasNext()) {
                DataSet nextData = test_iter.next();
                INDArray reshapedFeatures = nextData.getFeatures().reshape(1, 30);
                INDArray reshapedLabels = nextData.getLabels();
                INDArray output = model.output(reshapedFeatures);
                eval.eval(reshapedLabels, output);
            }

            System.out.println(eval.stats());
            train_iter.reset();
            test_iter.reset();
        }

        boolean saveUpdate = true;
        File locationToSave = new File("src/main/resources/stressy_model.zip");
        ModelSerializer.writeModel(model, locationToSave, saveUpdate);
    }
}