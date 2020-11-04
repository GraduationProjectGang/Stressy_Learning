package stressy;

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

import java.io.File;
import java.io.IOException;

public class StressyWithRR2 {

    public static void main(String[] args) throws IOException, InterruptedException {

        String TrainFilePath = "src/main/resources/stressy_train_data.csv";
        String TestFilePath = "src/main/resources/stressy_test_data.csv";

        RecordReader rrTrain = new CSVRecordReader();
        rrTrain.initialize(new FileSplit(new File(TrainFilePath)));

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(TestFilePath)));

        DataSetIterator train_iter = new RecordReaderDataSetIterator(rrTrain, 1, 0, 4);
        DataSetIterator test_iter = new RecordReaderDataSetIterator(rrTest, 1, 0, 4);

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
