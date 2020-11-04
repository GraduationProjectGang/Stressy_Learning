package stressy;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class StressyWithRR {

    public static void main(String[] args) throws IOException, InterruptedException {

        String FilePath = "src/main/resources/stressy_data.csv";

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(FilePath)));

        DataSetIterator iter = new RecordReaderDataSetIterator(rr, 10, 0, 4);

        System.out.println(iter);

        DataSet allData = iter.next();
        System.out.println(allData);
        allData.shuffle();

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Adam(0.001))
//                .list()
//                .layer(0, new DenseLayer.Builder().nIn(30).nOut(1024)
//                        .activation(Activation.LEAKYRELU)
//                        .build())
//                .layer(1, new DenseLayer.Builder().nIn(1024).nOut(512)
//                        .activation(Activation.LEAKYRELU)
//                        .build())
//                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.SOFTMAX)
//                        .nIn(512).nOut(4).build())
//                .build();

        File locationToLoad = new File("src/main/resources/stressy_model_nn_rr.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad, false);
        model.init();
        model.setListeners(new ScoreIterationListener(1000));

        Evaluation eval = new Evaluation(4);

        double last_accuracy = 0.34;

        while (true) {
            for (int i = 0; i < 10; i++) {

                model.fit(iter);
                iter.reset();

                while (iter.hasNext()) {
                    DataSet nextData = iter.next();
                    INDArray reshapedFeatures = nextData.getFeatures();
                    INDArray reshapedLabels = nextData.getLabels();
                    INDArray output = model.output(reshapedFeatures);
                    eval.eval(reshapedLabels, output);
                }

                System.out.println(eval.stats());

            }

            if (last_accuracy < eval.accuracy()) {
                boolean saveUpdate = true;
                File locationToSave = new File("src/main/resources/stressy_model_nn_rr.zip");
                ModelSerializer.writeModel(model, locationToSave, saveUpdate);
                last_accuracy = eval.accuracy();
            }



        }




    }

}
