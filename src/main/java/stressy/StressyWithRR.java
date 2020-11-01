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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
        int idx = 0;
        while (iter.hasNext()) {
            iter.next();
            idx++;
        }
        System.out.println(idx);

        iter.reset();
        DataSet allData = iter.next();

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainData = testAndTrain.getTrain();
        System.out.println(trainData.toString());
        DataSet testData = testAndTrain.getTest();
        System.out.println(testData.toString());

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(30).nOut(1024)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(1024).nOut(512)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(512).nOut(4).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1000));

        model.fit(trainData);

        INDArray output = model.output(testData.getFeatures());

        Evaluation eval = new Evaluation(4);
        eval.eval(testData.getLabels(), output);

        System.out.println(eval.stats());

        boolean saveUpdate = true;
        File locationToSave = new File("src/main/resources/stressy_model_nn_rr.zip");
        ModelSerializer.writeModel(model, locationToSave, saveUpdate);


    }

}
