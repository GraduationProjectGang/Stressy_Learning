package stressy;

import com.google.gson.JsonParser;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.*;

// 0_W 0_RW 0_b 2_W 2_b

public class FindWeights {

    static INDArray w_0;
    static INDArray rw_0;
    static INDArray b_0;
    static INDArray w_2;
    static INDArray b_2;

    final static double STD_MAX = 44014018.0;

    public static void readWeights(int idx) throws FileNotFoundException {

        switch (idx) {
            case 0:
                double[][] weight_list = new double[6][512];
                try {
                    File file = new File("src/main/resources/weights_0.json");
                    FileReader reader = new FileReader(file);
                    BufferedReader br = new BufferedReader(reader);

                    String line = "";
                    while ((line = br.readLine()) != null) {
                        String[] temp = line.replaceAll("\\[", "").split("], ");
                        for (int i = 0; i < temp.length; i++) {

                            String temp2 = temp[i].replaceAll("]", "");
                            String[] values = temp2.split(", ");

                            for (int j = 0; j < values.length; j++) {
                                weight_list[i][j] = Double.parseDouble(values[j]);
                            }
                        }
                    }
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
                w_0 = Nd4j.create(weight_list);
                break;
            case 1:
                double[][] weight_list_2 = new double[128][512];
                try {
                    File file = new File("src/main/resources/weights_1.json");
                    FileReader reader = new FileReader(file);
                    BufferedReader br = new BufferedReader(reader);

                    String line = "";
                    while ((line = br.readLine()) != null) {
                        String[] temp = line.replaceAll("\\[", "").split("], ");
                        for (int i = 0; i < temp.length; i++) {

                            String temp2 = temp[i].replaceAll("]", "");
                            String[] values = temp2.split(", ");

                            for (int j = 0; j < values.length; j++) {
                                weight_list_2[i][j] = Double.parseDouble(values[j]);
                            }
                        }
                    }
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
                rw_0 = Nd4j.create(weight_list_2);
                break;
            case 2:
                double[][] weight_list_3 = new double[1][512];
                try {
                    File file = new File("src/main/resources/weights_2.json");
                    FileReader reader = new FileReader(file);
                    BufferedReader br = new BufferedReader(reader);

                    String line = "";
                    while ((line = br.readLine()) != null) {
                        String[] temp = line.replaceAll("\\[", "").split("], ");
                        for (int i = 0; i < temp.length; i++) {

                            String temp2 = temp[i].replaceAll("]", "");
                            String[] values = temp2.split(", ");

                            for (int j = 0; j < values.length; j++) {
                                weight_list_3[i][j] = Double.parseDouble(values[j]);
                            }
                        }
                    }
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
                b_0 = Nd4j.create(weight_list_3);
                break;
            case 3:
                double[][] weight_list_4 = new double[128][4];
                try {
                    File file = new File("src/main/resources/weights_3.json");
                    FileReader reader = new FileReader(file);
                    BufferedReader br = new BufferedReader(reader);

                    String line = "";
                    while ((line = br.readLine()) != null) {
                        String[] temp = line.replaceAll("\\[", "").split("], ");
                        for (int i = 0; i < temp.length; i++) {

                            String temp2 = temp[i].replaceAll("]", "");
                            String[] values = temp2.split(", ");

                            for (int j = 0; j < values.length; j++) {
                                weight_list_4[i][j] = Double.parseDouble(values[j]);
                            }
                        }
                    }
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
                w_2 = Nd4j.create(weight_list_4);
                break;
            case 4:
                double[][] weight_list_5 = new double[1][4];
                weight_list_5[0] = new double[]{-4.282070159912109, -4.228370666503906, -4.528663158416748, -5.757140636444092};
                b_2 = Nd4j.create(weight_list_5);
            default:
                break;
        }

    }

    public static void main(String[] args) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        readWeights(0);
        InputStream modelByteStream = new BufferedInputStream(new FileInputStream("src/main/resources/stressy_model.h5"));
        MultiLayerNetwork stressy_keras = KerasModelImport.importKerasSequentialModelAndWeights(modelByteStream);

        System.out.println(stressy_keras.summary());
//
        Map<String, INDArray> paramTable = stressy_keras.paramTable();
        Set<String> keys = paramTable.keySet();
        Iterator<String> it = keys.iterator();

        for (int i = 0; i < 5; i++) {
            readWeights(i);
        }

        System.out.println(w_0.shapeInfoToString());
        System.out.println(rw_0.shapeInfoToString());
        System.out.println(b_0.shapeInfoToString());
        System.out.println(w_2.shapeInfoToString());
        System.out.println(b_2.shapeInfoToString());

        while (it.hasNext()) {
            String key = it.next();
            INDArray values = paramTable.get(key);
            System.out.print(key + " ");//print keys

//            for (int i = 0; i < values.rows(); i++) {
//                System.out.println(values.getRow(i));
//            }

//            System.out.println(Arrays.toString(values.shape()));//print shape of INDArray
            System.out.println(values);
//            transferred_model.setParam(key, Nd4j.rand(values.shape()));//set some random values
        }

        stressy_keras.setParam("0_W", w_0);//set some random values
        stressy_keras.setParam("0_RW", rw_0);
        stressy_keras.setParam("0_b", b_0);
        stressy_keras.setParam("2_W", w_2);
        stressy_keras.setParam("2_b", b_2);

        Iterator<String> it2 = keys.iterator();

        while (it2.hasNext()) {
            String key = it2.next();
            INDArray values = paramTable.get(key);
            System.out.print(key + " ");//print keys

//            for (int i = 0; i < values.rows(); i++) {
//                System.out.println(values.getRow(i));
//            }

//            System.out.println(Arrays.toString(values.shape()));//print shape of INDArray
            System.out.println(values);
//            transferred_model.setParam(key, Nd4j.rand(values.shape()));//set some random values
        }

//        File locationToSave = new File("src/main/resources/stressy_weights_copy.zip");
//        ModelSerializer.writeModel(stressy_keras, locationToSave, true);

        System.out.println(stressy_keras.summary());

        INDArray inputArr = Nd4j.create(new double[][][]{{
                {0, 1, 0, 0.29238041595804914, 0, 0},
                {0, 1, 0, 0.29238041595804914, 14, 2280715},
                {0, 1, 0, 0.29238041595804914, 6, 4663934},
                {0, 1, 0, 0.29238041595804914, 5, 155388},
                {0, 1, 0, 0.29238041595804914, 7, 91596}
        }});

        INDArray inputArr2 = Nd4j.create(new double[][][]{{
                {0, 1, 0, 0.0, 0, 0},
                {0, 1, 0, 0.0, 0, 0},
                {0, 1, 0, 0.0, 0, 0},
                {0, 1, 0, 0.0, 7, 61838},
                {0, 1, 0, 0.0, 5, 440983}
        }});

        INDArray inputArr3 = Nd4j.create(new double[][][]{{
                {0, 1, 3, 1.8550337827306764, 0, 0},
                {0, 1, 3, 1.8550337827306764, 0, 0},
                {0, 1, 3, 1.8550337827306764, 14, 749915},
                {0, 1, 3, 1.8550337827306764, 5, 2644291},
                {0, 1, 3, 1.8550337827306764, 7, 75112}
        }});

        INDArray inputNormalize = inputArr.mul(2).div(STD_MAX).sub(-1);
        INDArray inputNormalize2 = inputArr2.mul(2).div(STD_MAX).sub(-1);
        INDArray inputNormalize3 = inputArr3.mul(2).div(STD_MAX).sub(-1);


        INDArray result = stressy_keras.output(inputNormalize);
        INDArray result2 = stressy_keras.output(inputNormalize2);
        INDArray result3 = stressy_keras.output(inputNormalize3);

        System.out.println(result);
        System.out.println(result2);
        System.out.println(result3);

        stressy_keras.init();
        stressy_keras.setListeners(new ScoreIterationListener(1000));

        boolean saveUpdate = true;
        File locationToSave = new File("src/main/resources/stressy_final_model_2mall.zip");
        ModelSerializer.writeModel(stressy_keras, locationToSave, saveUpdate);


    }

}
