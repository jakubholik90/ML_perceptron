package pl.programujodpodstaw;

import smile.data.DataFrame;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws InterruptedException, InvocationTargetException {

        // step 1. creating and filling a list of training data - can be modified
        System.out.println("--step 1. creating and filling a list of training data - can be modified");
        List<DataRecord> trainingDataList = new LinkedList<>();
        // coordinates of points type A
        trainingDataList.add(new DataRecord("B",1,2,5.5));
        trainingDataList.add(new DataRecord("B",2,2.2,4.4));
        trainingDataList.add(new DataRecord("B",3,3.3,3.3));
        trainingDataList.add(new DataRecord("B",4,4.4,2.2));
        trainingDataList.add(new DataRecord("B",5,5.5,0));
        // coordinates of points type B-
        trainingDataList.add(new DataRecord("A",-1,-1.1,-5.5));
        trainingDataList.add(new DataRecord("A",-2,-2.2,-4.4));
        trainingDataList.add(new DataRecord("A",0,-3.3,-3.3));
        trainingDataList.add(new DataRecord("A",-4,-4.4,-2.2));
        trainingDataList.add(new DataRecord("A",5,-5.5,-1.1));
         System.out.println(trainingDataList);

        // step 2. creating a new Percepton instance
        System.out.println("--step 2. creating a new Percepton instance");
        Perceptron perceptron = new Perceptron(0.1, 10000);
        System.out.println(perceptron);

        // step 3. getting data names map
        System.out.println("--step 3. getting data names map");
        HashMap<String, Integer> dataNamesMap = perceptron.getDataNamesMap(trainingDataList);
        System.out.println(dataNamesMap);

        // step 4. show results from training
        System.out.println("--step 4. show results from training");
        double[] trainedWeights = perceptron.train(trainingDataList, dataNamesMap);
        System.out.println("current iteation: " + perceptron.getCurrentIteration());
        System.out.println("predicted w0: " + perceptron.getW0());
        System.out.println("predicted w1: " + perceptron.getW1());
        System.out.println("predicted w2: " + perceptron.getW2());
        System.out.println("predicted bias: " + perceptron.getBias());
        System.out.println("trainedValues: " + Arrays.toString(trainedWeights));

        // step 5. calculate and interpret results
        DataRecord inputDataRecord = new DataRecord("inputDataRecord", 1.3, 1.4, 1.5);
        double resultsFromSingleInput = perceptron.runOnSingleInput(trainedWeights, inputDataRecord);
        HashMap<String, Double> resultAndProbabilities = perceptron.interpretResults(resultsFromSingleInput, dataNamesMap);
        System.out.println("resultAndProbabilities: " + resultAndProbabilities);

        //step 6. show plots

        DataRecord resultDataRecord = new DataRecord(resultAndProbabilities.toString(),inputDataRecord.dataX0(),inputDataRecord.dataX1(),inputDataRecord.dataX2());
        trainingDataList.add(resultDataRecord);
        System.out.println(trainingDataList);
        for (DataRecord dataRecord : List.copyOf(trainingDataList)) {
            DataRecord borderPoint = perceptron.getBorderPoint(dataRecord);
            trainingDataList.add(borderPoint);
        }
        System.out.println("--");
        System.out.println(trainingDataList);
        DataFrame trainingDataFrame = Plot.generateDataFrame(trainingDataList);
        Plot.showPlot(trainingDataFrame);


    }
}