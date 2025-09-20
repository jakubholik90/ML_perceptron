package pl.programujodpodstaw;

import smile.data.DataFrame;

import java.lang.reflect.InvocationTargetException;
import java.util.*;

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

        // step 6.
        double minDataX0 = 0;
        double maxDataX0 = 0;
        double minDataX1 = 0;
        double maxDataX1 = 0;
        for (DataRecord dataRecord : trainingDataList) {
            if(dataRecord.dataX0()<=minDataX0) {
                minDataX0 = dataRecord.dataX0();
            }
            if(dataRecord.dataX0()>=maxDataX0) {
                maxDataX0 = dataRecord.dataX0();
            }
            if(dataRecord.dataX1()<=minDataX1) {
                minDataX1 = dataRecord.dataX1();
            }
            if(dataRecord.dataX1()>=maxDataX1) {
                maxDataX1 = dataRecord.dataX1();
            }
        }

        int numOfBorderPoints = 51;
        double borderX0[] = new double[numOfBorderPoints];
        double borderX1[] = new double[numOfBorderPoints];
        List<DataRecord> borderPointList = new LinkedList<>();

        for (int i = 0; i < numOfBorderPoints; i++) {
            for (int j = 0; j < numOfBorderPoints; j++) {
                borderX0[i] = minDataX0 + i*(maxDataX0-minDataX0)/(numOfBorderPoints-1);
                borderX1[j] = minDataX1 + j*(maxDataX1-minDataX1)/(numOfBorderPoints-1);
                DataRecord borderPoint = perceptron.getBorderPoint(new DataRecord(null, borderX0[i], borderX1[j],1));
                trainingDataList.add(borderPoint);
                borderPointList.add(borderPoint);
            }

        }

        DataFrame trainingDataFrame = Plot.generateDataFrame(trainingDataList);
        Plot.showPlot(trainingDataFrame);


    }
}