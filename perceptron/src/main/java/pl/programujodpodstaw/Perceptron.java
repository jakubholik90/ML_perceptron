package pl.programujodpodstaw;

import java.util.*;

public class Perceptron {
    private double w0; // weight of dataX0
    private double w1; // weight of dataX1
    private double w2; // weight of dataX2
    private double bias; // bias
    private double eta; // eta factor for step adjustment by backpropagation
    private int maxNumberOfIterations; // max. number of iterations to perform
    private int currentIteration;

    public Perceptron(double eta, int maxNumberOfIterations) {
        Random random = new Random();
        this.w0 = random.nextDouble(10) -5;
        this.w1 = random.nextDouble(10) -5;
        this.w2 = random.nextDouble(10) -5;
        this.bias = random.nextDouble(10) - 5;
        this.eta = eta;
        this.maxNumberOfIterations = maxNumberOfIterations;
        this.currentIteration = 0;
    }

    public double[] train(List<DataRecord> trainingDataList, HashMap<String, Integer> trainingDataNamesMap) {
        int totalNumberOfPoints = trainingDataList.size(); // total number of points

        double deltaW0; // modification delta for w0
        double deltaW1; // modification delta for w1
        double deltaW2; // modification delta for w2
        double deltaBias;// modification delta for bias
        double totalError; // total error between whole trainingData and known output


        //main loop for training (iterations)
        boolean runTraining = true;
        while (runTraining) {
            deltaW0=0; // resetting delta W0
            deltaW1=0; // resetting delta W1
            deltaW2=0; // resetting delta W1
            deltaBias=0; // resetting delta Bias
            totalError=0; //resetting total error
            this.currentIteration++; // increasing current iteration number

            //loop for each point in the list
            for (DataRecord trainingData : trainingDataList) {
                //1. sum of weights (x0*w0 + x1*x1 + bias)
                double z = trainingData.dataX0()*this.w0 + trainingData.dataX1()*this.w1 + trainingData.dataX2()*this.w2 + this.bias;

                //2. activation function (0 for negative z, 1 for positive z)
                double a = stepFunction(z);

                //3. calculation of error (prediction - expected output)
                int y = trainingDataNamesMap.get(trainingData.dataName());
                double error = y - a;

                //4. propagating error into deltas
                deltaW0 = deltaW0 + error*trainingData.dataX0() + this.eta;
                deltaW1 = deltaW1 + error*trainingData.dataX1() + this.eta;
                deltaW2 = deltaW2 + error*trainingData.dataX2() + this.eta;
                deltaBias = deltaBias + error + this.eta;
                totalError = totalError + Math.pow(error,2);
            }

            //breaking while loop when totalError == 0;
            if (totalError==0) {
                System.out.println("totalError==0, breaking the iteration loop after " + currentIteration + " iterations");
                runTraining=false;
                break;
            }

            //breaking while loop when max number of iterations is reached
            if (currentIteration>=this.maxNumberOfIterations) {
                System.out.println("solution not found, breaking after " + currentIteration + " iterations");
                runTraining=false;
                break;
            }

            //average deltas for every input
            deltaW0 = deltaW0 / totalNumberOfPoints;
            deltaW1 = deltaW1 / totalNumberOfPoints;
            deltaW2 = deltaW2 / totalNumberOfPoints;
            deltaBias = deltaBias / totalNumberOfPoints;

            //modification of weights
            this.w0 = this.w0 + deltaW0;
            this.w1 = this.w1 + deltaW1;
            this.w2 = this.w2 + deltaW2;
            this.bias = this.bias + deltaBias;
        }

        double[] returnArray = new double[4];
        returnArray[0]  = this.w0;
        returnArray[1]  = this.w1;
        returnArray[2]  = this.w2;
        returnArray[returnArray.length-1]  = this.bias;
        return returnArray;
    }

    public HashMap<String, Integer> getDataNamesMap(List<DataRecord> trainingDataList) {
        //create set of training data names
        SortedSet<String> trainingDataNamesSet = new TreeSet<>();
        for (DataRecord trainingDataRecord : trainingDataList) {
            trainingDataNamesSet.add(trainingDataRecord.dataName());
        }

        //create HashMap with nummerations (values) of training data names (keys)
        HashMap<String,Integer> dataNamesMap = new HashMap<>();
        int i=0;
        for (String trainingDataName : trainingDataNamesSet) {
            dataNamesMap.put(trainingDataName,i);
            i++;
        }
        return dataNamesMap;
    }


    public double runOnSingleInput(double[] weightsArray, DataRecord inputDataRecord) {
        double returnValue = 0;
        double z = inputDataRecord.dataX0()*weightsArray[0] + inputDataRecord.dataX1()*weightsArray[1] + inputDataRecord.dataX2()*weightsArray[2] + weightsArray[weightsArray.length-1];
        returnValue = stepFunction(z);
        return returnValue;
    }

    private double stepFunction(double z) {
        double returnValue;
        if (z <0) {
            returnValue = 0;
        } else {
            returnValue = 1;
        }
        return returnValue;
    }

    public HashMap<String, Double> interpretResults(double result, HashMap<String, Integer> dataNamesMap) {
        Set<String> dataNamesSet = dataNamesMap.keySet();
        HashMap<String, Double> resultMap = new HashMap<>();
        for (String dataName : dataNamesSet) {
            double probability = 1 - Math.abs(dataNamesMap.get(dataName) - result);
            resultMap.put(dataName,probability);
        }
        return resultMap;
    }

    public DataRecord getBorderPoint(DataRecord inputDataRecord) {
        double outputDataX2 = (-inputDataRecord.dataX0()*this.w0 - inputDataRecord.dataX1()*this.w1 ) /  this.w2 - this.bias / this.w2;
        DataRecord resultDataRecord = new DataRecord("Bordersurface",inputDataRecord.dataX0(),inputDataRecord.dataX1(),outputDataX2);
        return resultDataRecord;
    }



    public double getW0() {
        return w0;
    }

    public double getW1() {
        return w1;
    }

    public double getW2() {
        return w2;
    }

    public double getBias() {
        return bias;
    }

    public int getCurrentIteration() {
        return currentIteration;
    }
}
