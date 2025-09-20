package pl.programujodpodstaw;

import java.awt.Color;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

import smile.data.DataFrame;
import smile.io.*;
import smile.plot.swing.*;
import smile.stat.distribution.*;
import smile.math.matrix.*;



public class Plot {

    public static DataFrame generateDataFrame(List<DataRecord> dataRecordList)  {
        DataFrame returnDataFrame = DataFrame.of(DataRecord.class,dataRecordList);
        return returnDataFrame;
    }

    public static void showPlot(DataFrame dataFrame) throws InterruptedException, InvocationTargetException {

        var figure = ScatterPlot.of(dataFrame,
                "dataX0",
                "dataX1",
                "dataX2",
                "dataName",
                '*').figure();
        figure.setAxisLabels("dataX0", "dataX1", "dataX2");

        var pane = new FigurePane(figure);
        pane.window();
    }
}
