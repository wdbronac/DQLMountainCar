package regression.reinforce.vectorPlots;

import org.jfree.data.xy.VectorSeries;
import org.jfree.data.xy.VectorSeriesCollection;
import org.jfree.chart.renderer.xy.VectorRenderer;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.ChartFrame;

import java.awt.*;
import java.util.ArrayList;

/**
 * A simple introduction to using JFreeChart.
 */
public class VectorProject2 {
    /**
     * .
     *
     * @param args ignored.
     */
    public static void main(String[] args) {
// create a dataset...
// First we create a dataset



// We create a vector series collection
        ArrayList<VectorSeriesCollection> dataSet= new ArrayList<VectorSeriesCollection>();

        VectorSeries vectorSeries=new VectorSeries("First Series");

        vectorSeries.add(0, 0, 5, 5);
        vectorSeries.add(1, 4, 3, -2);
        vectorSeries.add(4, 5, 2, 1);
        vectorSeries.add(10, 0, 0, -5);

        VectorSeriesCollection coll1= new VectorSeriesCollection();
        coll1.addSeries(vectorSeries);
        dataSet.add(coll1);
        VectorRenderer r = new VectorRenderer();


        VectorSeries vectorSeries2=new VectorSeries("First Series");

        vectorSeries.add(0, 9, 9, 0);
        vectorSeries.add(2, 1, -4, 5);
        vectorSeries.add(4, 4, 5, 9);
        vectorSeries.add(1, 10, -4, 2);
        VectorSeriesCollection coll2= new VectorSeriesCollection();
        coll2.addSeries(vectorSeries);
        dataSet.add(coll2);

//r.setBasePaint(Color.white);
r.setSeriesPaint(0, Color.blue);
        r.setSeriesPaint(1, Color.red);

//
//        XYPlot xyPlot = new XYPlot(dataSet, new NumberAxis("Axis X"), new NumberAxis("Axis Y"), r);
//
//// Create a Chart
//        JFreeChart theChart;
//
//        theChart = new JFreeChart(xyPlot);
//        theChart.setTitle("El Nuevo Chart");
//
//
//
//
//// create and display a frame...
//        ChartFrame frame = new ChartFrame("First", theChart);
//        frame.pack();
//        frame.setVisible(true);
    }
}