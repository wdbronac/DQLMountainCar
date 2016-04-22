package regression.reinforce;


import java.io.FileWriter;
import java.util.Arrays;

import java.awt.Color;
import javax.swing.JPanel;

import au.com.bytecode.opencsv.CSVWriter;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.experimental.chart.renderer.xy.VectorRenderer;
import org.jfree.experimental.data.xy.VectorSeries;
import org.jfree.experimental.data.xy.VectorSeriesCollection;
import org.jfree.experimental.data.xy.VectorXYDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RectangleInsets;
import org.jfree.ui.RefineryUtilities;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.indexing.NDArrayIndex;
import regression.function.*;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.plot.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
/**Example: Train a network to reproduce certain mathematical functions, and plot the results.
 * Plotting of the network output occurs every 'plotFrequency' epochs. Thus, the plot shows the accuracy of the network
 * predictions as training progresses.
 * A number of mathematical functions are implemented here.
 * Note the use of the identity function on the network output layer, for regression
 *
 * @author Alex Black
 */
public class Simulation {

    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Number of iterations per minibatch
    public static final int iterations = 1; // avant: 1
    //Number of epochs (full passes of the data)
//<<<<<<< HEAD
//    public static final int nEpochs = 2; // avant: 2000
//=======
    public static final int nEpochs = 500; // avant: 2000
//>>>>>>> 19b0a6be78d36e2ab13c2897376adacc3b76c5c9
    //How frequently should we plot the network output?
    public static final int plotFrequency = 500;
    //Number of data points
    public static final int nSamples = 1000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
//<<<<<<< HEAD
//    public static final int batchSize = 4096;
//=======
    public static final int batchSize = 300000;
//>>>>>>> 19b0a6be78d36e2ab13c2897376adacc3b76c5c9
    //Network learning rate
    public static final double learningRate = 0.01;
    public static final Random rng = new Random(seed);
    public static final int numInputs = 2;
    public static final int numOutputs = 3;


    //taille de la dataset qu on genere, voir comment j agence tout ca
    public static int size_dataset = 300000;
    public static int num_iterations = 150; // nombre d updates du Q network

    //resolution de l image de sortie:
    public static int resolution = 100;


    public static void main(final String[] args) throws Exception {
        String type = "MLP";
        //String type  = "RF";
        //Switch these two options to do different functions with different networks
        final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();

        //Create the network
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        MemoIterationListener itlis = new MemoIterationListener(1);
        net.setListeners(Collections.singletonList((IterationListener) itlis));
//        net.setListeners(new );
//        net.setListeners(new HistogramIterationListener(1));


        //generer la dataset
        int max_step= 300;
        double[] position= new double[]{-1.2, 0.5} ;
        double[] velocity = new double[]{-0.07, 0.07};
        //generate a dataset
        Mountain_car mountain_car  = new Mountain_car(max_step,position,velocity);
        INDArray states = Nd4j.zeros(2,size_dataset);
        INDArray rewards = Nd4j.zeros(1, size_dataset);
        INDArray eoes = Nd4j.zeros(1, size_dataset);
        INDArray actions = Nd4j.zeros(1, size_dataset);
        INDArray next_states = mountain_car.gen_dataset(size_dataset, states, rewards, eoes, actions);
        plot_dataset(states, actions, next_states);


        //on plot la dataset

        //pour un certain nombre de pas, faire update
        double gamma = 0.99;
        double lrate = 0.5;
        int na = 3;


        DeepQ Q = new DeepQ(net, gamma, lrate, na, position, velocity);

//                RandFQ Q = new RandFQ(gamma, lrate, na); //todo: switch
//
        for (int p = 0; p<num_iterations; p++) {
            String path_iterations = System.getProperty("user.dir")+"/images/iterationsCSV"+p+".csv";
            itlis.flushBuffer();
            System.out.println("Iteration " + p + " on " + num_iterations);
//            Q.update(states, actions, next_states, rewards,eoes,  batchSize, rng, nEpochs);//todo: mettre les actions dans gen_dataset
            Q.update(states, actions, next_states, rewards,eoes,  batchSize, rng, nEpochs);//todo: mettre les actions dans gen_dataset
//            Q.update(states, actions, next_states, rewards,eoes,  batchSize, rng, nEpochs);//todo: mettre les actions dans gen_dataset
           //todo: attention j ai fait add exception to method signature ici (ligne du dessus)
            //on plot les Q-values obtenues
            plot_Q(position, velocity, Q, resolution,p );
            itlis.writeCSV(path_iterations);
        }
        //todo: c est cici qu il faut rajouter le randf


    }


/*
    /** Create a DataSetIterator for training
     * @param x X values
     * @param function Function to evaluate
     * @param batchSize Batch size (number of examples for every call of DataSetIterator.next())
     * @param rng Random number generator (for repeatability)

    private static DataSetIterator getTrainingData(final INDArray x, final INDArray y, final int batchSize, final Random rng) {
        final DataSet allData = new DataSet(x,y);
        final List<DataSet> list = allData.asList();
        Collections.shuffle(list,rng);
        return new ListDataSetIterator(list,batchSize);
    }
*/
    //Plot the data
    /*
    private static void plot(final MathFunction function, final INDArray x, final INDArray y, final INDArray... predicted) {
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");

        for( int i=0; i<predicted.length; i++ ){
            addSeries(dataSet,x,predicted[i],String.valueOf(i));
        }

        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Regression Example - " + function.getName(),      // chart title
                "X",                        // x axis label
                function.getName() + "(X)", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);

        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }
    */
    /*
    private static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
        final double[] xd = x.data().asDouble();
        final double[] yd = y.data().asDouble();
        final XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }
    */
    private static void plot_Q(double [] position, double[] velocity, DeepQ deepQ, int resolution, int i ) throws IOException{
        System.out.println("Creating the image...");
        //verifier parce que je vais avoir un max de pbs avec la shape...
        //building the matrix for plotting
        double coeff_pos = position[1]-position[0];
        double ori_pos = position[0];
        double coeff_velo = velocity[1]-velocity[0];
        double ori_velo = velocity[0];
        //petite boucle en attendant de trouver mieux
        INDArray matrix_tot = Nd4j.zeros(2, resolution*resolution); //todo: attention je sais pas si le carre ca marche
        for (int p = 0; p<resolution; p++) {
            for (int m = 0; m < resolution; m++){
                matrix_tot.put(0, resolution*p+m, p*(coeff_pos)/resolution+ori_pos);
                matrix_tot.put(1, resolution*p+m, m*(coeff_velo)/resolution+ori_velo);
            }
        }
        System.out.println("Computing the results of the Q value...");
        INDArray result = deepQ.net.output(deepQ.normalize(matrix_tot.transpose())).transpose(); //todo: voir si il faut pas faire des iterateurs
        //attention c est la que je normalise mes inputs
        INDArray image_greedy_policy = Nd4j.getExecutioner().exec(new IAMax(result.dup()), 0);
        image_greedy_policy = image_greedy_policy.subi(image_greedy_policy.minNumber());
        image_greedy_policy = image_greedy_policy.divi(image_greedy_policy.maxNumber());
        //todo: changer pour mettre ici le randf

        System.out.println("Result computed.");

        //writes the header of the csv file
        String[] header = {"policyValues"};
        CSVWriter writer = new CSVWriter(new FileWriter(System.getProperty("user.dir")+"/images/policyCSV"+i+".csv"), ',');
        // feed in your array (or convert your data to an array)
        writer.writeNext(header);
        for(int k=0; k<resolution*resolution; k++){
            String[] entries = new String[1];
            entries[0] = String.valueOf(image_greedy_policy.getDouble(0, k));
            // feed in your array (or convert your data to an array)
            writer.writeNext(entries);
        }

        image_greedy_policy = image_greedy_policy.reshape(resolution, resolution);


        String outputFile = System.getProperty("user.dir")+"/images/imagePolicy"+i+".png";
        ImageRender.render(image_greedy_policy, outputFile);
        System.out.println("Image Created.");



        System.out.println("Computing the results of the Q value...");
        INDArray image_Qvalue = Nd4j.getExecutioner().exec(new Max(result.dup()), 0); //todo: voir si je m en foutrais pas du dup
        image_Qvalue = image_Qvalue.subi(image_Qvalue.minNumber());
        image_Qvalue = image_Qvalue.divi(image_Qvalue.maxNumber());
        //todo: changer pour mettre ici le randf
        writer.close();

        System.out.println("Result computed.");

        //writes the header of the csv file
        header[0] = "Qvalues";
        writer = new CSVWriter(new FileWriter(System.getProperty("user.dir")+"/images/qvalueCSV"+i+".csv"), ',');
        // feed in your array (or convert your data to an array)
        writer.writeNext(header);
        for(int k=0; k<resolution*resolution; k++){
            String[] entries = new String[1];
            entries[0] = String.valueOf(image_Qvalue.getDouble(0, k));
            // feed in your array (or convert your data to an array)
            writer.writeNext(entries);
        }
        image_Qvalue =image_Qvalue.reshape(resolution, resolution);
        writer.close();




        String outputFile2 = System.getProperty("user.dir")+"/images/imageValue"+i+".png";
        ImageRender.render(image_Qvalue, outputFile2);
        System.out.println("Image Created.");


        //todo: ici, plutot utiliser la methode predict c est pplus simple

    }

    private static void plot_Q(double [] position, double[] velocity, RandFQ rfQ, int resolution, int i ) throws Exception {
        System.out.println("Creating the image...");
        //verifier parce que je vais avoir un max de pbs avec la shape...
        //building the matrix for plotting
        double coeff_pos = position[1]-position[0];
        double ori_pos = position[0];
        double coeff_velo = velocity[1]-velocity[0];
        double ori_velo = velocity[0];
        //petite boucle en attendant de trouver mieux


        INDArray matrix_tot = Nd4j.zeros(2, resolution*resolution); //todo: attention je sais pas si le carre ca marche
        for (int p = 0; p<resolution; p++) {
            for (int m = 0; m < resolution; m++){
                matrix_tot.put(0, resolution*p+m, p*(coeff_pos)/(double)resolution+ori_pos);
                matrix_tot.put(1, resolution*p+m, m*(coeff_velo)/(double)resolution+ori_velo);
            }
        }

        System.out.println("Computing the results of the Q value...");
        INDArray image_greedy_policy = Nd4j.zeros(1, resolution*resolution);
        INDArray image_Qvalue = Nd4j.zeros(1, resolution*resolution);
        rfQ.predict(matrix_tot, image_Qvalue, image_greedy_policy); //todo: voir si il faut pas faire des iterateurs
        //todo: changer pour mettre ici le randf

        image_greedy_policy = image_greedy_policy.subi(image_greedy_policy.minNumber());
        image_greedy_policy = image_greedy_policy.divi(image_greedy_policy.maxNumber());
        image_greedy_policy = image_greedy_policy.reshape(resolution, resolution); //todo see if tranqpose is needed
        System.out.println("Result computed.");
        String outputFile = System.getProperty("user.dir")+"/images/imagePolicy"+i+".png";
        ImageRender.render(image_greedy_policy, outputFile);
        System.out.println("Image Created.");



        System.out.println("Computing the results of the Q value...");
        image_Qvalue = image_Qvalue.subi(image_Qvalue.minNumber());
        image_Qvalue = image_Qvalue.divi(image_Qvalue.maxNumber());
        image_Qvalue =image_Qvalue.reshape(resolution, resolution);
        //todo: changer pour mettre ici le randf


        System.out.println("Result computed.");
        String outputFile2 = System.getProperty("user.dir")+"/images/imageValue"+i+".png";
        ImageRender.render(image_Qvalue, outputFile2);
        System.out.println("image_Qvalue[0][0] = " + image_Qvalue.getDouble(0,0));
        System.out.println("image_Qvalue[0][resolution-1] = " + image_Qvalue.getDouble(0,resolution-1));
        System.out.println("image_Qvalue[resolution-1][resolution-1] = " + image_Qvalue.getDouble(resolution-1,resolution-1));
        System.out.println("image_Qvalue[resolution-1][0] = " + image_Qvalue.getDouble(resolution-1,0));
        System.out.println("Image Created.");



        //todo: voir quand le writer se close
    }


    public static void plot_dataset(INDArray states, INDArray actions, INDArray next_states) throws IOException {
        System.out.println("Plotting Dataset...");
//        JPanel chartPanel =new ChartPanel(createChart(createDataset(states, actions, next_states)));
//        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));

        int N = states.shape()[1]; //todo verifier que ce soit bien le bon truc


        //writes the header of the csv file
        String[] header = {"state1", "state2", "action", "next_state1", "next_state2"};
        CSVWriter writer = new CSVWriter(new FileWriter(System.getProperty("user.dir")+"/images/dataset.csv"), ',');
        // feed in your array (or convert your data to an array)

        writer.writeNext(header);
        for(int i=0; i<N; i++){
            String[] entries = new String[5];
            entries[0] = String.valueOf(states.getDouble(0, i));
            entries[1] = String.valueOf(states.getDouble(1, i));
            entries[2] = String.valueOf(actions.getDouble(0, i));
            entries[3] = String.valueOf(next_states.getDouble(0, i));
            entries[4] = String.valueOf(next_states.getDouble(1, i));
            // feed in your array (or convert your data to an array)
            writer.writeNext(entries);
        }
        writer.close(); //warning j ai fait add exception avec un truc de signature


        System.out.println("CSV Created.");
        //setContentPane(chartPanel);
    }




    /**
     * Creates a sample chart.
     *
     * @param dataset  the dataset.
     *
     * @return A sample chart.
//     */

//    private static JFreeChart createChart(VectorXYDataset dataset) {
//
//        //todo: rajouter un truc pour mettre de couleurs differentes les differents graphes
//        // todo : il y a surement plein de parametres a changer
//
//        NumberAxis xAxis = new NumberAxis("X");
//        xAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
//        xAxis.setLowerMargin(0.01);
//        xAxis.setUpperMargin(0.01);
//        xAxis.setAutoRangeIncludesZero(false);
//
//        NumberAxis yAxis = new NumberAxis("Y");
//        yAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
//        yAxis.setLowerMargin(0.01);
//        yAxis.setUpperMargin(0.01);
//        yAxis.setAutoRangeIncludesZero(false);
//        VectorRenderer renderer = new VectorRenderer();
//        renderer.setSeriesPaint(0, Color.blue);
//        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
//        plot.setBackgroundPaint(Color.lightGray);
//        plot.setDomainGridlinePaint(Color.white);
//        plot.setRangeGridlinePaint(Color.white);
//        plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5));
//        plot.setOutlinePaint(Color.black);
//        JFreeChart chart = new JFreeChart("Vector Plot Demo 1", plot);
//        chart.setBackgroundPaint(Color.white);
//        return chart;
//    }
//
//    /**
//     * Creates a sample dataset.
//     */
//    private static VectorXYDataset createDataset(INDArray states, INDArray actions, INDArray next_states) {
//        //pour chaque action
//        int N = states.shape()[1];
//        VectorSeriesCollection dataset = new VectorSeriesCollection();
//        //pour commencer on fait simple: je sais qu il n y a que 3 actions. todo: generaliser
//        //todo: truc pas terrible: mon sens d orientation des colonnes n est vraiment pas terrible
//        for (int k = 0; k<N; k++) {
//            /*
//            System.out.println(actions.shape()[1]);
//            System.out.println(k);
//            */
//            double current_action = actions.getDouble(0, k); //on peut ptetre enlever le 1
//            VectorSeries s1 = new VectorSeries("Action 1");
//            VectorSeries s2 = new VectorSeries("Action 2");
//            VectorSeries s3 = new VectorSeries("Action 3");
//            double x_or = states.getDouble(0, k);
//            double y_or  =  states.getDouble(1,k);
//            double x_add = next_states.getDouble(0, k) - x_or;
//            double y_add = next_states.getDouble(1, k) - y_or;
//            switch ((int) current_action){
//                case 0: s1.add(x_or, y_or, x_add, y_add);
//                    break; //todo: verif si on en a bien besoin
//                case 1: s2.add(x_or, y_or, x_add, y_add);
//                    break;
//                case 2: s3.add(x_or, y_or, x_add, y_add);
//                    break;
//            }
//            dataset.addSeries(s1);
//            dataset.addSeries(s2);
//            dataset.addSeries(s3);
//        }
//        return dataset;
//    }
    /** Returns the network configuration, 2 hidden DenseLayers of size 50.
     */
    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
//<<<<<<< HEAD
//        final int numHiddenNodes = 6;
//=======
        final int numHiddenNodes = 16;
//>>>>>>> 19b0a6be78d36e2ab13c2897376adacc3b76c5c9
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .weightInit(WeightInit.ZERO)
//                        .dropOut(0.5) //bonne idee de mettre du dropout sur la derniere couche ?
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
    }


}


//todo: il y a qqch qui cloche avec ca essayer de regler





