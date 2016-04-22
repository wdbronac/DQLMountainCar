package regression.reinforce;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by jspw4161 on 21/04/16.
 */



import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetPreProcessor;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ReinforcementDataSetIterator implements DataSetIterator{
    /**
    * Implements a DataSetIterator for reinforcement learning. The DataSet should be of this form:
     * Features: as before
     * Labels: [the new Qvalue we want to fit , the index of he action corresponding]
    *
    *
    * */
        private static final long serialVersionUID = -7569201667767185411L;
        private int curr = 0;
        private int batch = 10;
        private DataSetPreProcessor preProcessor;
        private MultiLayerNetwork net;
        DataSet data;
//        private INDArray features;
//        private INDArray labels;

        public ReinforcementDataSetIterator(DataSet data, int batch,MultiLayerNetwork net) {
            this.batch = batch;
            this.net = net;
            if(preProcessor != null)
                preProcessor.preProcess(data);
            this.data = data;
//            this.features = data.getFeatures();
//            this.labels = data.getLabels();
        }

        /**
//         * Initializes with a batch of 5
//         * @param coll the collection to iterate over
//         */
//        public ReinforcementDataSetIterator(Collection<DataSet> coll) {
//            this(coll,5);
//
//        }

        @Override
        public synchronized boolean hasNext() {
            return curr < data.getLabels().shape()[0];
        }

        @Override
        public synchronized DataSet next() {
            return next(batch);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int totalExamples() {
            return data.getLabels().shape()[0];
        }

        @Override
        public int inputColumns() {
            return data.getFeatures().columns();
        }

        @Override
        public int totalOutcomes() {
            return net.getLabels().columns(); //todo:: I do not know this function so I should verify it is good
        }

        @Override
        public synchronized void reset() {
            curr = 0;
            data.shuffle();
            //todo: see if I should put the "shuffle" method
        }

        @Override
        public int batch() {
            return batch;
        }

        @Override
        public synchronized int cursor() {
            return curr;
        }

        @Override
        public int numExamples() {
            return data.getLabels().shape()[0];
        }

        @Override
        public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
            this.preProcessor = (DataSetPreProcessor) preProcessor;
        }

        @Override
        public List<String> getLabels() {
            return null;
        }


        @Override
        public DataSet next(int num) {
            int end = curr + num;
            INDArray labels = data.getLabels();
            int N = labels.shape()[0];
            if(end >= N)
                end = N;

            INDArray batchFeatures = data.getFeatures().dup().get(NDArrayIndex.interval(curr, end), NDArrayIndex.all()); //todo: maybe no need for dup()
            INDArray batchLabels = net.output(batchFeatures);

            for(int n=0; curr< end; n++) {
                //put the value "labels.getDouble(n,0)" for action "labels.getDouble(n,1)" in the labels array
                batchLabels.put(n,(int) labels.getDouble(curr,1),labels.getDouble(curr,0));
                curr++;
            }

            DataSet d = new DataSet(batchFeatures, batchLabels);

            return d;

        }

}

