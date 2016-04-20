package regression.reinforce;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Max;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.*;
import weka.classifiers.trees.lmt.SimpleLinearRegression;
import weka.core.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by william on 25/03/16.
 */
public class RandFQ {

//    public MultiLayerNetwork net;
    public double gamma;
    public double lrate;
    public int na;
//    public SimpleLinearRegression rf;
    public RandomForest rf;
//    public SMOreg rf;
    public int t  =0;

    public RandFQ(double gamma, double lrate, int na) {
//        this.net = net;
        this.gamma = gamma;
        this.lrate = lrate;
        this.na = na;
//        this.rf = new SimpleLinearRegression();
//        this.rf = new SimpleLogistic();
        this.rf = new RandomForest();
//        this.rf = new SMOreg();
//        this.rf.setNumTrees(50);
//        this.rf.setNumFeatures(10); //todo : je ne suis pas sur
//        this.rf.setMaxDepth(10); //todo : je ne suis pas sur




    }

//todo : peut etre convertir les inputs avant l'appel a la fonction



    public void update(INDArray states_input, INDArray actions,
                  INDArray next_states, INDArray rewards, INDArray eoes,
                  int batchSize, Random rng , int nEpochs) throws Exception { //update le reseau avec les inputs donnes,
                    // et les actions suivantes possibles (fait un fitting)
        // c est la grosse fonction du programme
        INDArray states_input_copy = states_input.dup();
        INDArray actions_copy = actions.dup();
        INDArray next_states_copy = next_states.dup();
        int N = states_input_copy.shape()[1];


//        FastVector atts = new FastVector(4);
        ArrayList<Attribute> atts = new ArrayList<Attribute>(4);
        Attribute position = new Attribute("position");
        Attribute speed= new Attribute("speed");
        Attribute action= new Attribute("action");
        Attribute qvalue = new Attribute("qvalue");
        atts.add(position);
        atts.add(speed);
        atts.add(action);
        atts.add(qvalue);
        Instances newDataset = new Instances("Dataset", atts, N);
        newDataset.setClassIndex(newDataset.numAttributes() - 1);

//        atts.addElement(current); //todo peut etre que finalement je n ai pas besoin de les ajouter dans le fast vector ?

        Instance instance = new DenseInstance(4); //todo: peut etre qu il ne faut pas mettre en entree un tableau pour s a


        //todo : a faire a chaque etape

        for(int p=0; p<N;p++) { //todo: verifier que c est bien ca
            instance.setValue(0, states_input_copy.getDouble(0,p)); //verifier selon sa taille
            instance.setValue(1, states_input_copy.getDouble(1,p));
            instance.setValue(2, actions_copy.getDouble(0, p)); //todo: peut etre pas ca
//            instance.setMissing(3);
            double newQvalue;
            //computes the new qvalue we want to fit
            if (t==0) {//if it is the first ime we go in the loop (he first iteration of the program
                newQvalue = rewards.getDouble(0, p);
            }
            else{
                int numactions = 3;
                double[] oldvalues = new double[numactions]; //todo verifier shape 0
                    Instance instanceForPrediction = new DenseInstance(4); //todo: automatiser le 2
                    instanceForPrediction.setValue(0, next_states_copy.getDouble(0,p));
                    newDataset.setClassIndex(newDataset.numAttributes() - 1); //todo: c est peut etre n importe quoi de le mettre la
                    instanceForPrediction.setDataset(newDataset);//c est vraiment pas optimisé voir comment je peux faire pour l optimiser
                    instanceForPrediction.setValue(1, next_states_copy.getDouble(1, p));
                for(int k = 0; k<numactions; k++){
                    instanceForPrediction.setValue(2, k);
                    oldvalues[k] = rf.classifyInstance(instanceForPrediction);
//                    oldvalues[k] = rf.distributionForInstance(instanceForPrediction)[0];
//                    oldvalues[k] = rf.distributionForInstance(instanceForPrediction)[0];
                }

                newQvalue = rewards.getDouble(0, p) +this.gamma*(1-eoes.getDouble(0,p))*getMaxValue(oldvalues);
                if (Math.abs(newQvalue) >1){
                    System.out.println("probleme ici");
                }
            }
            instance.setValue(3, newQvalue); //todo verifier comment il sait que la qvalue c est le label
//            if(n){
////                System.out.println("la valeur est positive");
//            }
            newDataset.add(instance);
        }


        newDataset.setClassIndex(newDataset.numAttributes() - 1);

        rf.buildClassifier(newDataset);
        t++;

        }





        //todo: tout ça c'est faux pcq il faut que je ma rf prenne en argument state ET une action, et qu elle predise la valeurQ(s,a)
//
//    //todo conversino de ma dataset en instances weka
//
//
//        FastVector atts = new FastVector();
//
//
//        Attribute current = new Attribute("input", ) ;
//        Attribute[] input = new Attribute[4];
//        Attribute[] label = new Attribute();
//
//
//        for(Attribute att: input) {
//            atts.addElement(att);
//        }
//        for(Attribute att: label) {
//            atts.addElement(att);
//        }
//
//
//        Instances newDataset = new Instances("Dataset", atts, 10);
//
//
//        Instance instance = new Instance(2 + 3 * gamesize + 1);
//
//        instance.setValue(attributes[0],1)
//
//        Instance instance =
//        newDataset.add(instance);
//
//        for(int obj = 0; obj < numInstances; obj++)
//        {
//            instances.add(new SparseInstance(numDimensions));
//        }
//
//        List<Instance> instances = new ArrayList<Instance>();
//        for(int dim = 0; dim < numDimensions; dim++)
//        {
//            // Create new attribute / dimension
//            Attribute current = new Attribute("Attribute" + dim, dim);
//            // Create an instance for each data object
//            if(dim == 0)
//            {
//                for(int obj = 0; obj < numInstances; obj++)
//                {
//                    instances.add(new SparseInstance(numDimensions));
//                }
//            }
//
//            // Fill the value of dimension "dim" into each object
//            for(int obj = 0; obj < numInstances; obj++)
//            {
//                instances.get(obj).setValue(current, data[dim][obj]);
//            }
//
//            // Add attribute to total attributes
//            atts.addElement(current);
//        }
//
//// Create new dataset
//        Instances newDataset = new Instances("Dataset", atts, instances.size());
//
//// Fill in data objects
//        for(Instance inst : instances)
//            newDataset.add(inst);
//
////
////
////
////        ArrayList<Attribute> atts = new ArrayList<Attribute>(2);
//        ArrayList<String> classVal = new ArrayList<String>();
//        classVal.add("A");
//        classVal.add("B");
//        atts.addElement(new Attribute("content",(ArrayList<String>)null));
//        atts.add(new Attribute("@@class@@",classVal));
//
//        Instance instance_to_predict = ;
//        INDArray oldQvalues = this.rf.distributionForInstance(instance_to_predict);
//        INDArray bestnextvalues = Nd4j.zeros(1,N);//le max selon toutes les values de next_state
//        INDArray gactions =  Nd4j.zeros(1,N);
//
//
//        int numFolds = 10;
//
//        Instances trainData = new Instances();
//        trainData.setClassIndex(trainData.numAttributes() - 1);
//        br.close();
//        RandomForest rf = new RandomForest();
//        rf.setNumTrees(100); //todo : voir ce que matthieu geist met dans son code
//
//
//
//        // on ajoute un novuel argument, le label
//        FastVector nominalFv = new FastVector(labels.length);
//
//        for (int i = 0; i < labels.length; i++) {
//            nominalFv.addElement(labels[i]);
//        }
//
//        label = new Attribute("label", nominalFv);
//
//        fastVector = new FastVector();
//        fastVector.addElement(label);
//        for (Attribute a : attributes) {
//            fastVector.addElement(a);
//        }
//
//        return fastVector;
//    }
//
//    /**
//     * @return the fastVector
//     */
//    public FastVector getFastVector() {
//        return fastVector;
//
//
//
//
//
//







//
//        this.predict(next_states, bestnextvalues, gactions) ; // il y a un pb a regler avec l action
//        INDArray betterQvalues = rewards.add(bestnextvalues.dup().mul(gamma)); //todo: verifier s il y a bien une copie
//
//        //INDArray betterQvalues  = this.net.output(next_states); //todo: voir pcq il y a un pb avec
//        // les actions
//        /*todo: je pense que
//        il y a beaucoup trop de Qold dans l affaire ca peut se simplifier mais pour l instant on laisse
//        */
//        //todo: aussi il faut que je change tous les tableaux en nd4j je pense
//        INDArray newQvalues = oldQvalues.dup();
//        System.out.println(states_input_copy.shape()[0]);
//        System.out.println(states_input_copy.shape()[1]);
//        System.out.println(oldQvalues.shape()[0]);
//        System.out.println(oldQvalues.shape()[1]);
//        //todo: if ( c est l action choisie):
//        for (int k = 0; k<N; k++) { //parcourir chaque transition
//            int index_action = (int) actions.getDouble(0,k);
//            System.out.println(index_action);
//            newQvalues.put(index_action, k ,oldQvalues.getDouble(index_action, k) + betterQvalues.getDouble(0, k)
//                    - oldQvalues.getDouble(index_action, k) * lrate);
//        }
//        //TODO: faire en sorte de regler quand on donne la reward
//        System.out.println(oldQvalues.shape()[0]);
//        System.out.println(oldQvalues.shape()[1]);
//        System.out.println(newQvalues.shape()[0]);
//        System.out.println(newQvalues.shape()[1]);
//        final DataSetIterator iterator = getTrainingData(states_input_copy,newQvalues.transpose(), batchSize,rng);
//        train_net(nEpochs,  iterator);
//
//
//    }

    public void predict(INDArray states, INDArray v, INDArray gactions) throws Exception { //todo: doit retourner les values et les greedy actions en tout cas dans le python
        /*
        int N = states.shape()[1];
        // todo : initialiser  a 0 les reward et les eoe meme si on s en fout
        //todo : verifier que je mets bien des nd4j partout
        INDArray rewards = Nd4j.zeros(1, N);
        //int[] rewards = new int[N];
        INDArray eoes = Nd4j.zeros(1, N);
        //boolean[] eoes = new boolean[N];
        //return this.net.output(next_states);
        for (int k = 0; k<na; k++) {
            int action = k;
            INDArray action_tab = Nd4j.ones(1,N).mul(k);
            //INDArray next_next_states = mountain_car.transition(next_states, action_tab, rewards, eoes); //todo: regler le pb du non static
            //compare avec le precedent vecteur et garde uniquement la valeur la plus
            //grande et l action correspondante
            INDArray newValue = net.output(next_states.transpose()).transpose(); // ici j assimile state, a , a next_state //todo: enlever les transpose et remettre bien
            for (int p = 0; p<N ; p++){ //todo: regler length ca doit etre shape ou qqch comme ca
                System.out.println(v.shape()[0]);
                System.out.println(newValue.shape()[0]);
                if (v.getDouble(k,p) <  newValue.getDouble(k,p)) {
                    v.put(0,p, newValue.getDouble(k, p));
//                    gactions.put(0, p, k);
//                }
//            } //on a notre v et notre gactions
//        }
//        */
//        INDArray newValues = net.output(states.dup().transpose()).transpose();
//        System.out.println(newValues.shape()[0]);
//        System.out.println(newValues.shape()[1]);
//        v.addi(Nd4j.getExecutioner().exec(new Max(newValues.dup()), 0));
//        gactions.addi(Nd4j.getExecutioner().exec(new IAMax(newValues.dup()), 0));


        //todo: c est degueu il faudrait que je mette tout ca dans le constructeur mais deja on va voir si ca marche...
        int N = states.shape()[1];
        ArrayList<Attribute> atts = new ArrayList<Attribute>(4);
        Attribute position = new Attribute("position");
        Attribute speed= new Attribute("speed");
        Attribute action= new Attribute("action");
        Attribute qvalue = new Attribute("qvalue");
        atts.add(position);
        atts.add(speed);
        atts.add(action);
        atts.add(qvalue);
        Instances newDataset = new Instances("Dataset", atts, 10);
        newDataset.setClassIndex(newDataset.numAttributes() - 1);

        double[] oldvalues = new double[3]; //todo: a generaliser
        int numactions = 3; //todo: automatiser ca0

        for (int p = 0; p<N; p++) {
            Instance instanceForPrediction = new DenseInstance(4); //todo: automatiser le 2
            instanceForPrediction.setValue(0, states.getDouble(0, p));
            instanceForPrediction.setValue(1, states.getDouble(1, p));
            for (int k = 0; k < numactions; k++) {
                instanceForPrediction.setValue(2, k);
                instanceForPrediction.setDataset(newDataset);
//                oldvalues[k] = rf.distributionForInstance(instanceForPrediction)[0]; //attention j ai fait ajouter exception puor method signature
                oldvalues[k] = rf.classifyInstance(instanceForPrediction); //attention j ai fait ajouter exception puor method signature
            }
            v.put(0,p, getMaxValue(oldvalues));

//
//            //for debugging:
//            if(getMaxValue(oldvalues) >= 1){
//                System.out.println("la valeur est positive ici");
//            }

            gactions.put(0, p, getMaxIndex(oldvalues));
        }

    }

//    public void train_net(int nEpochs, DataSetIterator iterator) {
//        //Train the network on the full data set, and evaluate in periodically
//        for (int i = 0; i < nEpochs; i++) {
//            //iterator.reset(); // todo: voir si je le mets ou pas
//            net.fit(iterator);
//        }
//    }


//    private static DataSetIterator getTrainingData(final INDArray x, final INDArray y, final int batchSize, final Random rng) {
//        final DataSet allData = new DataSet(x,y);
//        final List<DataSet> list = allData.asList();
//        Collections.shuffle(list,rng);
//        return new ListDataSetIterator(list,batchSize);
//    }
//

    public static double getMaxValue(double[] array){
        double maxValue = array[0];
        for(int i=1;i < array.length;i++){
            if(array[i] > maxValue){
                maxValue = array[i];

            }
        }
        return maxValue;
    }

    public static int getMaxIndex(double[] array){
        double maxValue = array[0];
        int index = 0;
        for(int i=1;i < array.length;i++){
            if(array[i] > maxValue){
                maxValue = array[i];
                index = i;
            }
        }
        return index;
    }



}
