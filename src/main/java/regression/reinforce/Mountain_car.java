package regression.reinforce;
import java.lang.Math;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Floor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by william on 25/03/16.
 */
public class Mountain_car {
    public int max_step;
    public double[] position;
    public double[] velocity;

    //this is the initialisation of the mountain car
    public Mountain_car(int max_step, double[] position, double[] velocity) {
        this.max_step = max_step;
        this.position = position;
        this.velocity = velocity;
    }


    public INDArray transition(INDArray state, INDArray action, INDArray r, INDArray eoe) {
        /*
        System.out.println(action.shape()[0]);
        System.out.println(action.shape()[1]);
        System.out.println(state.shape()[0]);
        System.out.println(state.shape()[1]);
        System.out.println(state.getRow(0).length());
        */
        INDArray next_state = Nd4j.zeros(2, state.shape()[1]);
        next_state.putRow(1, state.getRow(1).addi((action.addi(-1)).addi(cos(state.getRow(0).mul(3)).mul(-0.0025)).mul(0.001)));
        next_state.putRow(0, state.getRow(0).addi(next_state.getRow(1)));
        for (int i = 0; i < state.shape()[1]; i++) {
            if (next_state.getDouble(0,i) > this.position[1]) {
                r.put(0, i, 10);
                eoe.put(0,i, 1);
            }
            if (next_state.getDouble(0,i) < this.position[0]) {
                next_state.put(0, i, 0);
                r.put(0,i,-10);
            }
        }
        // je l ai fait avec un test de merde mais voir comment utiliser la fonction max
        //TODO: regler ce pb de min max
        for (int k = 0; k < state.shape()[1]; k++) {
            next_state.put(0,k, Math.min(Math.max(next_state.getDouble(0, k), this.position[0]), this.position[1]));
            next_state.put(1, k, Math.min(Math.max(next_state.getDouble(0, k), this.velocity[0]), this.velocity[1]));
            //ok c est cette methode qu il faut utiliser
        }
        return next_state;
    }


    public INDArray gen_dataset(int n, INDArray states, INDArray rewards, INDArray eoes, INDArray actions) {
        //INDArray states = Nd4j.zeros(2,n);
        //INDArray next_states = Nd4j.zeros(2, n);
        //int[] rewards = new int[n];
        //boolean[] eoes = new boolean[n];
        //fill the states array:
        states.putRow(0, Nd4j.rand(1,n).mul(this.position[1] - this.position[0]).addi(this.position[0]))  ;
        states.putRow(1, Nd4j.rand(1,n).mul(this.velocity[1] - this.velocity[0]).addi(this.velocity[0]));
        //actions = Nd4j.rand(1, n).mul(3); //todo: verifier si c est bien affecte
        actions = Nd4j.getExecutioner().execAndReturn(new Floor(Nd4j.rand(1, n).mul(3))); //todo: verifier si c est bien affecte
        INDArray next_states = transition(states, actions, rewards, eoes);
        return next_states;
    }

    public void sample_traj(DeepQ Q) { // todo: attention si ca marche pas c est ptetre pcq mes fonctions ne gerent pas les tableaux a un element

        int cmpt = 0;
        INDArray trajectory = Nd4j.zeros(2, this.max_step);
        INDArray eoe =Nd4j.zeros(1,1);
        INDArray state = Nd4j.create(new double[]{-Math.PI/6, 0}, new int[] {2, 1}); //todo: verif que ce soit la bonne shape
        while (eoe.getDouble(0,0) == 0 && cmpt <this.max_step) {
            trajectory.put(cmpt, state); // todo : a finir
            //je pense que je m en fous je peux mettre n importe quel v et action
            // apres il faudra voir si c est pas inutile computationnellement
            INDArray v = Nd4j.zeros(1, this.max_step);
            INDArray action = Nd4j.zeros(1, this.max_step);
            INDArray rewards = Nd4j.zeros(1,1);
            //todo: reserver l espace max et apres ne prendre que les elements jusqu au dernier point
            Q.predict(state, v, action); // TODO: predict doit retourner les values et les greedy actions
            INDArray next_state = this.transition(state, action, rewards, eoe);
            state = next_state;
            cmpt += 1;
        }
        trajectory = trajectory.get(NDArrayIndex.interval(0, cmpt));
        return;
    }
}


