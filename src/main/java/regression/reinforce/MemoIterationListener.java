package regression.reinforce;

import au.com.bytecode.opencsv.CSVWriter;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by jspw4161 on 21/04/16.
 */
public class MemoIterationListener implements org.deeplearning4j.optimize.api.IterationListener{




    /*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

//    package org.deeplearning4j.optimize.listeners;

    /**
     * Score iteration listener
     *
     * @author Adam Gibson
     */
        private int printIterations = 10;
        private static final Logger log = LoggerFactory.getLogger(org.deeplearning4j.optimize.listeners.ScoreIterationListener.class);
        private boolean invoked = false;
        private List<Double> buffer;


    //todo: here i use a buffer for writing at the end fr more speed during training. I could also write time iteration after iteration and try to
    //watch memory issues
        public MemoIterationListener(int printIterations) throws IOException {
            this.printIterations = printIterations;
            this.buffer = new ArrayList<Double>();
        }


    //todo: implementer des constructeurs par defaut
//        public MemoIterationListener(String path) {
//        }

        @Override //warning j ai fait passer le langage a niveau 6 ou je sais pas quoi
        public boolean invoked(){ return invoked; }

        @Override
        public void invoke() { this.invoked = true; }

        @Override
        public void iterationDone(Model model, int iteration) {

            if(printIterations <= 0)
                printIterations = 1;
            if(iteration % printIterations == 0) {
                invoke();
                double result = model.score();
                log.info("Score at iteration " + iteration + " is " + result);
                buffer.add(result);
//todo: absolument faire qqch pour closer le writer
            }
        }

        public void writeCSV(String path) throws IOException { //todo: je pourrais specifier le path uniquement a la fin limite
            CSVWriter writer= new CSVWriter(new FileWriter(path), ',');
            String[] entries = new String[1];
            for (Double e:buffer){
                entries[0] = String.valueOf(e);
                writer.writeNext(entries);
            }
            writer.close();
        }

        public  void flushBuffer(){
            buffer.clear();
        }
    }
