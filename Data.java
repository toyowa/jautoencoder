/*
 * (C) 2017 Toyoaki WASHIDA
 */
package jdeeplearning;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import static jdeeplearning.JDeepLearning.reserved_weights;
import static jdeeplearning.JDeepLearning.DEBUG;

/**
 *
 * @author Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 */
public class Data {
    File m_trainingDataFile;
    BufferedReader dataReader;
    
    boolean setAutoencoderFinalWeight(String weightFileName, int layerLoc){
        File file = new File(weightFileName);
        BufferedReader weight_file;
        String[] terms;
        try {
            weight_file = new BufferedReader(new FileReader(file));
            String line;
            System.out.println("ウェイトを読み込みます");
            while ((line = weight_file.readLine()) != null) {
                terms = line.split("\\s+");
                // ウェイトは、きちんと順番に入っているはず
                if ("Weight:".equals(terms[0])) {
                    if(terms.length < 8) {
                        System.out.println("不適切なウェイトデータが混ざっています");
                        System.out.println("terms.length: "+terms.length);
                        System.out.println("データ行: " + line);
                        return false;
                    }
                    int wnum = Integer.valueOf(terms[1]); // 前のレイヤーのニューロン番号
                    int lnum = Integer.valueOf(terms[3]); // レイヤー番号
                    int nnum = Integer.valueOf(terms[5]); // ニューロン番号
                    double weight = Double.valueOf(terms[7]);
                    if(lnum == 1){
                        if (DEBUG) System.out.println("Fine tuning: layerNo:"+layerLoc+" neuronNo:"+nnum+" weightNo:"+wnum+" = "+weight);
                        reserved_weights[layerLoc][nnum][wnum] = weight;
                    }
                }
            }
            weight_file.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
        return true;
    }
    
    void setTrainData(String dataPath){
        m_trainingDataFile = new File(dataPath);
        try {
            dataReader = new BufferedReader(new FileReader(m_trainingDataFile));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    void rewindData(){
        // データを読み出すためにバッファリーダーを作り直す
        try {
            dataReader.close();
            dataReader = new BufferedReader(new FileReader(m_trainingDataFile));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    boolean isReady(){
        boolean ret = false ;
        try {
            ret = dataReader.ready();
        } catch (IOException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
        return ret;
    }
    
    int openWeightFile(String weight_path) {
        String[] terms;
        if (weight_path.length() > 0) {
            //System.out.println("weight_path = "+weight_path);
            terms = weight_path.split("\\.");
            //System.out.println("terms[terms.length - 1] = "+terms[terms.length - 1]);
            if (!"wgt".equals(terms[terms.length - 1])) {
                System.out.println("正しいweightファイルではありません：中止 weight_path = "+weight_path);
                return -1;
            }
        }
        File file = new File(weight_path);
        BufferedReader weight_file = null;
        int iteration = 0;
        try {
            weight_file = new BufferedReader(new FileReader(file));
            String line;
            int [] topology = null;
            System.out.println("ウェイトを読み込みます");
            while ((line = weight_file.readLine()) != null) {
                terms = line.split("\\s+");
                // ウェイトは、きちんと順番に入っているはず
                if ("Weight:".equals(terms[0])) {
                    if(terms.length < 8) {
                        System.out.println("不適切なウェイトデータが混ざっています");
                        System.out.println("terms.length: "+terms.length);
                        System.out.println("データ行: " + line);
                        return 0;
                    }
                    int wnum = Integer.valueOf(terms[1]); // 前のレイヤーのニューロン番号
                    int lnum = Integer.valueOf(terms[3]); // レイヤー番号
                    int nnum = Integer.valueOf(terms[5]); // ニューロン番号
                    double weight = Double.valueOf(terms[7]);
                    reserved_weights[lnum-1][nnum][wnum] = weight;
                    if (DEBUG) System.out.println("Fine tuning: layerNo:"+lnum+" neuronNo:"+nnum+" weightNo:"+wnum+" = "+weight);
                } else if ("Topology:".equals(terms[0])) {
                    // トポロジーデータは、ウェイトデータに先立ってある
                    topology = new int[terms.length-1];
                    System.out.print("ウェイトのネットワークトポロジー: ");
                    for(int i=0;i<terms.length-1;i++){
                        topology[i] = Integer.valueOf(terms[i+1].trim());
                        System.out.print(topology[i]+" ");
                    }
                    System.out.println();
                    int maxNeuron = 0;
                    for(int i=0;i<topology.length;i++){
                        if(topology[i] > maxNeuron) maxNeuron = topology[i];
                    }
                    // ニューロン数とウェイト数は、最大ニューロン数にしておく、若干無駄があるが
                    JDeepLearning.reserved_weights = new double[topology.length-1][maxNeuron][maxNeuron];
                } else if ("Iteration:".equals(terms[0])) {
                    iteration = Integer.valueOf(terms[1]);
                    if (DEBUG) {
                        System.out.println("weightファイルから読み込んだ繰り返し数 = " + iteration);
                    }
                }
            }
            weight_file.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println("weightファイルの読み込みを終了しました");
        return iteration;
    }

    double[] getTargetOutputs() {
        String line = "";
        double[] outputdata = null;
        try {
            while ((line = dataReader.readLine()) != null) {
                String[] terms = line.split(" ");
                String label = terms[0].trim();
                if ("out:".equals(label)) {
                    outputdata = new double[terms.length - 1];
                    for (int i = 0; i < terms.length - 1; i++) {
                        outputdata[i] = Double.valueOf(terms[i + 1].trim());
                    }
                    break;
                }
            }
        } catch (IOException ex) {
            System.out.println("データからの読み込みに失敗しています：ファイルの終わりかもしれません");
            //Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
        return outputdata;
    }

    void fileClose(){
        try {
            dataReader.close();
        } catch (IOException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    double [] getNextInputs(){
        String line = "";
        double [] inputdata = null;
        try {
            while((line = dataReader.readLine()) != null){
                if (DEBUG) {
                    System.out.println("データファイルから読み込んだ line = " + line);
                }
                String[] terms = line.split(" ");
                String label = terms[0].trim();
                if ("in:".equals(label)) {
                    inputdata = new double[terms.length - 1];
                    for (int i = 0; i < terms.length - 1; i++) {
                        inputdata[i] = Double.valueOf(terms[i + 1].trim());
                    }
                    break;
                }
            }
        } catch (IOException ex) {
            System.out.println("データからの読み込みに失敗しています：ファイルの終わりかもしれません");
            //Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
        return inputdata;
    }

    
    int [] getTopology() {
        // ネットワークのトポロジーだけを得る
        String line = "";
        try {
            line = dataReader.readLine();
        } catch (IOException ex) {
            Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
        }
        String [] terms = line.split(" ");
        String label = terms[0].trim();
        int [] topology = null;
        if("topology:".equals(label)){
            topology = new int[terms.length-1];
            for(int i=0;i<terms.length-1;i++){
                topology[i] = Integer.valueOf(terms[i+1].trim());
            }
        }else{
            System.out.println("トポロジーの取得に失敗しています");
        }
        return topology;
    }
}
