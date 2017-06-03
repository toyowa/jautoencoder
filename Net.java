/*
 * (C) 2017 Toyoaki WASHIDA
 */
package jdeeplearning;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.logging.Level;
import java.util.logging.Logger;
import static jdeeplearning.JDeepLearning.DEBUG;

/**
 *
 * @author Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 */
public class Net {
    double [] outputs;
    Layer [] layers;
    int [] topology;
    double adjusting;

    Net(int [] topology) {
        this.topology = topology;
        adjusting = 0.15; // ウェイトを改訂する程度
        // topologyに基づいて、ネットワークを構成する
        layers = new Layer[topology.length];
        for (int i = 0; i < topology.length; i++) {
            Layer layer = new Layer(topology,i,adjusting);
            layers[i] = layer;
        }
        // 入力側のウェイトと出力側のウェイトの整合性を図る
        for (int i = topology.length - 1; i > 0; i--) {
            adjustingPreLayerWeightsOut(i);
        }
    }

    Layer getOutputLayer() {
        return layers[topology.length- 1];
    }

    String makeAutoencoderData(Data dataProc){
        // dataProcは、元々のデータではなく、前のautoencoderから出力されたデータの場合もある
        // 次のステップの入力データを作成する
        Calendar c = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHmmss");
        String filename = "tmpout_" + sdf.format(c.getTime()) + "." + "txt";
        File file = new File(filename);
        FileWriter writing_file;
        // データを最初から読み直す
        dataProc.rewindData();
        double[] inputVals;
        try {
            writing_file = new FileWriter(file);
            // トポロジーは使わないのだが、念のために保存しておく
            writing_file.append("Topology: ");
            for (int i = 0; i < topology.length; i++) {
                writing_file.append(topology[i] + " ");
            }
            writing_file.append("\n");
            while(dataProc.isReady()){
                inputVals = dataProc.getNextInputs();
                if(inputVals == null) break;
                getForwardOutput(inputVals);
                writing_file.append("in: ");
                // 第１隠れ層の出力値だけを保存する
                Layer layer1 = layers[1];
                for (int i = 0; i < layer1.layerOutput.length; i++) {
                    writing_file.append(layer1.layerOutput[i] + " ");
                }
                writing_file.append("\n");
            }
            writing_file.close();
        } catch (IOException ex) {
            Logger.getLogger(Net.class.getName()).log(Level.SEVERE, null, ex);
        }
        return filename;
    }
    
    String printAllWeights(int iteration, String dataFileName) {
        Calendar c = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHmmss");
        String filename = "weight_" + sdf.format(c.getTime()) + "." + "wgt";
        File file = new File(filename);
        FileWriter writing_file;
        try {
            writing_file = new FileWriter(file);

            System.out.println("形成されたウェイトをファイル[ " + filename + " ]に書き出します");
            writing_file.append("File name: [ " + filename + " ]\n");
            writing_file.append("Iteration: " + iteration + "\n");
            writing_file.append("InputData: " + dataFileName + "\n");
            writing_file.append("Topology: ");
            for (int i = 0; i < topology.length; i++) {
                writing_file.append(topology[i] + " ");
            }
            writing_file.append("\n");
            writing_file.append("Adjusting: " + adjusting + "\n");
            writing_file.append("Label: Pre_neuron No." + " -> " + " Layer No. : Neuron No. = Weight\n");
            for (int i = 1; i < topology.length; i++) {
                Layer layer = layers[i];
                layer.printAllWeights(writing_file);
            }
            writing_file.close();
        } catch (IOException ex) {
            Logger.getLogger(Net.class.getName()).log(Level.SEVERE, null, ex);
        }
        return filename;
    }

    void adjustingPreLayerWeightsOut(int layerNo) {
        Layer this_layer = layers[layerNo];
        Neuron [] this_neurons = this_layer.neurons;
        Layer pre_layer = layers[layerNo - 1];
        Neuron [] pre_neurons = pre_layer.neurons;
        for (int i = 0; i < pre_neurons.length; i++) {
            Neuron pre_neuron = pre_neurons[i];
            for (int j = 0; j < this_neurons.length; j++) {
                Neuron this_neuron = this_neurons[j];
                // 一つ前のレイヤー(i番目)の出力側j番目のニューロンに向けてのウェイトは、
                // このレイヤー(j番目)の入力側第i番目のニューロンからのウェイトに同じ
                // だから、コピーする
                if (DEBUG) {
                    System.out.println("adjustingPreLayerWeightsOut L:N[" + (layerNo - 1) + ":" + i + "] ==>> " + "L:N[" 
                            + layerNo + ":" + j + "] 出力ウェイト調整 = " + this_neuron.weightsIn[i]);
                }
                pre_neuron.weightsOut[j] = this_neuron.weightsIn[i];
            }
        }
    }

    void reviseAllNetworkWeights() {
        for (int i = 1; i < topology.length; i++) {
            Layer layer = layers[i];
            layer.reviseNetworkWeights();
        }
    }

    void execBackpropagation(double [] deltaE) {
        // バックプロパゲーションの実行
        Layer layerOut = layers[topology.length - 1];
        if (DEBUG) {
            System.out.println("execBackpropagation 出力レイヤーのDelta値を計算する:他と計算方法が異なるので");
        }
        for (int i=0;i<layerOut.neurons.length;i++) {
            Neuron neuron = layerOut.neurons[i];
            neuron.getNeuronDelta(deltaE);
            layerOut.layerDelta[i] = neuron.delta;
        }
        if (DEBUG) {
            JDeepLearning.showVectorVals("execBackpropagation 出力層のDelta値", layerOut.layerDelta);
        }
        // 逆方向に計算を進める
        // 入力層については計算しない：ウェイトは2層分しかないので
        // i=0 はない
        for (int i = topology.length - 2; i >= 1; i--) {
            Layer layer = layers[i];
            Layer prev = layers[i + 1];
            if (DEBUG) {
                JDeepLearning.showVectorVals("execBackpropagation 一つ前の層のDelta", prev.layerDelta);
            }
            if (DEBUG) {
                System.out.println("execBackpropagation 層 [ " + i + " ] のDelta計算");
            }
            layer.getLayerDelta(prev.layerDelta);
            //adjustingPreLayerWeightsOut(i);
            if (DEBUG) {
                JDeepLearning.showVectorVals("execBackpropagation このレイヤーのDelta", layer.layerDelta);
            }
        }
        // ウェイトの変更は、デルタを全て作り終えてからにしなければならない
        // ウェイトの完全変更
        reviseAllNetworkWeights();
        // 入力側のウェイトと出力側のウェイトの整合性を図る
        for (int i = topology.length - 1; i > 0; i--) {
            adjustingPreLayerWeightsOut(i);
        }
    }

    void printForwardOutput() {
        // 順伝搬出力の表示
        // 最終レイヤーを取り出す
        Layer layer = layers[layers.length - 1];
        layer.printOutput();
    }

    double [] getOutput() {
        // 最終レイヤーを取り出す
        Layer layer = layers[layers.length - 1];
        return layer.layerOutput;
    }

    void getForwardOutput(double [] initVal) {
        if (DEBUG) {
            System.out.println("getForwardOutput Network Topology Size = " + topology.length);
        }
        // 入力を直接受け取るレイヤーの出力値を計算する
        Layer layer0 = layers[0];
        if (DEBUG) {
            System.out.println("getForwardOutput 入力層の出力は、入力データそのまま");
        }
        for (int i = 0; i < layer0.neurons.length; i++) {
            Neuron neuron = layer0.neurons[i];
            layer0.layerOutput[i] = neuron.output = initVal[i];
        }
        if (DEBUG) {
            JDeepLearning.showVectorVals("getForwardOutput 入力層からの出力", layer0.layerOutput);
        }
        for (int i = 1; i < topology.length; i++) {
            Layer layer = layers[i];
            Layer prev = layers[i - 1];
            if (DEBUG) {
                JDeepLearning.showVectorVals("getForwardOutput 一つ前の層からの出力", prev.layerOutput);
            }
            if (DEBUG) {
                System.out.println("getForwardOutput 層 [ " + i + " ] の出力計算");
            }
            layer.getLayerOutput(prev.layerOutput);
        }
    }

}
