/*
 * (C) 2017 Toyoaki WASHIDA
 */
package jdeeplearning;

import java.io.FileWriter;
import static jdeeplearning.JDeepLearning.DEBUG;

/**
 *
 * @author Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 */
public class Layer {

    Neuron[] neurons;
    double[] layerOutput;
    double[] layerDelta;
    double adjusting;
    int layerNo;

    Layer(int[] topology, int layerNo, double adjusting) {
        this.adjusting = adjusting;
        this.layerNo = layerNo;
        neurons = new Neuron[topology[layerNo]];
        layerOutput = new double[neurons.length];
        layerDelta = new double[neurons.length];
        // layerNo は層の番号
        for (int j = 0; j < topology[layerNo]; j++) {
            if (layerNo == 0) {
                // 入力レイヤーの場合、前のレイヤーがない
                Neuron neuron = new Neuron(0, topology[1], j, layerNo, adjusting);
                neuron.inputNeuron = true;
                neurons[j] = neuron;
            } else {
                // 出力層の場合
                if (layerNo == topology.length - 1) {
                    Neuron neuron = new Neuron(topology[layerNo - 1], 0, j, layerNo, adjusting);
                    neuron.outputNeuron = true;
                    neurons[j] = neuron;
                } else {
                    Neuron neuron = new Neuron(topology[layerNo - 1], topology[layerNo + 1], j, layerNo, adjusting);
                    neurons[j] = neuron;
                }
            }
        }
    }

    void printAllWeights(FileWriter writing_file) {
        for (Neuron neuron : neurons) {
            neuron.printWeightsIn(writing_file);
        }
    }

    void reviseNetworkWeights() {
        for (int i = 0; i < neurons.length; i++) {
            Neuron neuron = neurons[i];
            neuron.adjustingWeightsIn();
        }
    }

    void getLayerDelta(double [] deltas) {
        if (DEBUG) {
            System.out.println("getLayerDelta この層のニューロン数 = " + neurons.length);
        }
        if (DEBUG) {
            System.out.println("getLayerDelta 前の層のDelta値のサイズ = " + deltas.length);
        }
        for (int j = 0; j < neurons.length; j++) {
            Neuron neuron = neurons[j];
            neuron.getNeuronDelta(deltas);
            layerDelta[j] = neuron.delta;
            // javaの場合、配列への戻しは不要だ、いわゆる参照状態になっている
            // c++ vectorは、コピーを渡しているようだ
            //neurons[j] = neuron;
        }
    }

    void getLayerOutput(double [] outputs) {
        if (DEBUG) {
            System.out.println("getLayerOutput この層のニューロン数 = " + neurons.length);
        }
        if (DEBUG) {
            System.out.println("getLayerOutput 前の層の出力サイズ = " + outputs.length);
        }
        for (int j = 0; j < neurons.length; j++) {
            Neuron neuron = neurons[j];
            neuron.getNeuronOutput(outputs);
            layerOutput[j] = neuron.output;
            // javaの場合、戻しは必要ない
        }
    }

    void printOutput() {
        if (layerOutput.length != neurons.length) {
            System.out.println("printOutput Error: I have no output data! layerOutput.size() = " 
                    + layerOutput.length + " neurons.size() = " + neurons.length);
            return;
        }
        System.out.println("printOutput ");
        for (int i = 0; i < neurons.length; i++) {
            System.out.println(i + ":" + layerOutput[i] + " ");
        }
    }
 
}
