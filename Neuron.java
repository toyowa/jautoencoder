/*
 * (C) 2017 Toyoaki WASHIDA
 */

package jdeeplearning;

import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import static jdeeplearning.JDeepLearning.DEBUG;

/**
 *
 * @author Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 */
public class Neuron {
    int neuronNo;
    int layerNo;
    double adjusting;
    boolean inputNeuron = false;
    boolean outputNeuron = false;
    double [] weightsIn;
    double [] weightsOut;
    double [] prev_output;
    double output;
    double delta;
    double value;

    public Neuron(int inputNum, int outNum, int neuronNo, int layerNo, double adjusting){
        if (inputNum > 0) {
            weightsIn = new double[inputNum];
            if (JDeepLearning.reserved_weights != null && JDeepLearning.reserved_weights.length > 0) {
                // 保存されていたウェイトを使用する場合
                for (int i = 0; i < inputNum; i++) {
                    if (DEBUG) {
                        System.out.println("Layer[" + layerNo + "] Neuron ["
                                + neuronNo + "] のNo.[" + i + "]のweight[ "
                                + JDeepLearning.reserved_weights[layerNo - 1][neuronNo][i] + " ]をセット");
                    }
                    weightsIn[i] = JDeepLearning.reserved_weights[layerNo - 1][neuronNo][i];
                }
            } else {
                for (int i = 0; i < inputNum; i++) {
                    // ウェイトの初期値は乱数で与える
                    // スケールが大きすぎると不都合が発生するのでinputNumで割る
                    weightsIn[i] = Math.random() / (double) inputNum;
                    if (DEBUG) {
                        System.out.println("Layer[" + layerNo + "] Neuron ["
                                + neuronNo + "] の、前の層、No.[" + i + "]ニューロンからのweight[ "
                                + weightsIn[i] + " ]をセット");
                    }
                }
            }
        }
        if (outNum > 0) {
            // 出力側のウェイトの初期値はなんでも良い
            // バックプロパゲーションの際に上の層のニューロンによって書き換えられる
            weightsOut = new double[outNum];
            for (int i = 0; i < outNum; i++) {
                weightsOut[i] = 1.0;
            }
        }
        inputNeuron = false;
        outputNeuron = false;
        this.neuronNo = neuronNo;
        this.layerNo = layerNo;
        this.adjusting = adjusting;
    }
    
    void printWeightsIn(FileWriter writing_file){
        // ウェイトのファイルへの書き出し
        for (int i = 0; i < weightsIn.length; i++) {
            try {
                writing_file.append(String.format("Weight: %d -> %d : %d = %15.10f\n", i, layerNo, neuronNo, weightsIn[i]));
            } catch (IOException ex) {
                Logger.getLogger(Neuron.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    void adjustingWeightsIn() {
        for (int i = 0; i < weightsIn.length; i++) {
            if (DEBUG) {
                System.out.println("adjustingWeightsIn prev_output.at(i) = " + prev_output[i] 
                        + " delta = " + delta + " org weight = " + weightsIn[i]);
            }
            weightsIn[i] = weightsIn[i] - adjusting * delta * prev_output[i];
            if (DEBUG) {
                System.out.println("adjustingWeightsIn L:N[" + layerNo + ":" + neuronNo 
                        + "] 入力ウェイト No.[" + i + "]を調整 weight = " + weightsIn[i]);
            }
        }
    }

    void getNeuronDelta(double[] prev_delta) {
        if (prev_delta == null || prev_delta.length == 0) {
            System.out.println("getNeuronDelta エラー: ニューロンのデルタ値の計算に必要な入力データがありません");
            return;
        }
        if (outputNeuron) {
            // 出力ニューロンだった場合
            double ev = Math.exp(-value);
            delta = ev / ((1 + ev) * (1 + ev));
            delta = delta * prev_delta[neuronNo]; // 損失関数の微分値
            if (DEBUG) {
                System.out.println("getNeuronDelta 出力ニューロン のDeltaを計算 value = " + value 
                        + " prev_delta.at(neuronNo) = " + prev_delta[neuronNo]);
            }
        } else {
            delta = 0.0;
            for (int i = 0; i < prev_delta.length; i++) {
                delta += prev_delta[i] * weightsOut[i];
                if (DEBUG) {
                    System.out.println("getNeuronDelta prev_delta.at(" + i + ") = " + prev_delta[i]
                            + " weightsOut.at(" + i + ") = " + weightsOut[i]);
                }
            }
            double ev = Math.exp(-value);
            delta = delta * (ev / ((1 + ev) * (1 + ev)));
        }
        if (DEBUG) {
            System.out.println("getNeuronDelta レイヤー No.[ " + layerNo + " ]  出力ニューロン No.[ " 
                    + neuronNo + " ] の 保存されていた value = " + value + " Delta = " + delta);
        }
    }

    void getNeuronOutput(double [] prev_output) {
        this.prev_output = prev_output;
        if (prev_output == null || prev_output.length == 0) {
            System.out.println("getNeuronOutput エラー: ニューロンの出力値の計算に必要な入力データがありません");
            return;
        }
        value = 0.0;
        for (int i = 0; i < weightsIn.length; i++) {
            if (DEBUG) {
                System.out.println("getNeuronOutput L:N[" + layerNo + ":" + neuronNo + "] weightsIn.at(" 
                        + i + ") = " + weightsIn[i]
                        + " prev_output.at(" + i + ") = " + prev_output[i]);
            }
            value += prev_output[i] * weightsIn[i];
        }
        output = 1 / (1 + Math.exp(-value));
    }

}
