/*
 * (C) 2017 Toyoaki WASHIDA
 */
package jdeeplearning;

import java.util.ArrayList;

/**
 *
 * @author Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 */
public class JDeepLearning {
    static double[][][] reserved_weights;
    Data dataProc = new Data();
    String weightFile = "";
    String dataFileName = "trainingData.txt";
    int iteration = 0;
    int maxIteration = -1;
    boolean testPhase = false;
    boolean autoencoder = false;
    String autoDataName;
    // autoencoderのばあい、
    // 毎回の出力ウェイトファイル名を保存しておく
    ArrayList<String> weghtDataSet = new ArrayList<>();

    static boolean DEBUG = false;
    
    static void showVectorVals(String label, double[] v) {
        System.out.print(label + " ");
        for (int i = 0; i < v.length; ++i) {
            System.out.print(i + ":" + v[i] + " ");
        }
        System.out.println();
    }

    void execAutoencoder(int[] topology) {
        if (topology.length < 4) {
            System.out.println("エラー: Autoencoderを行うためには、隠れ層は2つ以上なければなりません");
            return;
        }
        for (int n = 0; n < topology.length - 2; n++) {
            // 入力層と出力層を除く隠れ層について、自己符号化手続きを実施する
            int [] topologyAuto = {topology[n], topology[n + 1], topology[n]};
            System.out.println("Autoencoder Phase[ "+ n + " ] 0:"
                    +topology[n] + " 1:" + topology[n + 1] + " 2:" + topology[n]);
            // n=0の時は、元のデータがセット済みなので
            if(n > 0){
                dataProc.fileClose();
                dataProc.setTrainData(autoDataName);
            }
            execNeuralnet(topologyAuto);
        }
        // ファインチューニングの作業に入る
        // これまで作成したウェイトと、最終層へのウェイトは乱数で作成し
        // 全体の完全なウェイトを作っておけば、自動的に、それを読み込んで
        // ファインチューニングは実行される
        System.out.println("ファインチューニングの作業に入る");
        int maxNeuron = 0;
        for (int i = 0; i < topology.length; i++) {
            if (topology[i] > maxNeuron) {
                maxNeuron = topology[i];
            }
        }
        // ニューロン数とウェイト数は、最大ニューロン数にしておく、若干無駄があるが
        reserved_weights = new double[topology.length - 1][maxNeuron][maxNeuron];
        for(int i=0;i<weghtDataSet.size();i++){
            String weightFileName = weghtDataSet.get(i);
            System.out.println("ウェイトの形成: File = "+weightFileName);
            dataProc.setAutoencoderFinalWeight(weightFileName, i);
            
        }
        
        // 最終レイヤーのウェイトだけは、確率的に与える
        for(int i=0;i<topology[topology.length-1];i++){ // 最終層ニューロン
            System.out.println("最終レイヤーのウェイトだけは、確率的に与える");
            for (int j = 0; j < topology[topology.length - 2]; j++) { // 最終より一つ前の層ニューロン数＝ウェイト数
                reserved_weights[topology.length-2][i][j] = Math.random() / (double) topology[topology.length - 2];
            }
        }
        // ファインチューニングは、autoencoder = falseにする
        autoencoder = false;
        // データを最初から読み直す
        dataProc.fileClose();
        dataProc.setTrainData(dataFileName);
        execNeuralnet(topology);
        System.out.println("Autoencoderアルゴリズムによる学習を終えました");
        
    }

    void execNeuralnet(int[] topology) {
        // autoEncoderをしない場合は、直接このメソッドに来れば良い
        // autoEncoderは、出力層のデータの使い方を変えるために使う
        // ネットワーク構造は、topologyで、どちらでも良いように
        ///////////////////////<AuttoEncoder>///////////////////////////////
        if(autoencoder){
            if(topology.length != 3){
                System.out.println("Autoencoderの場合は、3層にしてください");
                return;
            }
            if(topology[0] != topology[2]){
                System.out.println("Autoencoderの場合は、入力層と出力層のニューロン数は同じにしてください");
                return;
            }
        }
        ///////////////////////<AuttoEncoder>///////////////////////////////
        // ここでニューラルネットを作ります
        System.out.println("ここでニューラルネットを作ります");
        Net net = new Net(topology);

        double[] inputVals;
        double[] targetOutputVals;
        int errorNum = 0;
        int correctNum = 0;
        int[] correctHist = new int[1000]; // 正解のヒストグラム
        for (int i = 0; i < correctHist.length; i++) {
            correctHist[i] = 0;
        }
        while (dataProc.isReady()) {
            if (iteration % 100 == 0) {
                System.out.println("\n--No." + iteration + "--");
            }
            // 最大繰り返し数が指定されていたらそれで中止する
            if (maxIteration >= 0 && iteration >= maxIteration) {
                break;
            }
            inputVals = dataProc.getNextInputs();
            //System.out.println("データ [ " + num + " ] 個、読み込みました");
            if(inputVals == null){
                System.out.println("読み込む入力データがもうありません");
                break;
            }
            if (inputVals.length == topology[0]) {
                // データが入力ニューロン数と一致している場合に限る
                if (DEBUG) {
                    if (iteration % 100 == 0) {
                        showVectorVals("入力データ", inputVals);
                    }
                }
                net.getForwardOutput(inputVals);
                if(!autoencoder){
                    if (iteration % 100 == 0) {
                        net.printForwardOutput();
                    }
                }
            }
            ///////////////////////<AuttoEncoder>///////////////////////////////
            if(autoencoder){
                // オートエンコーダーの場合は、教師データを入力データにする
                targetOutputVals = inputVals;
            }else{
                targetOutputVals = dataProc.getTargetOutputs();
                if(targetOutputVals == null){
                    System.out.println("読み込む教師データがもうありません");
                    break;
                }
            }
            ///////////////////////<AuttoEncoder>///////////////////////////////
            if (targetOutputVals.length == topology[topology.length - 1]) {
                // データが出力ニューロン数と一致している場合に限る
                if(!autoencoder){
                    if (iteration % 100 == 0) {
                        showVectorVals("main: target ", targetOutputVals);
                    }
                }
                if (testPhase) {
                    double[] output = net.getOutput();
                    double recogval = -100.0;
                    int recognum = 0;
                    for (int i = 0; i < output.length; i++) {
                        if (output[i] > recogval) {
                            recognum = i;
                            recogval = output[i];
                        }
                    }
                    if (targetOutputVals[recognum] > 0.5) {
                        // 1か0なので、0.5で分ければ良い
                        // 最大値が、答えに一致していれば正解とする
                        correctNum++;
                        // 0から1の間を1000刻んで、ヒストグラムを得る
                        for (int i = correctHist.length - 1; i >= 0; i--) {
                            if (recogval > (double) i * 0.001) {
                                int res = correctHist[i];
                                res++;
                                correctHist[i] = res;
                                break;
                            }
                        }
                    } else {
                        errorNum++;
                    }
                } else {
                    // バックプロパゲーションの実行
                    // targetOutputValsをセミバッチにすべきか
                    double[] deltaE = new double[topology[topology.length - 1]];
                    double loss = 0.0;
                    for (int i = 0; i < topology[topology.length - 1]; i++) {
                        Layer layer = net.getOutputLayer();
                        deltaE[i] = layer.layerOutput[i] - targetOutputVals[i];
                        loss += (layer.layerOutput[i] - targetOutputVals[i]) * (layer.layerOutput[i] - targetOutputVals[i]);
                    }
                    loss *= 0.5;
                    if (iteration % 100 == 0) {
                        System.out.println("誤差: " + loss);
                    }
                    if (DEBUG) {
                        showVectorVals("損失関数微分値 deltaE ", deltaE);
                    }
                    net.execBackpropagation(deltaE);
                }
            }
            iteration++;
        }
        dataProc.fileClose();
        if (testPhase) {
            System.out.println("****** テストの結果 *********");
            double rate = (double) correctNum / ((double) correctNum + (double) errorNum);
            System.out.println("正解数 = " + correctNum + " 不正解数 = " + errorNum + " 正解率 = " + rate);
            System.out.println("ヒストグラム（正解の出力値）");
            for (int i = correctHist.length - 1; i >= 0; i--) {
                double fromRange = (double) (i + 1) * 0.001;
                double toRange = (double) i * 0.001;
                System.out.println(fromRange + " - " + toRange + " : " + correctHist[i]);
            }
        } else {
            // トレーニングを実行した場合は、最終ウェイトをファイルに出力する
            String weightFileName = net.printAllWeights(iteration, dataFileName);
            ///////////////////////<AuttoEncoder>///////////////////////////////
            if(autoencoder){
                // 最終ファインチューニング用に全ての出力ウェイトファイルを保存しておく
                this.weghtDataSet.add(weightFileName);
                // 隠れ層の出力データが次の入力になるが、そのファイル名を保存しておく
                autoDataName = net.makeAutoencoderData(dataProc);
            }
            ///////////////////////<AuttoEncoder>///////////////////////////////
        }
        //
        System.out.println("\nNeural Network Done");
        //        
    }

    /**
     * @param argv the command line arguments
     */
    public static void main(String[] argv) {
        // main.cppは、基本、オプションの処理
        JDeepLearning dpl = new JDeepLearning();
        for (int i = 0; i < argv.length; i++) {
            if (argv[i].equals("-weights")) {
                if (i >= argv.length) {
                    System.out.println("コマンドの引数が不足しています [ " + argv[i] + "]");
                    return;
                }
                String path = argv[++i];
                System.out.println("weightsファイルを読み込みます file = " + path);
                dpl.iteration = dpl.dataProc.openWeightFile(path);
                if (dpl.iteration < 0) {
                    System.out.println("weightsファイルを読み込みに失敗しました");
                    return;
                }                
            } else if (argv[i].equals("-data")) {
                if (i >= argv.length) {
                    System.out.println("コマンドの引数が不足しています [ " + argv[i] + "]");
                    return;
                }
                dpl.dataFileName = argv[++i];
                System.out.println("dataFileNameを変更します 新しいfile = " + dpl.dataFileName);
            } else if (argv[i].equals("-test")) {
                System.out.println("バックプロパゲーションを行わずテストを実行します ");
                dpl.testPhase = true;
            } else if (argv[i].equals("-auto")) {
                System.out.println("自動符号化で深層学習します ");
                dpl.autoencoder = true;
            } else if (argv[i].equals("-debug")) {
                System.out.println("デバグモードで実行します ");
                DEBUG = true;
            } else if (argv[i].equals("-maxiter")) {
                if (i >= argv.length) {
                    System.out.println("コマンドの引数が不足しています [ " + argv[i] + "]");
                    return;
                }
                dpl.maxIteration = Integer.valueOf(argv[++i]);
                System.out.println("最大繰り返し数を [ " + dpl.maxIteration + " ] に変更します");
            } else if (argv[i].equals("-help")) {
                System.out.println("Effective option: -weights -data -test -maxiter -auto -debug -help");
                return;
            }
        }
        dpl.dataProc.setTrainData(dpl.dataFileName);
        int[] topology = dpl.dataProc.getTopology();
        if (topology.length == 0) {
            System.out.println("トポロジーの読み込みに失敗しました 強制終了");
            return;
        }
        System.out.println("ネットワークトポロジー");
        for (int i = 0; i < topology.length; i++) {
            System.out.println("Layer [ " + i + " ] Neuron num = " + topology[i]);
        }
        if (dpl.autoencoder) {
            dpl.execAutoencoder(topology);
        } else {
            dpl.execNeuralnet(topology);
        }
    }
}
