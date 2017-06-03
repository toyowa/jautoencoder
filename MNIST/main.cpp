/* 
 * File:   main.cpp
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 *
 * Created on 2017/05/31, 5:04
 */

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "Dataset.h"
#include <vector>
#include <string>    // useful for reading and writing
#include <fstream>   // ifstream, ofstream
#include <iomanip>

int main (int argc, char** argv) {
    bool testPhase = false;
    string conversion = "orig";
    for(int i=1;i<argc;i++) {
        if (strcmp(argv[i], "-convert") == 0) {
            if(i+1 >= argc){
                cout << "コマンドの引数が不足しています [ " << argv[i] << "]" << endl;
                return 1;
            }
            string convert = argv[++i];
            if(convert == "bit"){
                cout << "データを0,1化します" << endl;
                conversion = "bit";
            }else if(convert == "norm"){
                cout << "データを0から1の間の数に正規化します" << endl;
                conversion = "norm";
            }else{
                cout << "-convert の引数が不正です = " << convert << endl;
                return 0;
            }
        }else if(strcmp(argv[i], "-test") == 0){
            cout << "テスト段階のデータを作成します" << endl;
            testPhase = true;
        }else if(strcmp(argv[i], "-help") == 0){
            cout << "有効なオブション: -convert bit|norm" << endl;
        }
    }
    //絶対パスを入力してください。
    Mnist mnist;
    vector<vector<double> > imgdata;
    vector<double> imglabel;
    if(testPhase){
        imgdata = mnist.readTrainingFile("/absolutePathto/t10k-images-idx3-ubyte");
        imglabel = mnist.readLabelFile("/absolutePathto/t10k-labels-idx1-ubyte");
    }else{
        imgdata = mnist.readTrainingFile("/absolutePathto/train-images-idx3-ubyte");
        imglabel = mnist.readLabelFile("/absolutePathto/train-labels-idx1-ubyte");
    }

    time_t timer;
    struct tm* tm;
    char datetime[20];
    timer = time(NULL);
    tm = localtime(&timer);
    strftime(datetime, 20, "%Y%m%d%H%M%S", tm);
    //cout << "DATETIME = " << datetime << endl;
    string dt = datetime;
    string filename;
    int datanum;
    if(testPhase){
        filename = "mnist_test_"+conversion+"_"+dt+"."+"txt";
        datanum = 10000;
    }else{
        filename = "mnist_train_"+conversion+"_"+dt+"."+"txt";
        datanum = 60000;
    }
    ofstream writing_file;
    writing_file.open(filename, ios::out);
    cout << "writing " << filename << "..." << endl;
    writing_file << "topology: 784 100 10" << endl;

    for (int n = 0; n < datanum; n++) {
        vector<double> data = imgdata.at(n);
        //cout << label << endl;
        writing_file << "in: ";
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                //cout << hex << (unsigned int) data.at(i * i + j);
                unsigned int igd = (unsigned int) data.at(i * 28 + j);
                if(conversion == "bit"){
                    if(igd > 0) writing_file << "1.0 ";
                    else writing_file << "0.0 ";
                }else if(conversion == "norm"){
                    writing_file << fixed << setprecision(3) << (double)igd/(double)0xff << " ";
                }else{
                    writing_file << igd << " ";
                }
            }
        }
        writing_file << endl;
        unsigned int label = (unsigned int)imglabel.at(n);
        writing_file << "out: ";
        for(unsigned int k=0;k<10;k++){
            if(label == k) writing_file << "1.0 ";
            else writing_file << "0.0 ";
        }
        writing_file << endl;
    }
    cout << "書き出しを終了しました " <<  endl;
}
