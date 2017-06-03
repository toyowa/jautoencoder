/* 
 * File:   Dataset.h
 * Author: Toyoaki WASHIDA <Toyoaki WASHIDA at ibot.co.jp>
 *
 * Created on 2017/05/31, 5:15
 */

#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Mnist {
public:
    vector<vector<double> > readTrainingFile(string filename);
    vector<double> readLabelFile(string filename);
};

#endif /* DATASET_H */

