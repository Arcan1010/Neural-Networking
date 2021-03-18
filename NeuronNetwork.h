#ifndef NEURON_NETWORKING_NEURONNETWORK_H
#define NEURON_NETWORKING_NEURONNETWORK_H

#include <memory>
#include <vector>
#include "Neuron.h"
using namespace std;

class NeuronNetwork // tylko jedna warstwa ukryta
{
    vector<shared_ptr<Neuron>> input;
    vector<shared_ptr<Neuron>> hiddenNeurons;
    vector<shared_ptr<Neuron>> output;
    bool isBias;

public:

    NeuronNetwork(int numberOfInput, int numberOfOutput, int numberOfHidden, bool cBias);

    virtual ~NeuronNetwork();

    void compute(vector<double> inputToCompute);

    void learn(double learningRate, int numberOfEras, vector<vector<double>> trainingInput, vector<vector<double>> trainingOutput);
};


#endif //NEURON_NETWORKING_NEURONNETWORK_H
