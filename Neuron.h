#ifndef NEURON_NETWORKING_NEURON_H
#define NEURON_NETWORKING_NEURON_H

#include <memory>
#include <vector>
#include <string>

using namespace std;

class Neuron
{
    double value;
    double error;
    vector<shared_ptr<Neuron>> parentNeurons;
    vector<shared_ptr<Neuron>> childNeurons;
    vector<double> factor;

    double activation(double x);
    double dActivation(double x);

public:

    Neuron(double cValue);

    Neuron(vector<shared_ptr<Neuron>> cParentNeurons, bool isBias);

    double getValue();

    void setValue(double newValue);

    void giveExpected(double expected);

    void addChildNeuron(shared_ptr<Neuron> neuron);

    double updateError();

    void updateWeights(double learningRate);

    shared_ptr<Neuron> getBias();
};

#endif //NEURON_NETWORKING_NEURON_H
