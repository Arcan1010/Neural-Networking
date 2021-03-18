#include "NeuronNetwork.h"
#include <iostream>
#include <algorithm>

NeuronNetwork::NeuronNetwork(int numberOfInput, int numberOfOutput, int numberOfHidden, bool cBias)
{
    isBias = cBias;
    // Tworzenie wejścia
    for(int i=0; i<numberOfInput; i++)
    {
        shared_ptr<Neuron> newNeuron = make_shared<Neuron>(0);
        input.push_back(newNeuron);
    }

    // Tworzenie warstwy ukrytej
    for(int i=0; i<numberOfHidden; i++)
    {
        shared_ptr<Neuron> newNeuron = make_shared<Neuron>(input, isBias);
        hiddenNeurons.push_back(newNeuron);
    }

    // Tworzenie wyjścia
    for(int i=0; i<numberOfOutput; i++)
    {
        shared_ptr<Neuron> newNeuron = make_shared<Neuron>(hiddenNeurons, isBias);
        output.push_back(newNeuron);
    }
}

NeuronNetwork::~NeuronNetwork()
{
    hiddenNeurons.clear();
    input.clear();
    output.clear();
}

void NeuronNetwork::compute(vector<double> inputToCompute)
{
    cout << "*** Network is computing..." <<endl;
    for(int i=0; i<input.size(); i++)
    {
        input[i]->setValue(inputToCompute[i]);
    }

    for(int i=0; i<input.size(); i++)
    {
        cout << "Input[" << i << "]: " << this->input[i]->getValue() << endl;
    }

    for(int i=0; i<output.size(); i++)
    {
        cout << "Output[" << i << "]: (" << int(2 * output[i]->getValue()) << ") " << output[i]->getValue() << endl;
    }
}

void NeuronNetwork::learn(double learningRate, int numberOfEras, vector<vector<double>> trainingInput, vector<vector<double>> trainingOutput)
{
    cout << "*** Network is learning..." <<endl;
    for(int i=0; i<numberOfEras; i++)
    {
        cout << "Era nr." << i <<endl;

        vector<int> shuffleVector;
        for(int j=0; j<trainingInput.size(); j++)
        {
            shuffleVector.push_back(j);
        }
        random_shuffle(shuffleVector.begin(), shuffleVector.end());

        cout << "Shuffle order: ";
        for(int j=0; j<trainingInput.size(); j++)
        {
            cout << shuffleVector[j] << " ";
        }
        cout <<endl;

        for(int j=0; j<trainingInput.size(); j++)
        {
            cout << "Iteration nr." << j << endl;

            // Na poczatku przetwarzamy
            compute(trainingInput[shuffleVector[j]]);

            // Podajemy sredni blad
            double sum = 0;
            for(int i=0; i<output.size(); i++)
            {
                sum = sum + (output[i]->getValue() - trainingOutput[shuffleVector[j]][i]) * (output[i]->getValue() - trainingOutput[shuffleVector[j]][i]);
            }
            cout << "Average cost: " << sum << endl;

            // Potem uzupełniamy bledy na wyjsciu
            for(int k=0; k<output.size(); k++)
            {
                output[k]->giveExpected(trainingOutput[shuffleVector[j]][k]);
            }

            // Potem rekurencyjnie uzupelniamy bledy
            for(int k=0; k<input.size(); k++)
            {
                input[k]->updateError();
            }

            // Potem rekurencyjnie uzupelniamy bledy biasów jeśli są biasy
            if(isBias)
            {
                for(int k=0; k<output.size(); k++)
                {
                    output[k]->getBias()->updateError();
                }
                for(int k=0; k<hiddenNeurons.size(); k++)
                {
                    hiddenNeurons[k]->getBias()->updateError();
                }
            }

            // Potem rekurencyjnie poprawiamy wagi
            for(int k=0; k<output.size(); k++)
            {
                output[k]->updateWeights(learningRate);
            }
        }
    }
}