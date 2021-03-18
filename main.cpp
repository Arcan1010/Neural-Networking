#include <iostream>
#include "Neuron.h"
#include "NeuronNetwork.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
using namespace std;

vector<vector<double>> initializeTrainingData(string filePath, int numberOfExamples, int numberOfData);
void transformationNetwork(int numberOfHidden, bool isBias);
void xorNetwork(int numberOfHidden, bool isBias);

int main()
{
    srand(time(0));

    cout << "*** Neural Network" <<endl;

    int quest = 0;
    do
    {
        cout << "*** What type of network ? (1/2)" <<endl;
        cout << "*** 1) 4 input > 4 output (training with transformation data)" <<endl;
        cout << "*** 2) 2 input > 1 output (learning XOR)" <<endl;
        cin >> quest;
    }
    while(quest != 1 && quest != 2);

    char bias = ' ';
    do
    {
        cout << "*** Network with bias? (y/n)" <<endl;
        cin >> bias;
    }
    while(bias != 'y' && bias != 'Y' && bias != 'n' && bias != 'N');

    int hidden = 0;
    do
    {
        cout << "*** Give number of Neurons in hidden layer:" << endl;
        cin >> hidden;
    }
    while(hidden ==0);

    if(quest == 1) transformationNetwork(hidden, bias == 'y');
    else xorNetwork(hidden, bias == 'y');

    cout << "*** End of program" <<endl;
    return 0;
}

void transformationNetwork(int numberOfHidden, bool isBias)
{
    vector<vector<double>> trainingInput1 = initializeTrainingData("D:\\CLion_projects\\Neuron-Networking\\transformation.txt",4,4);
    vector<vector<double>> trainingOutput1 = initializeTrainingData("D:\\CLion_projects\\Neuron-Networking\\transformation.txt",4,4);

    NeuronNetwork network1(4, 4, numberOfHidden, isBias);
    network1.learn(0.2, 1000, trainingInput1, trainingOutput1);

    cout << "\n\n Test Transformation: " <<endl;
    network1.compute(trainingInput1[0]);
    network1.compute(trainingInput1[1]);
    network1.compute(trainingInput1[2]);
    network1.compute(trainingInput1[3]);
    cout << "*** End of test" <<endl;
}

void xorNetwork(int numberOfHidden, bool isBias)
{
    vector<vector<double>> trainingInput2 = initializeTrainingData("D:\\CLion_projects\\Neuron-Networking\\xorInput.txt",4,2);
    vector<vector<double>> trainingOutput2 = initializeTrainingData("D:\\CLion_projects\\Neuron-Networking\\xorOutput.txt",4,1);

    NeuronNetwork network2(2, 1, numberOfHidden, isBias);
    network2.learn(0.35, 1000, trainingInput2, trainingOutput2);

    cout << "\n\n Test Xor: " <<endl;
    network2.compute(trainingInput2[0]);
    network2.compute(trainingInput2[1]);
    network2.compute(trainingInput2[2]);
    network2.compute(trainingInput2[3]);
    cout << "*** End of test" <<endl;
}

vector<vector<double>> initializeTrainingData(string filePath, int numberOfExamples, int numberOfData)
{
    ifstream file;
    file.open(filePath);
    if(file.good()) cout << "*** File opened properly" << endl;
    else cout << "*** File not opened" <<endl;

    vector<vector<double>> result;
    for(int j=0; j<numberOfExamples; j++) // !file.eof())
    {
        vector<double> part;
        for(int i=0; i<numberOfData; i++)
        {
            double x;
            file >> x;
            part.push_back(x);
        }
        result.push_back(part);
    }
    file.close();
    return result;
}