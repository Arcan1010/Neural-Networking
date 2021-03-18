#include "Neuron.h"
#include <cstdlib>
#include <iostream>

using namespace std;

double Neuron::activation(double x)
{
    return (1 / (1 + exp((-1) * x)));
}

double Neuron::dActivation(double x)
{
    return (x * (1.0 - x));
}

Neuron::Neuron(double cValue)
: value(cValue)
{
}

Neuron::Neuron(vector<shared_ptr<Neuron>> cParentNeurons, bool isBias)
: parentNeurons(cParentNeurons)
{
    for(int i=0; i<cParentNeurons.size(); i++)
    {
        parentNeurons[i]->addChildNeuron(shared_ptr<Neuron>(this));
        factor.push_back( (2 * ((double)rand()/(double)RAND_MAX)) - 1 );
    }
    if(isBias)
    {
        parentNeurons.push_back(make_shared<Neuron>(1));
        factor.push_back( (2 * ((double)rand()/(double)RAND_MAX)) - 1 );
    }

}

void Neuron::setValue(double newValue)
{
    value = newValue;
}

double Neuron::getValue()
{
    if(parentNeurons.empty() || &parentNeurons == NULL) return value;
    else
    {
        double result = 0;
        for (int i = 0; i < factor.size(); i++)
        {
            result = result + parentNeurons[i]->getValue() * factor[i];
        }
        value = activation(result);
        return value;
    }
}

void Neuron::addChildNeuron(shared_ptr<Neuron> neuron)
{
    childNeurons.push_back(neuron);
}

double Neuron::updateError()
{
    if(childNeurons.empty()) //wyjście
    {
        return error;
    }
    else
    {
        error = 0;
        for(int i=0; i<childNeurons.size(); i++)
        {
            int j=0;
            while(childNeurons[i]->parentNeurons[j].get() != this)
            {
                j++;
            }
            error = error + childNeurons[i]->updateError() * childNeurons[i]->factor[j];
        }
        return error;
    }
}

void Neuron::updateWeights(double learningRate)
{
    if(!parentNeurons.empty() && &parentNeurons != NULL)
    {
        for(int i=0; i<parentNeurons.size(); i++)
        {
            factor[i] = factor[i] + (learningRate * error * parentNeurons[i]->value * dActivation(value) );
            parentNeurons[i]->updateWeights(learningRate);
        }
    }
}

void Neuron::giveExpected(double expected)
{
    error = expected - value;
}

shared_ptr<Neuron> Neuron::getBias()
{
    return parentNeurons[parentNeurons.size()-1];
}
