#include "include/core/Neuron.hpp"
#include <math.h>

Neuron::Neuron(double bias, std::vector<double> weights): bias(bias), weights(weights){}

double Neuron::GetLinnearComb()
{
    double z = bias;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        z += inputs[i] * weights[i];
    }
    return z;
}

double Neuron::ActivationFunc(double input)
{
    return (  (1/2) + (input/4) + (pow(input, 3)/48) + (pow(input, 5)/480) - ((17*pow(input, 7))/80640) );
}
