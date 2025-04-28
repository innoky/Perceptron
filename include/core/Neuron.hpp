#include <iostream>
#include <vector>


class Neuron
{
public:
    Neuron(double bias, std::vector<double> weights);
    std::vector<double> weights;
    std::vector<double> inputs;
    double bias;


    double GetOutput();
    double GetLinnearComb();
    double ActivationFunc(double input);
    double Compute();

private:

    double output;
};