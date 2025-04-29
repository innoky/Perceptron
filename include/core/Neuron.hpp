#include <iostream>
#include <vector>


class Neuron
{
public:
    Neuron(double bias, const std::vector<double>& weights);

    double GetOutput();
    double GetLinearComb(const std::vector<double> &input);
    double ActivationFunc(double input);
    double Compute(const std::vector<double> &input);
    void UpdateWeights(const std::vector<double> &inputs, double target, double learningRate);
    void UpdateWeightsCached(double prediction, const std::vector<double> &inputs, double target, double learningRate);

    std::vector<double> predictions;

private:

    double output;
    std::vector<double> weights;
    double bias;
};

double ComputeLoss(const std::vector<double> &y_true, const std::vector<double> &y_pred);