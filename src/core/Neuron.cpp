#include "core/Neuron.hpp"
#include <iostream>
#include <math.h>

Neuron::Neuron(double bias,const std::vector<double>& weights): bias(bias), weights(weights){}

double Neuron::GetLinearComb(const std::vector<double> &input)
{
    double z = bias;

    for(size_t i = 0; i < input.size(); i++)
    {
        z += input[i] * weights[i];
    }
    return z;
}
double exp_taylor(double x, int terms = 50)
{
    double result = 1.0;
    double term = 1.0;

    for (int n = 1; n <= terms; ++n)
    {
        term *= x / n;
        result += term;
    }

    return result;
}

double Neuron::ActivationFunc(double x)
{
    double exp_approx = exp_taylor(-x);
    return 1.0 / (1.0 + exp_approx);
}

double Neuron::Compute(const std::vector<double> &input)
{
    double interResult = GetLinearComb(input);
    this->output = ActivationFunc(interResult);
    return this->output;
}

double log_approx(double x)
{
    if (x <= 0.0)
        throw std::invalid_argument("Log undefined for x <= 0");

    double y = (x - 1) / (x + 1);
    double y2 = y * y;

    double result = 0.0;
    double term = y;
    int maxIter = 10;

    for (int n = 1; n <= maxIter; n += 2)
    {
        result += term / n;
        term *= y2;
    }

    return 2.0 * result;
}

double ComputeLoss(const std::vector<double> &y_true, const std::vector<double> &y_pred)
{
    double sum = 0.0;
    double epsilon = 1e-7;

    for (size_t i = 0; i < y_true.size(); ++i)
    {
        double y_hat = std::max(epsilon, std::min(1.0 - epsilon, y_pred[i]));
        sum += y_true[i] * log_approx(y_hat) + (1.0 - y_true[i]) * log_approx(1.0 - y_hat);
    }

    return -sum / y_true.size();
}

double Neuron::GetOutput()
{
    return this->output;
}

void Neuron::UpdateWeightsCached(double prediction, const std::vector<double> &inputs, double target, double learningRate)
{
    double error = prediction - target;

    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] -= learningRate * error * inputs[i];
    }
    bias -= learningRate * error;
}
