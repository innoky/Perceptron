#include <vector>
#include "include/core/Neuron.hpp"
#include <memory>

class NWeb{
public:
    std::vector<std::unique_ptr<Neuron>> Layer;

private:

};