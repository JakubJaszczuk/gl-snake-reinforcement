#include "random.h"

std::mt19937 random::random_generator{std::random_device()()};
