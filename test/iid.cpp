#include "test.hpp"

#include "../include/iid.hpp"

int main(int argc, char **argv) {

  {
    std::vector<double> s = {-1, 0, 1, 2, 3, 4, 5};

    if (true == sp_800_90B(s)) {
      return -1;
    }
  }

  {
    std::random_device rd;
    std::mt19937 g(rd());

    double lower_bound = 0;
    double upper_bound = 10000;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);

    while (true) {

      std::vector<double> s;
      for (int i = 0; i < 10; ++i) {
        s.push_back(unif(g));
      }

      if (sp_800_90B(s)) {
        // c++ mt19937 should pass iid eventually
        break;
      }

      for (double e : s) {
        std::cerr << e << " ";
      }
      std::cerr << "\n";
    }
  }

  return 0;
}