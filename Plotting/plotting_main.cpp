#include <matplot/matplot.h>

#include <cmath>
#include <vector>

#include "Random.h"
#include "plotting_utility.h"
#include "static_matrix.hpp"

void test0()
{
  using namespace matplot;
  const std::vector<double> x = linspace(0, 2 * pi);
  const std::vector<double> y = transform(x, [](auto e) { return sin(e); });

  plot(x, y, "-o");
  hold(on);
  plot(x, transform(y, [](auto e) { return -e; }), "--xr");
  plot(x, transform(x, [](auto e) { return e / pi - 1.; }), "-:gs");
  plot({ 1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1 }, "k");

  show();
  hold(off);
}

void test1()
{
  random::init();

  ga_sm::static_matrix<float, 30, 1> y{};

  y.fill(random::randfloat);

  plotting_utility::plot(y);
}

void test2()
{
  random::init();

  ga_sm::static_matrix<float, 10, 1> x{}, y1{}, y2{}, y3{}, y4{}, y5{}, y6{};

  x.iota_fill();
  y1.fill(random::randfloat);
  y2.fill(random::randfloat);
  y3.fill(random::randfloat);
  y4.fill(random::randfloat);
  y5.fill(random::randfloat);
  y6.fill(random::randfloat);

  plotting_utility::plot(x, { y1, y2, y3, y4, y5, y6 });
}

int main()
{
  // test0();
  test2();

  return EXIT_SUCCESS;
}
