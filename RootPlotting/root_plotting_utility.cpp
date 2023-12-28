#ifndef ROOT_PLOTTING_UTILITY
#define ROOT_PLOTTING_UTILITY

#include "root_plotting_utility.hpp"
#include "TGraph.h"

namespace root_plotting
{

auto plot(
    int                n,
    const float* const x,
    const float* const y1,
    const float* const y2
) -> void
{
    auto gr1 = new TGraph(n, x, y1);
    auto gr2 = new TGraph(n, x, y2);
    gr1->Draw("AC*");
    gr2->Draw("CP");
}

auto plot(int n, const float* const x, const float* const y) -> void
{
    auto gr1 = new TGraph(n, x, y);
    gr1->Draw("AC*");
}

// progress_plot::progress_plot(std::size_t size) :
//     n(size)
// {
// }

// auto progress_plot::plot(
//     const float* const x,
//     const float* const y1,
//     const float* const y2
// ) -> void
// {
//     gr1 = new TGraph(n, x, y1);
//     gr2 = new TGraph(n, x, y2);
//     gr1->Draw("AC*");
//     gr2->Draw("CP");
// }

// auto progress_plot::update_plot2(const float* const y) -> void
// {
//     for (std::size_t i = 0; i != n; ++i)
//     {
//         gr2->SetPointY(i, y[i]);
//     }
//     gr2->Draw();
// }


} // namespace root_plotting

#endif // ROOT_PLOTTING_UTILITY