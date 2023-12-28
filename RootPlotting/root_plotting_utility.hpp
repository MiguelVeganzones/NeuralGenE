#ifndef ROOT_PLOTTING_UTILITY
#define ROOT_PLOTTING_UTILITY

#include "TApplication.h"
#include "TCanvas.h"
#include "TGraph.h"
#include <memory>

namespace root_plotting
{

auto plot(
    int                n,
    const float* const x,
    const float* const y1,
    const float* const y2
) -> void;


auto plot(int n, const float* const x, const float* const y) -> void;

class progress_plot
{
public:
    progress_plot(
        int                size,
        const float* const x,
        const float* const y1,
        const float* const y2
    ) :
        n(size)
    {
        plot(x, y1, y2);
    }

    progress_plot(int size) :
        n(size)
    {
    }

    auto plot(
        const float* const x,
        const float* const y1,
        const float* const y2
    ) -> void
    {
        canvas = new TCanvas("c1");
        gr1    = new TGraph(n, x, y1);
        gr2    = new TGraph(n, x, y2);
        // canvas = std::make_unique<TCanvas>("c1");
        // gr1    = std::make_unique<TGraph>(n, x, y1);
        // gr2    = std::make_unique<TGraph>(n, x, y2);
        canvas->Show();
        gr1->Draw("AC");
        gr2->Draw("CP");
    }

    auto update_plot2(const float* const y) -> void
    {
        for (int i = 0; i != n; ++i)
        {
            gr2->SetPointY(i, y[i]);
        }
        // canvas->Show();
        canvas->Update();
    }

    ~progress_plot()
    {
        app.Run();
    }

private:
    int          n;
    TCanvas*     canvas{};
    TGraph*      gr1{};
    TGraph*      gr2{};
    TApplication app = TApplication("Root app", 0, nullptr);
    // std::unique_ptr<TCanvas> canvas{};
    // std::unique_ptr<TGraph>  gr1{};
    // std::unique_ptr<TGraph>  gr2{};
};

} // namespace root_plotting

#endif // ROOT_PLOTTING_UTILITY