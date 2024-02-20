#include "Random.hpp"
#include "neural_model.hpp"
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
    random::seed();

    ga_sm::static_matrix<float, 30, 1> y{};

    y.fill(random::randfloat);

    plotting_utility::plot(y);
}

void test2()
{
    random::seed();

    ga_sm::static_matrix<float, 10, 1> x{}, y1{}, y2{}, y3{}, y4{}, y5{}, y6{};

    x.iota_fill();
    y1.fill(random::randfloat);
    y2.fill(random::randfloat);
    y3.fill(random::randfloat);
    y4.fill(random::randfloat);
    y5.fill(random::randfloat);
    y6.fill(random::randfloat);

    plotting_utility::plot(x, std::vector{ y1, y2, y3, y4, y5, y6 });
}

void training_test()
{
    random::seed();

    constexpr std::size_t N = 20;
    constexpr std::size_t M = 1500;

    using namespace ga_snn;
    using namespace ga_sm;

    constexpr auto AF_relu =
        matrix_activation_functions::ActivationFunctionIdentifiers::GELU;
    constexpr auto AF_tanh = matrix_activation_functions::
        ActivationFunctionIdentifiers::UnsignedSigmoid;

    [[maybe_unused]] constexpr Layer_Signature a1{ 1, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a1_tanh{ 1, AF_tanh };
    [[maybe_unused]] constexpr Layer_Signature a9{ 9, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a25{ 25, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a42{ 42, AF_relu };
    [[maybe_unused]] constexpr Layer_Signature a16{ 16, AF_relu };

    using NET = static_neural_net<float, 1, a1, a9, a1_tanh>;

    using preprocessor  = data_processor::scalar_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = NET::value_type;

    using brain_t =
        ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;

    std::array<std::array<brain_t, N>, 2> gen{};

    for (size_t i = 0; i != N; ++i)
    {
        gen[0][i] = brain_t(random::randnormal, 0, 0.01);
        gen[1][i] = brain_t(random::randnormal, 0, 0.01);
    }

    std::vector<float> in(M), out(M), pred(M), best_pred(M);
    //, best_pred{};
    std::iota(in.begin(), in.end(), 0.f);
    std::ranges::transform(in, out.begin(), [](float i) {
        return std::sin(i / 100);
    });

    for (size_t iter = 0; iter != 100000; ++iter)
    {
        std::array<float, N> error{};
        for (size_t j = 0; j != N; ++j)
        {
            for (size_t i = 0; i != M; ++i)
            {
                pred[i] = gen[iter % 2][j](in[i]);
                error[j] += std::abs(pred[i] - out[i]);
            }
            if (j == 0 || (j > 0 && error[j] < error[j - 1]))
            {
                best_pred = pred;
            }
        }

        // for (const auto e :error)
        //{
        //     std::cout << e << ' ';
        // }
        // std::cout << '\n';

        auto temp = std::array(error);
        std::ranges::sort(temp);

        const auto idx0 = std::ranges::find(error, temp[0]) - error.begin();
        const auto idx1 = std::ranges::find(error, temp[1]) - error.begin();
        const auto idx2 = std::ranges::find(error, temp[2]) - error.begin();
        const auto idx3 = std::ranges::find(error, temp[3]) - error.begin();
        const auto idx4 = std::ranges::find(error, temp[4]) - error.begin();
        const auto idx5 = std::ranges::find(error, temp[5]) - error.begin();

        to_target_brain_x_crossover(
            gen[iter % 2][idx0],
            gen[iter % 2][idx1],
            gen[(iter + 1) % 2][0],
            gen[(iter + 1) % 2][1]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx0],
            gen[iter % 2][idx2],
            gen[(iter + 1) % 2][2],
            gen[(iter + 1) % 2][3]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx0],
            gen[iter % 2][idx3],
            gen[(iter + 1) % 2][4],
            gen[(iter + 1) % 2][5]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx0],
            gen[iter % 2][idx4],
            gen[(iter + 1) % 2][6],
            gen[(iter + 1) % 2][7]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx0],
            gen[iter % 2][idx5],
            gen[(iter + 1) % 2][8],
            gen[(iter + 1) % 2][9]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx1],
            gen[iter % 2][idx2],
            gen[(iter + 1) % 2][10],
            gen[(iter + 1) % 2][11]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx1],
            gen[iter % 2][idx3],
            gen[(iter + 1) % 2][12],
            gen[(iter + 1) % 2][13]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx1],
            gen[iter % 2][idx4],
            gen[(iter + 1) % 2][14],
            gen[(iter + 1) % 2][15]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx2],
            gen[iter % 2][idx3],
            gen[(iter + 1) % 2][16],
            gen[(iter + 1) % 2][17]
        );
        to_target_brain_x_crossover(
            gen[iter % 2][idx2],
            gen[iter % 2][idx4],
            gen[(iter + 1) % 2][18],
            gen[(iter + 1) % 2][19]
        );

        for (auto& e : gen[(iter + 1) % 2])
        {
            if (random::randfloat() < 0.6f)
            {
                e.mutate([]<std::floating_point F>(F f) {
                    const auto r = random::randfloat();
                    if (r < 0.0006f)
                    {
                        return random::randfloat();
                    }
                    if (r < 0.0012f)
                    {
                        return f * random::randnormal();
                    }
                    if (r < 0.0018f)
                    {
                        return decltype(f){};
                    }
                    if (r < 0.0024f)
                    {
                        return f += random::randnormal(
                                   random::randnormal(), random::randfloat()
                               );
                    }
                    if (r < 0.0030f)
                    {
                        return random::randnormal(f, 1);
                    }
                    return f;
                });
            }
        }

        if (iter % 51 == 0)
        {
            std::cout << iter << " " << *std::ranges::min_element(error)
                      << '\n';
            plotting_utility::plot(best_pred);
        }
    }
}

int main()
{
    // test0();
    // test2();
    training_test();

    return EXIT_SUCCESS;
}
