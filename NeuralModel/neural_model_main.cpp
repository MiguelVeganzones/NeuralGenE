#include <array>
#include <concepts>
#include <filesystem>
#include <future>
#include <thread>
#include <vector>

#include "Random.hpp"
#include "Stopwatch.hpp"
#include "activation_functions.hpp"
#include "data_processor.hpp"
#include "neural_model.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"

void init_plotting(const std::size_t len)
{
    const std::filesystem::path py_path =
        "../Python Plotting/Python_Plotting.py";
    const std::filesystem::path data_path =
        "../__Plotting_files/plotting_numbers.txt";

    std::stringstream s;

    s << "python " << py_path << ' ' << data_path << ' ' << len;

    [[maybe_unused]] auto t = system(s.str().c_str());
}

template <typename Iterable>
void output_to_file(const std::size_t gen, const Iterable& iter)
{
    const std::filesystem::path data_path =
        "../__Plotting_files/plotting_numbers.txt";
    std::ofstream f(data_path, std::ofstream::out | std::ofstream::trunc);

    if (!f.is_open())
        throw;
    f << gen << '\n';
    f << std::setprecision(2);
    for (const float e : iter)
        f << e << ' ';
    f.close();
}

void training_test()
{
    // random::init();

    // constexpr std::size_t N = 20;
    // constexpr std::size_t M = 2000;

    // // std::thread(init_plotting, M).detach();

    // using namespace ga_snn;
    // using namespace ga_sm;

    // constexpr auto fn = score_functions::score_functions<double,
    // std::uint16_t>::
    //     choose_function<score_functions::Identifiers::Weighted_normalized_score,
    //     3, 1, 0>();

    // constexpr auto AF_relu = matrix_activation_functions::Identifiers::GELU;
    // constexpr auto AF_tanh =
    // matrix_activation_functions::Identifiers::UnsignedSigmoid; auto SM =
    //     score_function_objects::score_function_object<decltype(fn),
    //     std::uint16_t, std::uint16_t, std::uint16_t>(fn);


    // using SM_t = decltype(SM);

    // [[maybe_unused]] constexpr Layer_Signature a1{ 1, AF_relu };
    // [[maybe_unused]] constexpr Layer_Signature a1_tanh{ 1, AF_tanh };
    // [[maybe_unused]] constexpr Layer_Signature a9{ 9, AF_relu };
    // [[maybe_unused]] constexpr Layer_Signature a25{ 25, AF_relu };
    // [[maybe_unused]] constexpr Layer_Signature a42{ 42, AF_relu };
    // [[maybe_unused]] constexpr Layer_Signature a16{ 16, AF_relu };

    // using NET = static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a9, a9,
    // a9, a1_tanh>;

    // std::cout << NET::parameter_count() << std::endl;
    // // std::cout << NET2::parameter_count() << std::endl;

    // std::array<std::array<ga_neural_model::brain<NET, SM_t>, N>, 2> gen{};

    // for (size_t i = 0; i != N; ++i)
    // {
    //     gen[0][i] = ga_neural_model::brain<NET, SM_t>(SM, [] { return
    //     random::randnormal() * 0.001f; }); gen[1][i] =
    //     ga_neural_model::brain<NET, SM_t>(SM, [] { return 0.f; });
    // }

    // std::vector<float> in(M), out(M), pred(M), best_pred(M);
    // std::iota(in.begin(), in.end(), 0.f);
    // std::ranges::transform(in, out.begin(), [](float i) { return std::sin(i /
    // 100); });

    // for (size_t iter = 0; iter != 100000; ++iter)
    // {
    //     std::array<float, N> error{};
    //     for (size_t j = 0; j != N; ++j)
    //     {
    //         for (size_t i = 0; i != M; ++i)
    //         {
    //             pred[i] = gen[iter % 2][j].weigh(in[i]);
    //             error[j] += std::abs(pred[i] - out[i]);
    //         }
    //         // if (j == 0 || (j > 0 && error[j] < error[j - 1]))
    //         //{
    //         //     best_pred = pred;
    //         // }
    //     }

    //     // for (const auto e :error)
    //     //{
    //     //     std::cout << e << ' ';
    //     // }
    //     // std::cout << '\n';

    //     auto temp = std::array(error);
    //     std::ranges::sort(temp);

    //     const auto idx0 = std::ranges::find(error, temp[0]) - error.begin();
    //     const auto idx1 = std::ranges::find(error, temp[1]) - error.begin();
    //     const auto idx2 = std::ranges::find(error, temp[2]) - error.begin();
    //     const auto idx3 = std::ranges::find(error, temp[3]) - error.begin();
    //     const auto idx4 = std::ranges::find(error, temp[4]) - error.begin();
    //     const auto idx5 = std::ranges::find(error, temp[5]) - error.begin();

    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx0], gen[iter % 2][idx1], gen[(iter + 1) % 2][0],
    //         gen[(iter + 1) % 2][1]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx0], gen[iter % 2][idx2], gen[(iter + 1) % 2][2],
    //         gen[(iter + 1) % 2][3]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx0], gen[iter % 2][idx3], gen[(iter + 1) % 2][4],
    //         gen[(iter + 1) % 2][5]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx0], gen[iter % 2][idx4], gen[(iter + 1) % 2][6],
    //         gen[(iter + 1) % 2][7]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx0], gen[iter % 2][idx5], gen[(iter + 1) % 2][8],
    //         gen[(iter + 1) % 2][9]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx1], gen[iter % 2][idx2], gen[(iter + 1) %
    //         2][10], gen[(iter + 1) % 2][11]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx1], gen[iter % 2][idx3], gen[(iter + 1) %
    //         2][12], gen[(iter + 1) % 2][13]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx1], gen[iter % 2][idx4], gen[(iter + 1) %
    //         2][14], gen[(iter + 1) % 2][15]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx2], gen[iter % 2][idx3], gen[(iter + 1) %
    //         2][16], gen[(iter + 1) % 2][17]);
    //     to_target_brain_x_crossover(
    //         gen[iter % 2][idx2], gen[iter % 2][idx4], gen[(iter + 1) %
    //         2][18], gen[(iter + 1) % 2][19]);

    //     for (auto& e : gen[(iter + 1) % 2])
    //     {
    //         if (random::randfloat() < 0.6f)
    //         {
    //             e.mutate([]<std::floating_point F>(F f) -> F {
    //                 const auto r = random::randfloat();
    //                 if (r >= 0.009)
    //                 {
    //                     return f * F(0.9999);
    //                 }
    //                 if (r < 0.0006f)
    //                 {
    //                     return random::randfloat();
    //                 }
    //                 if (r < 0.0012f)
    //                 {
    //                     return f * random::randnormal();
    //                 }
    //                 if (r < 0.0018f)
    //                 {
    //                     return F{};
    //                 }
    //                 if (r < 0.0024f)
    //                 {
    //                     return f += random::randnormal(random::randnormal(),
    //                     random::randfloat());
    //                 }
    //                 if (r < 0.0030f)
    //                 {
    //                     return random::randnormal(f, F(0.01));
    //                 }
    //                 return f;
    //             });
    //         }
    //     }

    //     if (iter % 99 == 0)
    //     {
    //         std::cout << iter << " " << *std::ranges::min_element(error) <<
    //         '\n';
    //         // output_to_file(iter, best_pred);
    //     }
    // }
}

void new_brain_test()
{
    // random::init();

    // std::thread(init_plotting, M).detach();

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
    // using NET = static_neural_net<float, 1, a1, a9, a9, a9, a25, a25, a9, a9,
    // a9, a1_tanh>;

    // std::cout << NET::parameter_count() << std::endl;
    // std::cout << NET2::parameter_count() << std::endl;

    using preprocessor  = data_processor::scalar_converter;
    using postprocessor = data_processor::scalar_converter;
    using return_type   = NET::value_type;

    using brain_t =
        ga_neural_model::brain<NET, preprocessor, postprocessor, return_type>;

    brain_t b(random::randnormal, 0, 1);

    static_matrix<float, 1, 1> m = { 0.f };
    std::cout << b.weigh(m) << std::endl;

    std::cout << b(1) << std::endl;

    std::cout
        << preprocessor::template process<int, static_matrix<float, 1, 1>>(-1)
        << std::endl;

    std::cout
        << postprocessor::template process<static_matrix<float, 1, 1>, int>(m)
        << std::endl;
}

int main()
{
    new_brain_test();

    return EXIT_SUCCESS;
}
