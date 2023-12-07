#include "Random.hpp"
#include "Stopwatch.hpp"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"
#include <iostream>
#include <memory>

int flat_test()
{
    random::init();
    stopwatch s0;
    using namespace ga_snn;
    using namespace ga_sm;

    constexpr size_t N1 = 50;

    ga_sm::static_matrix<float, N1, 1> in{}, pred{};
    ga_sm::static_matrix<float, N1, 2> pred2{};

    for (float i = 0; auto& e : in)
        e = i++;

    [[maybe_unused]] constexpr auto Identity =
        matrix_activation_functions::Identifiers::Identity;
    [[maybe_unused]] constexpr auto ReLU =
        matrix_activation_functions::Identifiers::ReLU;
    [[maybe_unused]] constexpr auto PReLU =
        matrix_activation_functions::Identifiers::PReLU;
    [[maybe_unused]] constexpr auto Sigmoid =
        matrix_activation_functions::Identifiers::Sigmoid;
    [[maybe_unused]] constexpr auto Swish =
        matrix_activation_functions::Identifiers::Swish;

    constexpr Layer_Signature a1{ 1, Identity };
    constexpr Layer_Signature a2{ 4, ReLU };
    constexpr Layer_Signature a3{ 9, PReLU };
    constexpr Layer_Signature a4{ 16, Sigmoid };
    constexpr Layer_Signature a5{ 25, Swish };

    using NET  = static_neural_net<float, 1, a1, a2, a3, a4, a5, a1>;
    using NET2 = static_neural_net<float, 2, a1, a2, a3, a4, a5, a1>;

    auto ptr_net   = std::make_unique<NET>();
    auto ptr_net2  = std::make_unique<NET2>();
    auto ptr_net21 = std::make_unique<NET2>();
    auto ptr_net3  = std::make_unique<NET2>();

    ptr_net->init(random::randnormal, 0, 1);

    for (size_t i = 0; const auto& e : in)
    {
        std::cout << ptr_net->forward_pass<1, 1>(static_matrix<float, 1, 1>{ e }
                     )[0, 0]
                  << std::endl;
        pred[i++, 0] =
            ptr_net->forward_pass<1, 1>(static_matrix<float, 1, 1>{ e })[0, 0];
    }

    ptr_net->print_net();
    std::cout << in << std::endl;

    std::cout << pred << std::endl;

    const std::string filename                  = "Flattened_test0.txt";
    const std::string serialized_store_filename = "Serialization_test0.bin";

    std::cout << "Here1\n";
    ptr_net->store(filename);
    ptr_net2->load(filename);

    std::cout << "Here2\n";
    ptr_net->serialize_store(serialized_store_filename);
    ptr_net21->deserialize_load(serialized_store_filename);

    ptr_net2->print_net();
    ptr_net21->print_net();

    std::cout << "L1:"
              << neural_net_distance<float>(
                     *ptr_net21.get(),
                     *ptr_net2.get(),
                     [](float a, float b) { return std::abs(a - b); }
                 )
              << std::endl;
    std::cout << "L1:"
              << neural_net_distance<float>(
                     *ptr_net.get(),
                     *ptr_net2.get(),
                     [](float a, float b) { return std::abs(a - b); }
                 )
              << std::endl;
    std::cout << "L1:"
              << neural_net_distance<float>(
                     *ptr_net.get(),
                     *ptr_net21.get(),
                     [](float a, float b) { return std::abs(a - b); }
                 )
              << std::endl;

    for (size_t i = 0; const auto& e : in)
    {
        const auto res =
            ptr_net2->batch_forward_pass(static_matrix<float, 2, 1>{ e, e });
        // std::cout << res << std::endl;
        pred2[i, 0]   = res[0, 0];
        pred2[i++, 1] = res[1, 0];
    }

    std::cout << pred << std::endl << pred2 << std::endl;

    return 0;
}

void abench(int n)
{
    using namespace ga_sm;
    using namespace ga_snn;
    using T = float;

    constexpr size_t N = 8;

    constexpr auto AF = matrix_activation_functions::Identifiers::ReLU;

    constexpr auto ls5  = Layer_Signature{ 5, AF };
    constexpr auto ls10 = Layer_Signature{ 10, AF };
    constexpr auto ls25 = Layer_Signature{ 25, AF };
    constexpr auto ls50 = Layer_Signature{ 50, AF };

    using NNet =
        static_neural_net<T, N, ls5, ls10, ls25, ls50, ls25, ls10, ls5>;

    static_matrix<T, N, 5> in{};

    random::init();

    in.fill(random::randfloat);

    const auto ptr_net =
        static_neural_net_factory<NNet>(random::randnormal, 0, 1);
    // ptr_net->print_net();

    // stopwatch s;

    while (n--)
    {
        std::cout << in << std::endl;
        in = ptr_net->batch_forward_pass(in);
        in.standarize();
    }
    std::cout << in << std::endl;
}

void get_layer()
{
    using namespace ga_sm;
    using namespace ga_snn;
    using T = float;

    constexpr size_t N = 8;

    constexpr auto AF = matrix_activation_functions::Identifiers::Sigmoid;

    constexpr auto ls5 = Layer_Signature{ 5, AF };
    // constexpr auto ls10 = Layer_Signature{ 10, AF };
    // constexpr auto ls25 = Layer_Signature{ 25, AF };
    // constexpr auto ls50 = Layer_Signature{ 50, AF };

    using NNet = static_neural_net<T, N, ls5, ls5, ls5>;

    const auto ptr_net =
        static_neural_net_factory<NNet>(random::randnormal, 0, 1);


    ptr_net->print_net();

    for (int i = 0; i < 10; ++i)
    {
        int idx = random::randint(0, NNet::s_Layers - 1);
        std::cout << idx << std::endl;
        ptr_net->mutate_layer(idx, [](auto) { return random::randfloat(); });
        ptr_net->print_net();
    }
}

void layer_swap_test()
{
    random::init();

    using namespace ga_sm;
    using namespace ga_snn;
    using T = float;

    constexpr size_t N = 8;

    constexpr auto AF    = matrix_activation_functions::Identifiers::Sigmoid;
    constexpr auto Swish = matrix_activation_functions::Identifiers::Swish;

    constexpr auto ls1 = Layer_Signature{ 1, AF };
    constexpr auto ls5 = Layer_Signature{ 5, Swish };
    // constexpr auto ls10 = Layer_Signature{ 10, AF };
    // constexpr auto ls25 = Layer_Signature{ 25, AF };
    // constexpr auto ls50 = Layer_Signature{ 50, AF };

    using NNet = static_neural_net<T, N, ls1, ls5, ls5, ls5, ls1>;

    NNet net1{};
    NNet net2{};
    NNet net3{};
    NNet net4{};

    net1.init(random::randfloat);
    net2.init(random::randfloat);
    // net3.init(random::randfloat);
    // net4.init(random::randfloat);

    net1.print_net();
    net2.print_net();
    // net3.print_net();
    // net4.print_net();

    std::cout << net1.parameter_count() << " // " << sizeof(NNet) << std::endl;

    // to_target_layer_swap(net1, net2, net3, net4);
    auto [p1, p2] = layer_swap(net1, net2);

    p1->print_net();
    p2->print_net();
    // net3.print_net();
    // net4.print_net();
}

int activation_functions_test()
{
    // random::init();
    stopwatch s0;
    using namespace ga_snn;
    using namespace ga_sm;

    constexpr int N = 5;

    ga_sm::static_matrix<float, N, 1> in{};

    for (float i = -(N / 2); auto& e : in)
        e = i++;

    constexpr auto Identity =
        matrix_activation_functions::Identifiers::Identity;
    constexpr auto ReLU    = matrix_activation_functions::Identifiers::ReLU;
    constexpr auto PReLU   = matrix_activation_functions::Identifiers::PReLU;
    constexpr auto Sigmoid = matrix_activation_functions::Identifiers::Sigmoid;
    constexpr auto Swish   = matrix_activation_functions::Identifiers::Swish;

    constexpr Layer_Signature identity_layer{ 1, Identity };
    constexpr Layer_Signature relu_layer{ 4, ReLU };
    constexpr Layer_Signature prelu_layer{ 5, PReLU };
    constexpr Layer_Signature sigmoid_layer{ 4, Sigmoid };
    constexpr Layer_Signature swish_layer{ 2, Swish };

    using NET = static_neural_net<
        float,
        1,
        identity_layer,
        relu_layer,
        prelu_layer,
        sigmoid_layer,
        swish_layer>;

    auto ptr_net = std::make_unique<NET>();

    ptr_net->init(random::randnormal, 0, 1);

    ptr_net->print_net();

    for (const auto& e : in)
    {
        std::cout << "Input:: " << e << "\n###############\n";
        std::cout << ptr_net->forward_pass<1, 2>(static_matrix<float, 1, 1>{ e }
                     )
                  << std::endl;
    }

    ptr_net->print_net();

    return 0;
}

void matirx_crossover_test()
{
    random::init();
    constexpr auto M = 1;
    constexpr auto N = 7;
    using value_type = float;

    ga_sm::static_matrix<value_type, M, N> mat1;
    ga_sm::static_matrix<value_type, M, N> mat2;
    ga_sm::static_matrix<value_type, M, N> mat3;
    ga_sm::static_matrix<value_type, M, N> mat4;

    mat1.fill(random::randnormal, 0, 1);
    // mat2.fill(random::randnormal, 0, 1);
    mat3.fill(random::randnormal, 0, 1);
    mat4.fill(random::randnormal, 0, 1);

    std::cout << mat1;
    std::cout << mat2;
    std::cout << mat3;
    std::cout << mat4;

    ga_sm::to_target_x_crossover(mat1, mat2, mat3, mat4);

    std::cout << mat1;
    std::cout << mat2;
    std::cout << mat3;
    std::cout << mat4;
}

int main()
{
    // get_layer();

    // abench(100);

    // layer_swap_test();

    matirx_crossover_test();

    // flat_test();

    // // activation_functions_test();

    // ga_sm::static_matrix<double, 6, 5> mat{};
    // mat.fill(random::randnormal, 0, 10);

    // std::cout << mat << std::endl;
    // std::cout << ga_sm::reduce<1, 1>(mat, [](auto a, auto b) { return a + b;
    // }) << std::endl; std::cout << ga_sm::reduce<6, 1>(mat, [](auto a, auto b)
    // { return a + b; }) << std::endl; std::cout << ga_sm::reduce<1, 5>(mat,
    // [](auto a, auto b) { return a + b; }) << std::endl;

    // auto af = matrix_activation_functions::
    //     activation_function<decltype(mat),
    //     matrix_activation_functions::Identifiers::Softmax>();

    // std::cout << mat << std::endl;

    // af(mat);

    // std::cout << mat << std::endl;


    return EXIT_SUCCESS;
}
