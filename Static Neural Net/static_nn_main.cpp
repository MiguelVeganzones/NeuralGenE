#include <iostream>
#include <memory>

#include "Random.h"
#include "Stopwatch.h"
#include "static_matrix.hpp"
#include "static_neural_net.hpp"

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

    constexpr auto AF = matrix_activation_functions::Identifiers::ReLU;
    constexpr auto Swish = matrix_activation_functions::Identifiers::Swish;

    constexpr Layer_Signature a1{ 1, AF };
    // constexpr Layer_Signature a2{ 4, AF };
    constexpr Layer_Signature a3{ 9, AF };
    // constexpr Layer_Signature a4{ 16, AF };
    constexpr Layer_Signature a5{ 25, Swish };

    using NET  = static_neural_net<float, 1, a1, a3, a5, a5, a3, a1>;
    using NET2 = static_neural_net<float, 2, a1, a3, a5, a5, a3, a1>;

    auto ptr_net  = std::make_unique<NET>();
    auto ptr_net2 = std::make_unique<NET2>();
    auto ptr_net3 = std::make_unique<NET2>();

    ptr_net->init(random::randnormal, 0, 1);

    for (size_t i = 0; const auto& e : in)
    {
        std::cout << ptr_net->forward_pass<1, 1>(static_matrix<float, 1, 1>{ e })(0, 0) << std::endl;
        pred(i++, 0) = ptr_net->forward_pass<1, 1>(static_matrix<float, 1, 1>{ e })(0, 0);
    }

    ptr_net->print_net();
    std::cout << in << std::endl;

    std::cout << pred << std::endl;

    const std::string filename = "Flattened_test0.txt";
    ptr_net->store(filename);
    ptr_net2->load(filename);

    ptr_net2->print_net();

    std::cout << "L11:" << L11_net_distance<double>(*ptr_net3.get(), *ptr_net2.get()) << std::endl;

    for (size_t i = 0; const auto& e : in)
    {
        const auto res = ptr_net2->batch_forward_pass(static_matrix<float, 2, 1>{ e, e });
        std::cout << res << std::endl;
        pred2(i, 0)   = res(0, 0);
        pred2(i++, 1) = res(1, 0);
    }

    ptr_net3->init_from_ptr(ptr_net.get());

    // std::cout << L11_net_distance(ptr_net3.get(), ptr_net2.get()) << std::endl;

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

    using NNet = static_neural_net<T, N, ls5, ls10, ls25, ls50, ls25, ls10, ls5>;

    static_matrix<T, N, 5> in{};

    random::init();

    in.fill(random::randfloat);

    const auto ptr_net = static_neural_net_factory<NNet>(random::randnormal, 0, 1);
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

    const auto ptr_net = static_neural_net_factory<NNet>(random::randnormal, 0, 1);


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

    constexpr auto AF = matrix_activation_functions::Identifiers::Sigmoid;
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
    //net3.init(random::randfloat);
    //net4.init(random::randfloat);

    net1.print_net();
    net2.print_net();
    //net3.print_net();
    //net4.print_net();

    std::cout << net1.parameter_count() << " // " << sizeof(NNet) << std::endl;

    //to_target_layer_swap(net1, net2, net3, net4);
    auto[p1, p2] = layer_swap(net1, net2);

    p1->print_net();
    p2->print_net();
    //net3.print_net();
    //net4.print_net();

}

int main()
{
    //get_layer();

    //abench(100);

    layer_swap_test();


    return EXIT_SUCCESS;
}
