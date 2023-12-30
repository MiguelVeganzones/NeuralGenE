#include <array>
#include <bitset>
#include <iostream>
#include <ranges>

#include "Random.hpp"
#include "Stopwatch.hpp"
#include "c4_board.hpp"
#include "game_boardd.hpp"
#include "polystate.hpp"

int main0()
{
    random::seed();

    constexpr std::size_t M = 20;
    constexpr std::size_t N = 20;
    constexpr std::size_t K = 3;

    game_board::board_2D<M, N, K, uint8_t> b{};
    std::array<uint64_t, M * N>            a{};

    auto f = [&]() {
        for (int j = 0; j != M; ++j)
        {
            for (int i = 0; i != N; ++i)
            {
                b.set_position(
                    { j, i },
                    static_cast<decltype(b)::repr_type>(
                        random::randint(0, K - 1)
                    )
                );
            }
        }
    };

    auto g = [&]() {
        for (size_t j = 0; j != M; ++j)
        {
            for (size_t i = 0; i != N; ++i)
            {
                a[j * N + i] = random::randint(0, K - 1);
            }
        }
    };

    std::cout << b.m_State << std::endl;
    std::cout << sizeof b << std::endl;

    bench::multiple_run_bench(100000, f);
    bench::multiple_run_bench(100000, g);

    for (auto e : a)
    {
        std::cout << e << ' ';
    }
    std::cout << '\n';

    std::cout << b.m_State << "\n\n";
    b.reset();
    std::cout << b.m_State << '\n';


    return EXIT_SUCCESS;
}

int main()
{
    random::seed();

    constexpr std::size_t M = 6;
    constexpr std::size_t N = 7;
    constexpr std::size_t K = 2;

    using board_type = c4_board::board<M, N, 3, K>;

    board_type b{};

    for (int i = 0; i != 100000; ++i)
    {
        std::cout << random::randfloat() << std::endl;
    }

    int i = 0;
    while (b.any_moves_left() && random::randfloat() > 0.03f)
    {
        for (const auto& p : b.get_valid_moves())
        {
            std::cout << p << " :: ";
            if (p && random::randfloat() < 0.2f)
            {
                b.make_move(p, (i++) % K);
                std::cout << std::endl;
                std::cout << b.board_state();
                std::cout << "Winning move: " << b.winning_move(p) << std::endl;
                std::cout << "Leftd: " << b.leftd_count << std::endl;
                std::cout << "rightd: " << b.rightd_count << std::endl;
                std::cout << "horizontal: " << b.horizontal_count << std::endl;
                std::cout << "vertcal: " << b.vertical_count << std::endl
                          << std::endl;
                break;
            }
        }
        std::cout << std::endl;
    }

    auto e = b.encode();

    board_type::encode_type_hasher hash{};

    std::cout << "Hash:\n" << hash(e) << std::endl;

    for (auto _e : e)
    {
        std::cout << (int)_e << ' ';
    }
    std::cout << std::endl;


    const auto decoded = board_type::decode(e);

    std::cout << decoded.board_state() << std::endl;
    for (auto _e : decoded.get_valid_moves())
    {
        std::cout << _e << ' ';
    }
    std::cout << std::endl;

    std::cout << b.board_state() << std::endl;

    for (auto _e : b.get_valid_moves())
    {
        std::cout << _e << ' ';
    }
    std::cout << std::endl;
}