#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <variant>
#include <functional>
#include <cstdlib>

template<class... Ts>
struct overload : Ts... {
    using Ts::operator()...;   // 把所有 operator() 带进来
};

template<class... Ts>
overload(Ts...) -> overload<Ts...>; // CTAD 推导指引


struct Arg { 
    std::string name;
    std::variant<
        std::reference_wrapper<int>,
        std::reference_wrapper<u_int64_t>,
        std::reference_wrapper<float>,
        std::reference_wrapper<double>,
        std::reference_wrapper<bool>,
        std::reference_wrapper<std::string>
    > value;
} ;


class ArgParser {
private:
    std::vector<Arg> args_;
    void parse_args_(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string token = argv[i];
            for (auto& arg : args_) {
                if (token == arg.name) {
                    std::visit([&](auto&& ref) {
                        using T = std::decay_t<decltype(ref.get())>;
                        if constexpr (std::is_same_v<T, bool>) {
                            ref.get() = true;  // bool 类型不用取下一个参数
                        } else {
                            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + token);
                            ++i;
                            if constexpr (std::is_same_v<T, int>)
                                ref.get() = std::atoi(argv[i]);
                            else if constexpr (std::is_same_v<T, u_int64_t>)
                                ref.get() = std::atoll(argv[i]);
                            else if constexpr (std::is_same_v<T, float>)
                                ref.get() = std::atof(argv[i]);
                            else if constexpr (std::is_same_v<T, double>)
                                ref.get() = std::atof(argv[i]);
                            else if constexpr (std::is_same_v<T, std::string>)
                                ref.get() = argv[i];
                        }
                    }, arg.value);
                }
            }
        }
    }
public:
    template<typename T>
    void add_reference(const std::string& name, T& var){
        args_.push_back({name, std::ref(var)});
    }
    void parse(int argc, char**argv){
        try {
            parse_args_(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "Error during argument parsing: " << e.what() << std::endl;
            // 可以在这里选择退出程序或重新抛出
            throw; 
        }
    }
    void print_usage_and_exit(){ 
        std::cout<<"Usage: \n";
        for(auto &arg: args_){
            std::string item = arg.name+ " : ";
            item += std::visit(
                [&](auto &&ref) -> std::string {
                    using T = std::decay_t<decltype(ref.get())>;
                    if constexpr (std::is_same_v<T, int>)
                        return "int";
                    else if constexpr (std::is_same_v<T, u_int64_t>)
                        return "uint64";
                    else if constexpr (std::is_same_v<T, float>)
                        return "float";
                    else if constexpr (std::is_same_v<T, double>)
                        return "double";
                    else if constexpr (std::is_same_v<T, bool>)
                        return "bool (flag, no value needed)";
                    else if constexpr (std::is_same_v<T, std::string>)
                        return "string";
                    else
                        return "unknown";
            }, arg.value);
            std::cout<<item<<"\n";
        }
        exit(0);
    }
} ;
