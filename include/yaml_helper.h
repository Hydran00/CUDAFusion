#include<yaml-cpp/yaml.h>
#include<vector>
static std::vector<int> read_vec_i(const YAML::Node &n)
{
    std::vector<int> v;
    for (auto x : n) v.push_back(x.as<int>());
    return v;
}

static std::vector<float> read_vec_f(const YAML::Node &n)
{
    std::vector<float> v;
    for (auto x : n) v.push_back(x.as<float>());
    return v;
}

static float3 read_float3(const YAML::Node &n)
{
    return make_float3(
        n[0].as<float>(),
        n[1].as<float>(),
        n[2].as<float>()
    );
}

