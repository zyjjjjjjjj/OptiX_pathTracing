#pragma once
#include "gdt/math/vec.h"
#include "math.h"
#include "optix7.h"

using namespace gdt;

/*PRD是一个随机采样器，通过prd.random.init()播随机种子，然后prd.random()得到(0,1)的随机数*/
typedef gdt::LCG<16> Random;

struct PRD {
    Random random;
    vec3f  pixelColor;
};

__forceinline__ __device__ float my_min(const float a, const float b) {
    return a < b ? a : b;
}

__forceinline__ __device__ float length_squared(const vec3f v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

/*已知入射向量、法线，求反射向量*/
__forceinline__ __device__ vec3f reflect(const vec3f v, const vec3f n, float roughness, const int ix, const int iy, const int frame_id) {

    vec3f r = normalize(v - 2 * dot(v, n) * n);
    return r;
    /*vec3f rnd;
    PRD prd;
    prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921,
        frame_id * 348593 % 43832 + iy * 324123 % 23415);
    rnd.x = prd.random() * 2 - 1;
    prd.random.init(frame_id * 972823 % 12971 + ix * 743782 % 82013,
        frame_id * 893022 % 28191 + iy * 918212 % 51321);
    rnd.y = prd.random() * 2 - 1;
    prd.random.init(frame_id * 383921 % 48839 + ix * 572131 % 47128,
        frame_id * 389291 % 29301 + iy * 716271 % 63291);
    rnd.z = prd.random() * 2 - 1;
    vec3f wos = normalize(n + normalize(rnd));

    return normalize(roughness * wos + (1 - roughness) * r);
    */
    
}

/*已知入射向量uv、法线、折射率，求折射向量*/
__forceinline__ __device__ vec3f refract(const vec3f uv, const vec3f n, float roughness, double etai_over_etat, const int ix, const int iy, const int frame_id) {
    auto cos_theta = dot(-uv, n);
    vec3f r_out_perp = (float)etai_over_etat * (uv + cos_theta * n);
    vec3f r_out_parallel = (float)(-sqrt(fabs(1.0 - length_squared(r_out_perp)))) * n;
    vec3f r = normalize(r_out_perp + r_out_parallel);

    return r;
    /*

    */
}

/*菲涅尔项的近似拟合*/
__forceinline__ __device__ double schlick(double cosine, double ref_idx) {
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 *= r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__forceinline__ __device__ vec3f schlick_vec3f(double cosine, vec3f f0) {
    return f0 + (vec3f(1.0) - f0) * vec3f(pow((1 - cosine), 5));
}