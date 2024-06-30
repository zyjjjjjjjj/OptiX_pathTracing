#pragma once
#include "gdt/math/vec.h"
#include "math.h"
#include "LaunchParams.h"

using namespace gdt;

struct Ray
{
    vec3f origin;
    vec3f direction;
    float tmax = FLT_MAX;
};
struct Interaction
{
    float bias = 0.001f;
    float distance;
    vec3f position;
    vec3f geomNormal;
    vec3f realNormal;
    vec2f texcoord;
    cudaTextureObject_t* texture;
    material_mes mat_mes;

    __forceinline__ __device__ Ray spawn_ray(const vec3f& wi) const
    {
        vec3f N = geomNormal;
        if (dot(wi, geomNormal) < 0.0f)
        {
            N = -geomNormal;
        }
        Ray ray;
        ray.origin = position + N * bias;
        ray.direction = wi;
        ray.tmax = FLT_MAX;
        return ray;
    }
};
