#pragma once
#include "gdt/math/vec.h"
#include <cuda_runtime.h>
using namespace gdt;
enum material_kind
{
    DIFFUSE, METAL, DIELECTRIC, TESTMAT, NEW
};
struct material_mes {
    material_kind mat_kind;
    vec3f diffuse;
    vec3f specular;
    vec3f transmittance;
    vec3f emitter = 0;

    float roughness;
    float transparent;
    float metallic;
    float specTrans;
    float ior = 1.1;
    int diffuseTextureID{ -1 };
    cudaTextureObject_t diffuse_texture;
};