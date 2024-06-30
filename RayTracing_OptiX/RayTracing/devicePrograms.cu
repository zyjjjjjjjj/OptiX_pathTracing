// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "gdt/random/random.h"
#include "MyInteraction.h"
#include "PostProcess.h"
#include "Material_def.h"
#include "MyMaterial.h"
#include "MyTools.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 1
#define ENVMAP 1
#define PI 3.1415926

namespace osc {

    //typedef gdt::LCG<16> Random;

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    /*! per-ray data now captures random number generator, so programs
        can access RNG state */
    //struct PRD {
    //    Random random;
    //    vec3f  pixelColor;
    //};

    static __forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
        void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__shadow()
    {
        /* not going to be used ... */
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        uint32_t isectPtr0 = optixGetPayload_0();
        uint32_t isectPtr1 = optixGetPayload_1();
        Interaction* interaction = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        const int   primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;
        const vec3f& A = sbtData.vertex[index.x];
        const vec3f& B = sbtData.vertex[index.y];
        const vec3f& C = sbtData.vertex[index.z];
        const vec3f pos = (1.f - u - v) * A + u * B + v * C;
        interaction->position = pos;
        vec3f Ng = cross(B - A, C - A);
        vec3f Ns = (sbtData.normal)
            ? ((1.f - u - v) * sbtData.normal[index.x]
                + u * sbtData.normal[index.y]
                + v * sbtData.normal[index.z])
            : Ng;
        interaction->realNormal = Ng;

        const vec3f rayDir = optixGetWorldRayDirection();

        if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
        Ng = normalize(Ng);

        if (dot(Ng, Ns) < 0.f)
            Ns -= 2.f * dot(Ng, Ns) * Ng;
        Ns = normalize(Ns);

        interaction->geomNormal = Ns;
        if (sbtData.texcoord) {
            interaction->texcoord = (1.f - u - v) * sbtData.texcoord[index.x]
                + u * sbtData.texcoord[index.y]
                + v * sbtData.texcoord[index.z];
        }
        interaction->texture = sbtData.texture;
        interaction->mat_mes = sbtData.mat_mes;
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }

    extern "C" __global__ void __anyhit__shadow()
    { /*! not going to be used */
    }

    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    static __device__ vec2f sampling_equirectangular_map(vec3f n) {
        float u = atan(n.z / n.x);
		if (n.x < 0.0)
			u += PI;
        u = (u) / (2.0 * PI);
        float v = asin(n.y);
        v = (v * 2.0 + PI) / (2.0 * PI);
        v = 1.0f - v;
        return vec2f(u, v);
    }

    extern "C" __global__ void __miss__radiance()
    {
        uint32_t isectPtr0 = optixGetPayload_0();
        uint32_t isectPtr1 = optixGetPayload_1();
        Interaction* interaction = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
        interaction->distance = FLT_MAX;
        const cudaTextureObject_t& sbtData
            = *(const cudaTextureObject_t*)optixGetSbtDataPointer();
        vec3f ray_dir = optixGetWorldRayDirection();
        vec2f uv = sampling_equirectangular_map(ray_dir);
        vec4f fromTexture = tex2D<float4>(sbtData, uv.x, uv.y);
        interaction->mat_mes.emitter = (vec3f)fromTexture;
    }

    extern "C" __global__ void __miss__shadow()
    {
        
        // we didn't hit anything, so the light is visible
        vec3f& prd = *(vec3f*)getPRD<vec3f>();
        prd = vec3f(1.f);
        
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        PRD prd;
        prd.random.init(optixLaunchParams.frame.frameID * 384758 % 37895 + ix * 674673 % 13253,
            optixLaunchParams.frame.frameID * 564735 % 14524 + iy * 247857 % 45367);
        float x_bias = prd.random();
        prd.random.init(optixLaunchParams.frame.frameID * 477635 % 15664 + ix * 765843 % 36685,
            optixLaunchParams.frame.frameID * 153267 % 34663 + iy * 265547 % 75624);
        float y_bias = prd.random();

        const auto& camera = optixLaunchParams.camera;
        int numPixelSamples = optixLaunchParams.numPixelSamples;
		int numLightSamples = NUM_LIGHT_SAMPLES;
        vec3f pixelColor = 0.f;
        // normalized screen plane position, in [0,1]^2
        vec2f screen(vec2f(ix + x_bias, iy + y_bias) / vec2f(optixLaunchParams.frame.size));
        // generate ray direction
        vec3f rayDir = normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);
        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
            Ray myRay;
            myRay.origin = camera.position;
            myRay.direction = rayDir;
            vec3f radiance = 0.0f;
            vec3f accum = 1.0f;
            for (int bounces = 0; ; ++bounces)
            {
                prd.random.init(optixLaunchParams.frame.frameID * 384758 % 37895 + bounces * 478275 % 78475 + ix * 674673 % 13253,
                    optixLaunchParams.frame.frameID * 564735 % 14524 + bounces * 896835 % 56456 + iy * 247857 % 45367);
                //if (bounces > 0 && prd.random() > optixLaunchParams.prob) {
                if (bounces >= optixLaunchParams.maxBounce) {
                    //radiance = 0.0f;
                    break;
                }

                Interaction isect;
                isect.distance = 0;
                unsigned int isectPtr0, isectPtr1;
                packPointer(&isect, isectPtr0, isectPtr1);
                optixTrace(optixLaunchParams.traversable,
                    myRay.origin,
                    myRay.direction,
                    0,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                    RADIANCE_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    RADIANCE_RAY_TYPE,            // missSBTIndex 
                    isectPtr0, isectPtr1);
                if (isect.distance == FLT_MAX)
                {
                    if (bounces > 0)
                        radiance += isect.mat_mes.emitter* vec3f(2.0)* accum * optixLaunchParams.envmap;
                    else
                        radiance += isect.mat_mes.emitter * vec3f(2.0) * accum;
                    break;
                }
                radiance += isect.mat_mes.emitter * accum;
                vec3f wo;
                float pdf = 0.0f;
                vec3f bsdf = cal_bsdf(isect, myRay.direction, &wo, &pdf, ix, iy, optixLaunchParams.frame.frameID);
                float cosine = fabsf(dot(isect.geomNormal, wo));
                //accum *= bsdf * cosine / pdf;
                //if(bounces > 0)
                    //accum /= optixLaunchParams.prob;
                //accum *= bsdf / pdf;
                myRay = isect.spawn_ray(wo);
                
                if(isect.mat_mes.specTrans == 0 && isect.mat_mes.metallic == 0)
                {
                    vec3f lightPos;
                    for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++)
                    {
                        prd.random.init(optixLaunchParams.frame.frameID * 435874 % 45634 + bounces * 247857 % 78475 + ix * 478275 % 13253,
                            optixLaunchParams.frame.frameID * 674673 % 14524 + bounces * 564735 % 56456 + iy * 896835 % 45367);
                        lightPos = optixLaunchParams.light.origin + prd.random() * optixLaunchParams.light.du + prd.random() * optixLaunchParams.light.dv;
                        
                        vec3f lightDir = lightPos - myRay.origin;
                        float lightDist = gdt::length(lightDir);
                        lightDir = normalize(lightDir);
                        
                        float NdotL = dot(isect.geomNormal, lightDir);
                        if (NdotL < 0.0f)
                            continue;
                        
                        vec3f lightVisibility = 0.f;
                        uint32_t u0, u1;
                        packPointer(&lightVisibility, u0, u1);
                        optixTrace(optixLaunchParams.traversable,
                            myRay.origin + 1e-3f * isect.geomNormal,
                            lightDir,
                            1e-3f,      // tmin
                            lightDist * (1.f - 1e-3f),  // tmax
                            0.0f,       // rayTime
                            OptixVisibilityMask(255),
                            // For shadow rays: skip any/closest hit shaders and terminate on first
                            // intersection with anything. The miss shader is used to mark if the
                            // light was visible.
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT
                            | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                            | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                            SHADOW_RAY_TYPE,            // SBT offset
                            RAY_TYPE_COUNT,               // SBT stride
                            SHADOW_RAY_TYPE,            // missSBTIndex 
                            u0, u1);

                        vec3f color = getColor(isect);
                        vec3f rate = NdotL / (numLightSamples);
                        if(optixLaunchParams.light.lightType==0)
							rate /= (lightDist * lightDist);
                        radiance += lightVisibility * optixLaunchParams.light.power * rate * (color / (float)PI) * accum;
                        
                    }
                }
                
                accum *= bsdf * cosine/ pdf;
            }
            pixelColor += radiance;
        }

        vec4f rgba(pixelColor / numPixelSamples, 1.f);
        rgba.x = powf(rgba.x, 1 / 2.2f);
        rgba.y = powf(rgba.y, 1 / 2.2f);
        rgba.z = powf(rgba.z, 1 / 2.2f);
        if (rgba.x > 1)rgba.x = 1.0f;
        if (rgba.y > 1)rgba.y = 1.0f;
        if (rgba.z > 1)rgba.z = 1.0f;
        if (rgba.w > 1)rgba.w = 1.0f;
        // and write/accumulate to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        if (optixLaunchParams.frame.frameID > 0) {
            rgba
                += float(optixLaunchParams.frame.frameID)
                * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
            rgba /= (optixLaunchParams.frame.frameID + 1.f);
        }
        optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;

        HSV hsv; BGR bgr; bgr.r = rgba.x; bgr.g = rgba.y; bgr.b = rgba.z;
        BGR2HSV(bgr, hsv);
        hsv.v += optixLaunchParams.lightness_change;
        if (hsv.s >= 0.05f)
            hsv.s += optixLaunchParams.saturate_change;
        HSV2BGR(hsv, bgr);
        Contrast(bgr, optixLaunchParams.contrast_change, 0.5f);
        rgba.x = bgr.r; rgba.y = bgr.g; rgba.z = bgr.b;
        optixLaunchParams.frame.renderBuffer[fbIndex] = (float4)rgba;
    }

} // ::osc