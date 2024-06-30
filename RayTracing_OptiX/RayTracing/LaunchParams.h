#pragma once

#include "gdt/math/vec.h"
#include "Material_def.h"
#include "optix7.h"

#define ENVMAP 1

namespace osc {
    using namespace gdt;

    enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

    struct TriangleMeshSBTData {
        vec3f  color;
        vec3f* vertex;
        vec3f* normal;
        vec2f* texcoord;
        vec3i* index;
        material_mes mat_mes;
        bool                hasTexture;
        cudaTextureObject_t* texture;
    };

    struct LaunchParams
    {
        int envmap = ENVMAP;
        int numPixelSamples = 1;
		int maxBounce = 24;
        float prob = 0.5;
		//sponza 0.35 -0.03 1.5
        float lightness_change = 0.0f;       //亮度后处理
        float saturate_change = -0.00f;        //饱和度后处理
        float contrast_change = 0.0f;        //对比度后处理

        struct {
            int       frameID = 0;
            float4* colorBuffer;
            float4* renderBuffer;

            /*! the size of the frame buffer to render */
            vec2i     size;
        } frame;

        struct {
            vec3f position;
            vec3f direction;
            vec3f horizontal;
            vec3f vertical;
        } camera;

        OptixTraversableHandle traversable;

        struct {
            vec3f origin, du, dv, power;
			int lightType;//0:area light 1:direc light
        } light;
    };

} // ::osc