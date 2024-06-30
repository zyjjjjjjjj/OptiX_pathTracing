#pragma once
#include "gdt/math/AffineSpace.h"
#include "Material_def.h"
#include <vector>
/*! \namespace osc - Optix Siggraph Course */
namespace osc {
    using namespace gdt;
    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec2f> texcoord;
        std::vector<vec3i> index;
        // material data:
        material_mes mat_mes;
    };

    struct Texture {
        ~Texture()
        {
            if (pixel) delete[] pixel;
        }

        uint32_t* pixel{ nullptr };
        vec2i     resolution{ -1 };
    };

    struct Model {
        ~Model()
        {
            for (auto mesh : meshes) delete mesh;
            for (auto texture : textures) delete texture;
        }

        std::vector<TriangleMesh*> meshes;
        std::vector<Texture*>      textures;
        Texture* envmap;
        //! bounding box of all vertices in the model
        box3f bounds;
    };

    Model* loadOBJ(const std::string& objFile, material_kind mat_kind);

    int loadEnvmap(Model* model, const std::string& Path);

    struct QuadLight {
        vec3f origin, du, dv, power;
		int lightType;//0:area light 1:direc light
    };
}
