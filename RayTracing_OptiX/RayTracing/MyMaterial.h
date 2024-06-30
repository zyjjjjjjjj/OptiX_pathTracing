#pragma once
#include "MyInteraction.h"
#include "gdt/random/random.h"
#include "MyTools.h"
#include <iostream>

#define PI 3.1415926
#define E 2.7182818

using namespace osc;

__device__ vec3f getColor(const Interaction& isect) {
	vec3f color = isect.mat_mes.diffuse;
	if (isect.mat_mes.diffuseTextureID != -1) {
		float u = isect.texcoord.x;
		float v = isect.texcoord.y;
		vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
		color *= (vec3f)fromTexture;
	}
	return color;
}

 __device__ vec3f cal_diffuse_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    
    vec3f rnd;
    PRD prd4;
    prd4.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921,
        frame_id * 348593 % 43832 + iy * 324123 % 23415);
    rnd.x = prd4.random() * 2 - 1;
    prd4.random.init(frame_id * 972823 % 12971 + ix * 743782 % 82013,
        frame_id * 893022 % 28191 + iy * 918212 % 51321);
    rnd.y = prd4.random() * 2 - 1;
    prd4.random.init(frame_id * 383921 % 48839 + ix * 572131 % 47128,
        frame_id * 389291 % 29301 + iy * 716271 % 63291);
    rnd.z = prd4.random() * 2 - 1;
    vec3f wos = normalize(isect.geomNormal + normalize(rnd));
    *wo = wos;

    *pdf = 1 / (2 * float(PI));

    vec3f h = normalize(-wi + *wo);

    float cos_wo = dot(isect.geomNormal, *wo);
    float cos_wi = dot(isect.geomNormal, -wi);
    float cos_d = dot(h, -wi);

    float Fl = pow((1 - cos_wo), 5);
    float Fv = pow((1 - cos_wi), 5);

    float RR = 2 * isect.mat_mes.roughness * pow(cos_d, 2);

	//vec3f bsdf = diffuseColor / float(PI);
    //vec3f bsdf = (diffuseColor / float(PI)) * (float)(1 - 0.5 * Fl) * (float)(1 - 0.5 * Fv);
    //bsdf += (diffuseColor / float(PI)) * RR * (Fl + Fv + Fl * Fv * (RR - 1));

    vec3f F_D90 = vec3f(0.5 + 2.0 * isect.mat_mes.roughness) * diffuseColor;
    vec3f term1 = vec3f(1.0) + (F_D90 - vec3f(1.0)) * vec3f(pow(1 - abs(cos_wo), 5));
	vec3f term2 = vec3f(1.0) + (F_D90 - vec3f(1.0)) * vec3f(pow(1 - abs(cos_wi), 5));
    vec3f bsdf = (diffuseColor / (float)PI) * term1 * term2;

    //bsdf *= cos_wo;

    return bsdf;
}

__device__ vec3f cal_metal_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    

    /*
    vec3f out;
    out = wi - 2.0f * (vec3f)dot(wi, isect.geomNormal) * isect.geomNormal;
    out = normalize(out);
    vec3f out1 = cross(out, vec3f(1.0f));
    vec3f out2 = cross(out, out1);
    PRD prd;
    prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
    vec3f out3 = out + (prd.random() * 2 - 1) * out1 * isect.mat_mes.roughness + (prd.random() * 2 - 1) * out2 * isect.mat_mes.roughness;
    out3 = normalize(out3);
    if (dot(out3, isect.geomNormal) <= 0)
        out3 = out;
    *wo = out3;
    */
    
    vec3f rnd;
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
    vec3f wos = normalize(isect.geomNormal + normalize(rnd));

    vec3f out;
    out = wi - 2.0f * (vec3f)dot(wi, isect.geomNormal) * isect.geomNormal;
    out = normalize(out);

    vec3f Out = isect.mat_mes.roughness * wos + (1 - isect.mat_mes.roughness) * out;
    Out = normalize(Out);

    *wo = Out;

    //*pdf = 1 / ((2 * float(PI) - 1) * isect.mat_mes.roughness + 1);
    *pdf = 1 / (2 * float(PI));

    /*
    vec3f unit_direction = normalize(wi);
    double cos_theta = my_min(dot(-unit_direction, isect.geomNormal), 1.0);
    double etai_over_etat = 0;
    if (dot(wi, isect.realNormal) > 0) etai_over_etat = isect.mat_mes.ior;
    else etai_over_etat = 1.0f / isect.mat_mes.ior;
    vec3f specTerm = isect.mat_mes.diffuse + (vec3f(1.0f) - isect.mat_mes.diffuse) * powf(1.0f - cos_theta, 5.0f);

    vec3f color = isect.mat_mes.metallic * diffuseColor + (1 - isect.mat_mes.metallic) * diffuseColor;
    vec3f bsdf = color / (float)(PI / sqrt(1.0 / isect.mat_mes.roughness));
    if (isect.mat_mes.roughness < 1e-6)
        bsdf = color;

    float cos_wo = dot(isect.geomNormal, Out);
    float cos_wi = abs(dot(isect.geomNormal, normalize(wi)));
    bsdf = bsdf * ((1 - pow(isect.mat_mes.roughness, (float)0.1)) + pow(isect.mat_mes.roughness, (float)0.1) * cos_wo);
    */

    
    vec3f h = normalize(normalize(-wi) + normalize(*wo));
    double cos_wo = abs(dot(isect.geomNormal, *wo));
    double cos_wi = abs(dot(isect.geomNormal, normalize(wi)));
    double cos_h = abs(dot(h, isect.geomNormal));
    double cos_h2 = cos_h * cos_h;
    double sin_h2 = 1 - cos_h2;
    double alpha = (1 - isect.mat_mes.roughness)*0.002 + isect.mat_mes.roughness; 

    float D = alpha * alpha / (PI * pow(pow(cos_h, 2) * (alpha * alpha - 1) + 1, 1));

    double sin_wi = sqrt(1 - pow(cos_wi, 2));
    double tan_wi = sin_wi / cos_wi;
    double a = tan_wi * alpha;
    double A = (sqrt(1 + pow(a, 2)) - 1) / 2;

    float k = pow(alpha + 1, 2) / 8;
	float G1 = cos_wo / (cos_wo * (1 - k) + k);
	float G2 = cos_wi / (cos_wi * (1 - k) + k);
	float G = G1 * G2;
    //float G = 1.0 / (1 + A);

    double cos_d = abs(dot(h, normalize(wi)));
	vec3f F0 = (1 - isect.mat_mes.metallic) * vec3f(0.04) + isect.mat_mes.metallic * diffuseColor;
    vec3f F = schlick_vec3f(cos_d, F0);

    vec3f bsdf = D * G * F / (float)(4 * cos_wi * cos_wo);
    
    return bsdf;
}

__forceinline__ __device__ vec3f cal_specular_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    
    vec3f rnd;
    PRD prd2;
    prd2.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921,
        frame_id * 348593 % 43832 + iy * 324123 % 23415);
    rnd.x = prd2.random() * 2 - 1;
    prd2.random.init(frame_id * 972823 % 12971 + ix * 743782 % 82013,
        frame_id * 893022 % 28191 + iy * 918212 % 51321);
    rnd.y = prd2.random() * 2 - 1;
    prd2.random.init(frame_id * 383921 % 48839 + ix * 572131 % 47128,
        frame_id * 389291 % 29301 + iy * 716271 % 63291);
    rnd.z = prd2.random() * 2 - 1;
    vec3f wos = normalize(isect.geomNormal + normalize(rnd));

    vec3f out;
    out = wi - 2.0f * (vec3f)dot(wi, isect.geomNormal) * isect.geomNormal;
    out = normalize(out);

    vec3f Out = isect.mat_mes.roughness * wos + (1 - isect.mat_mes.roughness) * out;
    Out = normalize(Out);

    *wo = Out;
    
    *pdf = 1 / ((2 * float(PI) - 1) * isect.mat_mes.roughness + 1);
 
    vec3f h = normalize(normalize(-wi) + normalize(*wo));
    double cos_wo = dot(isect.geomNormal, *wo);
    double cos_h = abs(dot(h, isect.geomNormal));
    double cos_h2 = cos_h * cos_h;
    double sin_h2 = 1 - cos_h2;
    double tan_h2 = sin_h2 / cos_h2;
    if (sin_h2 < 0.0001)
        sin_h2 = 0.0001;
    double alpha = (1 - isect.mat_mes.roughness) * 0.002 + isect.mat_mes.roughness;

    //float D = alpha * alpha / (PI * (pow(cos_h, 2) * (alpha * alpha - 1) + 1));
    float D = (alpha * alpha) / (PI * pow((alpha * alpha * cos_h2 + sin_h2), 2));
    //float D = (alpha * alpha) / (PI * pow(cos_h * cos_h * (alpha * alpha -1), 2) + 0.0001);

    double cos_wi = abs(dot(isect.geomNormal, normalize(wi)));
    double sin_wi = sqrt(1 - pow(cos_wi, 2));
    double tan_wi = sin_wi / cos_wi;
    double a = tan_wi * sqrt(2.0) * pow(isect.mat_mes.roughness, 2);
    double A = (sqrt(1 + pow(a, 2)) - 1) / 2;

    float G = 1.0 / (1 + A);

    double cos_d = abs(dot(h, normalize(wi)));
    vec3f F0 = (1 - isect.mat_mes.metallic) * vec3f(0.04) + isect.mat_mes.metallic * isect.mat_mes.diffuse;
    vec3f F = schlick_vec3f(cos_d, F0);

    vec3f bsdf = D * G * F / (float)(4 * cos_wi * cos_wo);

    return bsdf;
}

__forceinline__ __device__ vec3f cal_mirror_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    vec3f bsdf = diffuseColor;
    vec3f out;
    out = wi - 2.0f * (vec3f)dot(wi, isect.geomNormal) * isect.geomNormal;
    *wo = normalize(out);
    *pdf = 1;
    return bsdf;
}

__device__ vec3f cal_dielectric_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    /*
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    vec3f bsdf = diffuseColor;
    *pdf = 1;
    */

    * pdf = 1;
    vec3f bsdf = isect.mat_mes.diffuse;

    //* pdf = 1 / ((2 * float(PI) - 1) * isect.mat_mes.roughness + 1);
    //vec3f bsdf = isect.mat_mes.diffuse / (float)(PI / sqrt(1.0 / isect.mat_mes.roughness));

    float etai_over_etat = 0;
    if (dot(wi, isect.realNormal) > 0) etai_over_etat = isect.mat_mes.ior;
    else etai_over_etat = 1.0f / isect.mat_mes.ior;
    vec3f unit_direction = normalize(wi);
    double cos_theta = my_min(dot(-unit_direction, isect.geomNormal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    if (etai_over_etat * sin_theta > 1.0f) { //全内反射
        *wo = reflect(unit_direction, isect.geomNormal, isect.mat_mes.roughness, ix, iy, frame_id);
        float cos_wo = abs(dot(isect.geomNormal, *wo));
        return bsdf/cos_wo;
        //return cal_specular_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    }

    double reflect_prob = schlick(cos_theta, etai_over_etat);//反射率
    PRD prd1;
    prd1.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
    if (prd1.random() < reflect_prob) {
        *pdf = reflect_prob;
        *wo = reflect(unit_direction, isect.geomNormal, isect.mat_mes.roughness, ix, iy, frame_id);
        float cos_wo = abs(dot(isect.geomNormal, *wo));
        return bsdf/cos_wo;
		//return cal_specular_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    }
    

    //*pdf = 1 - reflect_prob;
    vec3f perfecr_refract = refract(unit_direction, isect.geomNormal, isect.mat_mes.roughness, etai_over_etat, ix, iy, frame_id);


    vec3f rnd;
    prd1.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921,
        frame_id * 348593 % 43832 + iy * 324123 % 23415);
    rnd.x = prd1.random() * 2 - 1;
    prd1.random.init(frame_id * 972823 % 12971 + ix * 743782 % 82013,
        frame_id * 893022 % 28191 + iy * 918212 % 51321);
    rnd.y = prd1.random() * 2 - 1;
    prd1.random.init(frame_id * 383921 % 48839 + ix * 572131 % 47128,
        frame_id * 389291 % 29301 + iy * 716271 % 63291);
    rnd.z = prd1.random() * 2 - 1;
    vec3f wos2 = normalize(-isect.geomNormal + normalize(rnd));

    vec3f refractOut = normalize(isect.mat_mes.roughness * wos2 + (1 - isect.mat_mes.roughness) * perfecr_refract);

    *pdf = 1 - reflect_prob / ((2 * float(PI) - 1) * isect.mat_mes.roughness + 1);

    *wo = refractOut;
	float cos_wo = abs(dot(isect.geomNormal, refractOut));
    return bsdf/cos_wo;
}

 __device__ vec3f cal_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id)
{
    PRD prd3;
    vec3f result;

    float metallicBRDF = isect.mat_mes.metallic;
    float specularBSDF = (1.0f - isect.mat_mes.metallic) * isect.mat_mes.specTrans;
    float dielectricBRDF = (1.0f - isect.mat_mes.specTrans) * (1.0f - isect.mat_mes.metallic);

    float specularWeight = metallicBRDF + dielectricBRDF;
    float transmissionWeight = specularBSDF;
    float diffuseWeight = dielectricBRDF;

    float norm = 1.0f / (specularWeight + transmissionWeight + diffuseWeight);

    float pDiffuse = diffuseWeight * norm;
    float pSpecular = specularWeight * norm + pDiffuse;
    float pSpecTrans = 1;
    

    
    if (isect.mat_mes.mat_kind == DIELECTRIC)
    {
        //result = cal_dielectric_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    }
    else if (isect.mat_mes.mat_kind == DIFFUSE)
    {
        //result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    }
    else if (isect.mat_mes.mat_kind == METAL)
    {
        //result = cal_specular_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        /*
        prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
        float res = prd.random();
        if (res < isect.mat_mes.roughness)
            result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        else
            result = cal_mirror_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        */
    }
    else if (isect.mat_mes.mat_kind == TESTMAT) {

        prd3.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
        float rnd = prd3.random();

        if (rnd < pDiffuse)
        {
            result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        }
        else if (rnd < pSpecular)
        {
            result = cal_metal_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        }
        else if (rnd < pSpecTrans)
        {
            result = cal_dielectric_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        }
    }

    return result;
}