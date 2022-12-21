#include <iostream>
#include <sstream>

#include "deviceCode.h"
#include "owl/common/math/random.h"

namespace maui
{
    extern "C" __constant__ LaunchParams optixLaunchParams;


    typedef owl::common::LCG<4> Random;
    
#define DEBUGGING 1
#define DBG_X (getLaunchDims().x/2)
#define DBG_Y (getLaunchDims().y/2)

    __device__ inline bool debug()
    {
#if DEBUGGING
        return (getLaunchIndex().x == DBG_X && getLaunchIndex().y == DBG_Y);
#else
        return false;
#endif
    }

    inline  __device__
    vec3f backGroundColor()
    {
        const vec2i pixelID = owl::getLaunchIndex();
        const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
        const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
        return c;
    }

    inline  __device__ vec4f over(const vec4f &A, const vec4f &B)
    {
        return A + (1.f-A.w)*B;
    }

    inline __device__
    bool intersect(const Ray &ray,
                   const box3f &box,
                   float &t0,
                   float &t1)
    {
        vec3f lo = (box.lower - ray.origin) / ray.direction;
        vec3f hi = (box.upper - ray.origin) / ray.direction;
        
        vec3f nr = min(lo,hi);
        vec3f fr = max(lo,hi);

        t0 = max(ray.tmin,reduce_max(nr));
        t1 = min(ray.tmax,reduce_min(fr));

        return t0 < t1;
    }

    inline __device__ vec3f randomColor(unsigned idx)
    {
      unsigned int r = (unsigned int)(idx*13*17 + 0x234235);
      unsigned int g = (unsigned int)(idx*7*3*5 + 0x773477);
      unsigned int b = (unsigned int)(idx*11*19 + 0x223766);
      return vec3f((r&255)/255.f,
                   (g&255)/255.f,
                   (b&255)/255.f);
    }

    inline __device__ vec3f cosineSampleHemisphere(float u1, float u2)
    {
        const float PI(3.14159265358979323846264338328e+00);

        float r = sqrtf(u1);
        float theta = PI*2.f*u2;
        float x = r * cosf(theta);
        float y = r * sinf(theta);
        float z = sqrtf(1.f-u1);
        return vec3f(x,y,z);
    }

    inline __device__ void sampleLambertian(const vec3f &n, const vec3f &wo, vec3f &wi, float &pdf, Random& rnd)
    {
        const float invPI(3.18309886183790691216444201928e-01);

        vec3f w = n;
        vec3f v = fabsf(w.x)>fabsf(w.y) ? normalize(vec3f(-w.z,0.f,w.x))
                                        : normalize(vec3f(0.f,w.z,-w.y));
        vec3f u = cross(v,w);
        vec3f sp = cosineSampleHemisphere(rnd(),rnd());
        wi = normalize(sp.x*u+sp.y*v+sp.z*w);
        pdf = dot(n,wi) * invPI;
    }

    inline  __device__ Ray generateRay(const vec2f screen)
    {
        auto &lp = optixLaunchParams;
        vec3f org = lp.camera.org;
        vec3f dir
          = lp.camera.dir_00
          + screen.u * lp.camera.dir_du
          + screen.v * lp.camera.dir_dv;
        dir = normalize(dir);
        if (fabs(dir.x) < 1e-5f) dir.x = 1e-5f;
        if (fabs(dir.y) < 1e-5f) dir.y = 1e-5f;
        if (fabs(dir.z) < 1e-5f) dir.z = 1e-5f;
        return Ray(org,dir,0.f,1e10f);
    }
 
    // ==================================================================
    // Clusters user geometry
    // ==================================================================

    struct ClusterPRD {
      Cluster cluster;
      float tnear;
      float tfar;
    };

    OPTIX_BOUNDS_PROGRAM(ClusterBounds)(const void* geomData,
                                        box3f& result,
                                        int leafID)
    {
      const ClusterGeom &self = *(const ClusterGeom *)geomData;
      result = self.clusterBuffer[leafID].domain;
    }

    OPTIX_INTERSECT_PROGRAM(ClusterIsect)()
    {
      const ClusterGeom &self = owl::getProgramData<ClusterGeom>();
      int primID = optixGetPrimitiveIndex();
      owl::Ray ray(optixGetObjectRayOrigin(),
                   optixGetObjectRayDirection(),
                   optixGetRayTmin(),
                   optixGetRayTmax());
      float t0 = ray.tmin, t1 = ray.tmax;
      const box3f &box = self.clusterBuffer[primID].domain;
      const vec3f t_lo = (box.lower - ray.origin) / ray.direction;
      const vec3f t_hi = (box.upper - ray.origin) / ray.direction;

      const vec3f t_nr = min(t_lo,t_hi);
      const vec3f t_fr = max(t_lo,t_hi);

      t0 = max(ray.tmin,reduce_max(t_nr));
      t1 = min(ray.tmax,reduce_min(t_fr));

      if (t0 < t1) {
        if (t0 > ray.tmin && optixReportIntersection(t0, 0)) {
          ClusterPRD& prd = owl::getPRD<ClusterPRD>();
          prd.cluster = self.clusterBuffer[primID];
          prd.tnear = t1;
          prd.tfar = t1;
        } else if (t1 > ray.tmin && optixReportIntersection(t1, 0)) {
          ClusterPRD& prd = owl::getPRD<ClusterPRD>();
          prd.cluster = self.clusterBuffer[primID];
          prd.tnear = 0.f;
          prd.tfar = t1;
        }
      }
    }

    OPTIX_CLOSEST_HIT_PROGRAM(ClusterCH)()
    {
    }

    // ==================================================================
    // Triangle model
    // ==================================================================

    struct ModelPRD {
        unsigned clusterID;
        float t_hit;
        vec3f gn;
        int primID;
    };

    OPTIX_CLOSEST_HIT_PROGRAM(ModelCH)()
    {
        ModelPRD& prd = owl::getPRD<ModelPRD>();
        const TriangleGeom& self = owl::getProgramData<TriangleGeom>();
        prd.clusterID = self.clusterID;
        prd.t_hit = optixGetRayTmax();
        prd.primID = optixGetPrimitiveIndex();
        const vec3i index  = self.indexBuffer[prd.primID];
        const vec3f& v1     = self.vertexBuffer[index.x];
        const vec3f& v2     = self.vertexBuffer[index.y];
        const vec3f& v3     = self.vertexBuffer[index.z];
        prd.gn            = normalize(cross(v2 - v1, v3 - v1));
    }

    // ==================================================================
    // Volume
    // ==================================================================

    struct VolumePRD {
        float tnear, tfar;
        int primID;
    };

    OPTIX_BOUNDS_PROGRAM(VolBrickBounds)(const void* geomData,
                                         box3f& result,
                                         int leafID)
    {
        const BrickGeom &self = *(const BrickGeom *)geomData;
        result = self.brickBuffer[leafID].spaceRange;
    }

    OPTIX_INTERSECT_PROGRAM(VolBrickIsect)()
    {
        const BrickGeom &self = owl::getProgramData<BrickGeom>();
        int primID = optixGetPrimitiveIndex();
        owl::Ray ray(optixGetObjectRayOrigin(),
                optixGetObjectRayDirection(),
                optixGetRayTmin(),
                optixGetRayTmax());
        float t0 = ray.tmin, t1 = ray.tmax;
        const box3f &box = self.brickBuffer[primID].spaceRange;
        const vec3f t_lo = (box.lower - ray.origin) / ray.direction;
        const vec3f t_hi = (box.upper - ray.origin) / ray.direction;

        const vec3f t_nr = min(t_lo,t_hi);
        const vec3f t_fr = max(t_lo,t_hi);

        t0 = max(ray.tmin,reduce_max(t_nr));
        t1 = min(ray.tmax,reduce_min(t_fr));

        if (t0 < t1) {
            if (t0 > ray.tmin && optixReportIntersection(t0, 0)) {
                VolumePRD& prd = owl::getPRD<VolumePRD>();
                prd.tnear = t0;
                prd.tfar = t1;
                prd.primID = primID;
            } else if (t1 > ray.tmin && optixReportIntersection(t1, 0)) {
                VolumePRD& prd = owl::getPRD<VolumePRD>();
                prd.tnear = 0.f;
                prd.tfar = t1;
                prd.primID = primID;
            }
        }
    }

    OPTIX_CLOSEST_HIT_PROGRAM(VolBrickCH)()
    {
    }

    inline __device__
    float firstSampleT(const range1f &rayInterval,
                       const float dt,
                       const float ils_t0)
    {
        float numSegsf = floor((rayInterval.lower - dt*ils_t0)/dt);
        float t = dt * (ils_t0 + numSegsf);
        if (t < rayInterval.lower) t += dt;
        return t;
    }

    inline __device__
    void integrateBrick(const Ray &ray, const int brickID, const float ils_t0, vec4f &color)
    {
        auto &lp = optixLaunchParams;
        const VolBrick &brick = lp.brickBuffer[brickID];
        const box3f bounds = brick.spaceRange;
        const box3i voxelRange = brick.voxelRange;
        //vec3f cd = randomColor((unsigned)brickID);
        vec3f cd = 1.f;
        float t0, t1;
        if (!intersect(ray,bounds,t0,t1)) {
            return;
        }
        const float dt = lp.render.dt;
        for (float t = firstSampleT({t0,t1},dt,ils_t0); t < t1 && color.w < .99f; t += dt) {
            vec3f pos = ray.origin+t*ray.direction;
            vec3f tc = pos;
            tc -= voxelRange.lower;
            tc /= voxelRange.upper-voxelRange.lower;
            tc += .5f/vec3f(voxelRange.upper-voxelRange.lower);

            // const vec3f delta = lp.render.gradientDelta / (cellRange.upper-cellRange.lower);;
            const vec3f delta = .5f / vec3f(voxelRange.upper-voxelRange.lower);;
            vec3f gradient = vec3f(+tex3D<float>(brick.texture,tc.x+delta.x,tc.y,tc.z)
                                   -tex3D<float>(brick.texture,tc.x-delta.x,tc.y,tc.z),
                                   +tex3D<float>(brick.texture,tc.x,tc.y+delta.y,tc.z)
                                   -tex3D<float>(brick.texture,tc.x,tc.y-delta.y,tc.z),
                                   +tex3D<float>(brick.texture,tc.x,tc.y,tc.z+delta.z)
                                   -tex3D<float>(brick.texture,tc.x,tc.y,tc.z-delta.z));
            vec3f N = gradient / (length(gradient + 1e-4f));

            //vec4f sample = vec4f(tc,.01f);
            float value = tex3D<float>(brick.texture,tc.x,tc.y,tc.z);
            float4 xf = tex1D<float4>(lp.transferFunc.texture,value);
            //xf.w *= lp.transferFunc.opacityScale;
            vec4f sample = xf;
            vec3f rgb(color);
            rgb += (1.f-color.w) * sample.w * vec3f(sample)
              * cd * (.1f+.9f*fabsf(dot(N,ray.direction)));
            color.x = rgb.x; color.y = rgb.y; color.z = rgb.z;
            color.w += (1.f-color.w)*sample.w;
        }
    }

    inline __device__ vec3f hue_to_rgb(float hue)
    {
        float s = saturate( hue ) * 6.0f;
        float r = saturate( fabsf(s - 3.f) - 1.0f );
        float g = saturate( 2.0f - fabsf(s - 2.0f) );
        float b = saturate( 2.0f - fabsf(s - 4.0f) );
        return vec3f(r, g, b); 
    }
      
    inline __device__ vec3f temperature_to_rgb(float t)
    {
        float K = 4.0f / 6.0f;
        float h = K - K * t;
        float v = .5f + 0.5f * t;    return v * hue_to_rgb(h);
    }
      
                                      
    inline __device__
    vec3f heatMap(float t)
    {
#if 1
        return temperature_to_rgb(t);
#else
        if (t < .25f) return lerp(vec3f(0.f,1.f,0.f),vec3f(0.f,1.f,1.f),(t-0.f)/.25f);
        if (t < .5f)  return lerp(vec3f(0.f,1.f,1.f),vec3f(0.f,0.f,1.f),(t-.25f)/.25f);
        if (t < .75f) return lerp(vec3f(0.f,0.f,1.f),vec3f(1.f,1.f,1.f),(t-.5f)/.25f);
        if (t < 1.f)  return lerp(vec3f(1.f,1.f,1.f),vec3f(1.f,0.f,0.f),(t-.75f)/.25f);
        return vec3f(1.f,0.f,0.f);
#endif
    }

    inline int __device__ iDivUp(int a, int b)
    {
        return (a + b - 1) / b;
    }
 
    // ------------------------------------------------------------------
    // primary visibility renderer
    // ------------------------------------------------------------------

    OPTIX_RAYGEN_PROGRAM(renderFrame)()
    {
        auto& lp = optixLaunchParams;
#if 1
        const int spp = lp.render.spp;
        const vec2i threadIdx = owl::getLaunchIndex();
        const vec2i blockDim = { 16, 16 };
        const vec2i blockIdx = threadIdx/blockDim;
        const vec2i gridSize = { iDivUp(threadIdx.x,owl::getLaunchDims().x),
                                 iDivUp(threadIdx.y,owl::getLaunchDims().y) };

        vec4f bgColor = vec4f(backGroundColor(),1.f);
        int pixelID = threadIdx.x + owl::getLaunchDims().x*threadIdx.y;
        int blockID = blockIdx.x + gridSize.x*blockIdx.y;
        Random random(pixelID,lp.accumID);

        uint64_t clock_begin = clock64();

        vec4f accumColor = 0.f;

        float z = 1.f;
        if (blockID%lp.numIslands==lp.islandID) {
          for (int sampleID=0;sampleID<spp;sampleID++) {
            vec4f color = 0.f;

            Ray ray = generateRay(vec2f(threadIdx)+vec2f(.5f));

            // Tri-mesh
            ModelPRD prd{unsigned(-1),-1.f,vec3f(-1),-1};
            owl::traceRay(lp.world,ray,prd,
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT);
            if (prd.primID >= 0) {
              vec3f cd = randomColor((unsigned)prd.clusterID);
              color = vec4f(vec3f(.2f)+cd*vec3f(max(0.f,dot(-ray.direction,prd.gn))),1.f);
              z = prd.t_hit/lp.maxDepth;
            }
           
            // Volumes
            VolumePRD vprd{-1.f,-1.f,-1};
            do {
              vprd = {-1.f,-1.f,-1};
              owl::traceRay(lp.volBricks,ray,vprd,
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT);

              if (vprd.primID >= 0) {
                vprd.tnear += ray.tmin;
                const float ils_t0 = random();
                integrateBrick(ray,vprd.primID,ils_t0,color);
                z = min(z,vprd.tnear/lp.maxDepth);
                ray.tmin = vprd.tfar+1e-20f;
              }
            } while (vprd.primID >= 0 && color.w < .99f);

            //color = over(color,bgColor);
            accumColor += color;
          }
        }
        lp.fbDepth[pixelID]   = z;

        uint64_t clock_end = clock64();
        if (lp.render.heatMapEnabled > 0.f) {
            float t = (clock_end-clock_begin)*(lp.render.heatMapScale/spp);
            accumColor = over(vec4f(heatMap(t),.5f),accumColor);
        }

        if (lp.accumID > 0)
            accumColor += vec4f(lp.accumBuffer[pixelID]);
        lp.accumBuffer[pixelID] = accumColor;
        accumColor *= (1.f/(lp.accumID+1));
        
        bool crossHairs = (owl::getLaunchIndex().x == owl::getLaunchDims().x/2
                           ||
                           owl::getLaunchIndex().y == owl::getLaunchDims().y/2
                           );
        if (crossHairs) accumColor = vec4f(1.f) - accumColor;
        
        lp.fbPointer[pixelID] = make_rgba(accumColor*(1.f/spp));
#endif
        // const vec2i threadIdx = owl::getLaunchIndex();
        // int pixelID = threadIdx.x + owl::getLaunchDims().x*threadIdx.y;
        // lp.fbPointer[pixelID] = make_rgba(vec3f(1.f));
    }
}
