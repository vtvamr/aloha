#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include <owl/owl.h>
#include "owl/common/math/box.h"
#include "owl/common/math/random.h"

using namespace owl;
using namespace owl::common;

typedef owl::interval<float> range1f;

namespace maui {
  struct RayGen {
  };

  struct Cluster {
    // id of this cluster; unique *for this island*
    int id;

    // rank in *this island* that the cluster is on
    int rank;

    // domain bounds; those won't overlap
    box3f domain;
  };

  struct VolBrick {
    // cell range
    box3i cellRange;

    // voxel range
    box3i voxelRange;

    // space range
    box3f spaceRange;

    // value range
    range1f valueRange;

    // 3D texture
    cudaTextureObject_t texture;
  };

  struct ClusterGeom {
    Cluster *clusterBuffer;
  };

  struct TriangleGeom {
    unsigned clusterID;
    vec3f *vertexBuffer;
    vec3i *indexBuffer;
  };

  struct BrickGeom {
    VolBrick *brickBuffer;
  };

  struct LaunchParams {
    uint32_t *fbPointer;
    float    *fbDepth;
    float4   *accumBuffer;
    int       accumID;
    int       islandID;
    int       numIslands;
    OptixTraversableHandle world;
    OptixTraversableHandle clusters;
    OptixTraversableHandle volBricks;
    VolBrick *brickBuffer;
    float maxDepth;
    box3f domain;
    struct {
      cudaTextureObject_t texture;
      range1f             domain;
      float               opacityScale;
    } transferFunc;
    struct {
      vec3f org;
      vec3f dir_00;
      vec3f dir_du;
      vec3f dir_dv;
    } camera;
    struct {
      float dt;
      int   heatMapEnabled;
      float heatMapScale;
      int   spp;
    } render;
  };

}
