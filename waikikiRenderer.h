// ======================================================================== //
// Copyright 2022-2022 Stefan Zellmann                                      //
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

#pragma once

#include <IceTMPI.h>
#include "qtOWL/ColorMaps.h"
#include "DistributedRenderer.h"
#include "deviceCode.h"

namespace maui {

  struct Partitioner;

  struct Renderer : mpi::DistributedRenderer {

    enum class Mode { TriMesh, Volume, };

    Renderer(int commRank,
             int commSize,
             int numIslands,
             const std::string& inFileName);

   ~Renderer();

    void render(uint32_t *fbPointer);

    void handleCameraEvent(const vec3f &org,
                           const vec3f &dir_00,
                           const vec3f &dir_du,
                           const vec3f &dir_dv);

    void handleResizeEvent(const vec2i &newSize);

    void handleScreenShotEvent(const std::string &baseName);

    void handleColorMapEvent(const std::vector<vec4f> &newCM);

    Mode mode;

    box3f modelBounds;

    OWLContext owl;
    OWLModule  module;
    OWLParams  lp;
    OWLRayGen  rayGen;

    OWLGeomType clusterGeomType;
    OWLGroup tlasClusters;
    OWLGroup blasClusters;

    struct {
      OWLGeomType triangleGeomType;
      std::vector<OWLGroup> modelGroups;
      OWLGroup tlasGroup;
    } asTriMesh;

    struct {
      box3i cellRange;
      box3i voxelRange;
      OWLGeomType brickGeomType;
      OWLGroup tlasGroup;
      OWLGroup blasGroup;
      std::vector<vec4f> colorMap;
      OWLBuffer colorMapBuffer { 0 };
      cudaArray_t colorMapArray { 0 };
      cudaTextureObject_t colorMapTexture { 0 };
    } asVolume;

    std::shared_ptr<Partitioner> partitioner;

    // We store a pointer to the FB so we can take screen shots
    uint32_t *fbPointer = nullptr;
    OWLBuffer accumBuffer { 0 };
    int accumID { 0 };
    OWLBuffer depthBuffer { 0 };

    struct {
      int rank { 0 };
      int size { 0 };
    } commWorld;

    struct {
      MPI_Comm comm;
      int rank { 0 };
      int size { 0 };
    } commIsland;

    int islandID { 0 };
    int numIslands { 0 };
    int displayRank { 0 };

    void resetAccum() { accumID = 0; }
    
    OWLBuffer colorMapBuffer { 0 };
    vec2i      fbSize { 1 };
    
    static int   spp;
    static bool  heatMapEnabled;
    static float heatMapScale;

    IceTCommunicator icetComm;
    IceTContext icetCtx = NULL;
    IceTImage icetImg;
  };
  
} // ::maui
