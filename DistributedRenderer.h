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

#include <string>
#include <vector>
#include <mpi.h>
#include <owl/owl.h>
#include "owl/common/math/vec.h"

#define REQUIRED =0
#define OPTIONAL {}

namespace maui {
  namespace mpi {

    struct DistributedRenderer {
      DistributedRenderer();
      virtual ~DistributedRenderer();

      void setCamera(const owl::vec3f &org,
                     const owl::vec3f &dir_00,
                     const owl::vec3f &dir_du,
                     const owl::vec3f &dir_dv);

      void resize(const owl::vec2i &newSize);

      void screenShot(const std::string &baseName);

      void setColorMap(const std::vector<owl::vec4f> &newCM);

      virtual void handleCameraEvent(const owl::vec3f &org,
                                     const owl::vec3f &dir_00,
                                     const owl::vec3f &dir_du,
                                     const owl::vec3f &dir_dv) REQUIRED;

      virtual void handleResizeEvent(const owl::vec2i &newSize) REQUIRED;

      virtual void handleScreenShotEvent(const std::string &baseName) OPTIONAL;

      virtual void handleColorMapEvent(const std::vector<owl::vec4f> &newCM) OPTIONAL;

      // Synchronize events across ranks
      void processEvents();

      struct HostInfo
      {
        std::string name;
        int worldRank;
      };

      struct {
        int rank = 0;
        int size = 0;
        std::vector<HostInfo> hosts;
      } commWorld;

      int gpuID = 0;
    };

  } // ::mpi
} // ::maui

