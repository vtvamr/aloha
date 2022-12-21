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

#include <memory>
#include <string>
#include <vector>
#include "owl/owl.h"
#include "owl/common/math/vec.h"

namespace maui {

  using namespace owl;
  using namespace owl::common;
 
  struct Geometry {
    typedef std::shared_ptr<Geometry> SP;

    std::vector<vec3f> vertex;
    std::vector<vec3i> index;
  };

  struct Mesh {
    typedef std::shared_ptr<Mesh> SP;

    std::vector<Geometry::SP> geoms;

    static Mesh::SP loadOBJ(std::string fileName);
    static Mesh::SP loadMini(std::string fileName);
    static Mesh::SP load(std::string fileName);
  };

}