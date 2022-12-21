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

#include "Headless.h"

#ifndef ALOHA_CPU
#include <cuda_runtime.h>
#endif

using namespace owl;

namespace maui {

  Headless::Headless(const std::string &title, const vec2i &initWindowSize)
    : fbSize(initWindowSize)
    , title(title)
  {
    if (initWindowSize == vec2i(0,0)) {
      fbSize = vec2i(1024,1024);
      //fbSize = vec2i(1280,540);
    }

#ifdef ALOHA_CPU
    fbPointer = new uint32_t[fbSize.x*fbSize.y];
#else
    cudaMalloc(&fbPointer,fbSize.x*fbSize.y*sizeof(uint32_t));
#endif
  }

  Headless::~Headless()
  {
#ifdef ALOHA_CPU
    delete[] fbPointer;
#else
    cudaFree(fbPointer);
#endif
  }

  void Headless::run()
  {
    while (1) {
      render();
    }
  }

  void Headless::resize(const owl::vec2i &newSize)
  {
    if (newSize==fbSize)
      return;

    fbSize = newSize;

#ifdef ALOHA_CPU
    delete[] fbPointer;
    fbPointer = new uint32_t[fbSize.x*fbSize.y];
#else
    cudaFree(fbPointer);
    cudaMalloc(&fbPointer,fbSize.x*fbSize.y*sizeof(uint32_t));
#endif
  }

  void Headless::setTitle(const std::string &s)
  {
    title = s;
  }

  void Headless::setWorldScale(const float worldScale)
  {
    camera.motionSpeed = worldScale / sqrtf(3.f);
  }

} // ::maui
