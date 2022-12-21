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

#include <float.h>
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>
#include "owl/owl.h"
#include "owl/common/math/box.h"
#include "FileMapping.h"

namespace maui {

  using namespace owl;
  using namespace owl::common;

  struct Volume {
    typedef std::shared_ptr<Volume> SP;

    Volume() = default;
    Volume(std::string fileName, const vec3i dims, int bpc)
      : fileName(fileName)
      , file(fileName)
      , dims(dims)
      , bpc(bpc)
    {
      // fp = fopen(fileName.c_str(),"rb");
    }

    const float* get(const vec3i first, const vec3i last,
                     interval<float> *valueRange = NULL)
    {
      // if (fp == NULL || bpc == 0)
      //   return NULL;

      if (dims == vec3i(0))
        return NULL;

      if (bpc != 1 && bpc != 2 && bpc != 4)
        return NULL;

      vec3i range = last-first;
      size_t numVoxels = range.x*size_t(range.y)*range.z;
      voxelBuffer.resize(numVoxels);

      // Read line by line and convert
      std::vector<char> readBuffer(range.x*bpc);

      size_t voxelOff = 0;
      interval<float> vr(FLT_MAX,-FLT_MAX);
      for (int z=first.z; z!=last.z; ++z) {
        for (int y=first.y; y!=last.y; ++y, voxelOff += range.x) {
          size_t firstVoxel = z * dims.x * size_t(dims.y) + size_t(y) * dims.x + first.x;
          file.seek(firstVoxel*bpc);
          file.read(readBuffer.data(),readBuffer.size());
          // fseek(fp,firstVoxel*bpc,SEEK_SET);
          // fread(readBuffer.data(),1,readBuffer.size(),fp);

          for (int ix=0; ix<range.x; ++ix) {
            float value = 0.f;
            if (bpc == 1)
              value = (unsigned char)(readBuffer[ix])/255.f;
            else if (bpc == 2) {
              char bytes[2] = {readBuffer[ix*2],readBuffer[ix*2+1]};
              value = (*((unsigned short*)bytes))/65535.f;
            } else if (bpc == 4) {
              char bytes[4] = {readBuffer[ix*4],readBuffer[ix*4+1],
                               readBuffer[ix*4+2],readBuffer[ix*4+3]};
              value = *((float*)bytes);
            }
            voxelBuffer[voxelOff+ix] = value;
            vr.extend(value);
          }
        }
      }

      if (valueRange != NULL) {
        valueRange->extend(vr);
      }

      return voxelBuffer.data();
    }

    const float* get(box3i cellRange, interval<float> *valueRange = NULL)
    {
      return get(cellRange.lower,cellRange.upper,valueRange);
    }

    std::vector<float> voxelBuffer;

    std::string fileName;
    MappedFile file;
    vec3i dims;
    int bpc;
  };

}
