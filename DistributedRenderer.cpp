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

#include <string.h>
#ifndef ALOHA_CPU
#include <cuda_runtime.h>
#endif
#include "DistributedRenderer.h"

namespace maui {
  namespace mpi {

    struct Event {
      enum class Type { Camera, Resize, ScreenShot, ColorMap, };
      Type type;
      std::vector<char> bytes;
    };

    std::vector<Event> events;

    DistributedRenderer::DistributedRenderer()
    {
      MPI_Comm_rank(MPI_COMM_WORLD,&commWorld.rank);
      MPI_Comm_size(MPI_COMM_WORLD,&commWorld.size);

#ifndef ALOHA_CPU
      commWorld.hosts.resize(commWorld.size);
      typedef char ProcName[MPI_MAX_PROCESSOR_NAME];
      ProcName* namesOut = new ProcName[commWorld.size];
      int* lensOut = new int[commWorld.size];
      int* ranksOut = new int[commWorld.size];
      for (size_t i=0; i<commWorld.size; ++i) {
        MPI_Get_processor_name(namesOut[i],lensOut+i);
        ranksOut[i] = commWorld.rank;
      }
      ProcName* namesIn = new ProcName[commWorld.size];
      int* lensIn = new int[commWorld.size];
      int* ranksIn = new int[commWorld.size];
      MPI_Alltoall(namesOut,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,namesIn,
                   MPI_MAX_PROCESSOR_NAME,MPI_CHAR,MPI_COMM_WORLD);
      MPI_Alltoall(lensOut,1,MPI_INT,lensIn,1,MPI_INT,MPI_COMM_WORLD);
      MPI_Alltoall(ranksOut,1,MPI_INT,ranksIn,1,MPI_INT,MPI_COMM_WORLD);

      int numGPUs=0;
      cudaGetDeviceCount(&numGPUs);
      int myGPU=0;
      std::string myName(namesIn[commWorld.rank],lensIn[commWorld.rank]);
      for (size_t i=0; i<commWorld.size; ++i) {
        auto& host = commWorld.hosts[i];
        host.worldRank = ranksIn[i];
        host.name = std::string(namesIn[i],lensIn[i]);
        if (host.name == myName && host.worldRank < commWorld.rank) {
          myGPU++;
          myGPU %= numGPUs;
        }
      }

      delete[] namesOut;
      delete[] lensOut;
      delete[] ranksOut;
      delete[] namesIn;
      delete[] lensIn;
      delete[] ranksIn;

      std::stringstream s;
      s << "Rank: " << commWorld.rank << ", host name: "
        << myName << ", GPU: " << myGPU << '\n';
      std::cout << s.str();

      gpuID = myGPU;

      cudaSetDevice(gpuID);
#endif
    }

    DistributedRenderer::~DistributedRenderer()
    {
    }

    void DistributedRenderer::setCamera(const owl::vec3f &org,
                                        const owl::vec3f &dir_00,
                                        const owl::vec3f &dir_du,
                                        const owl::vec3f &dir_dv)
    {
      size_t off=0;
      Event ev;
      ev.type = Event::Type::Camera;
      ev.bytes.resize(4*sizeof(owl::vec3f));
      memcpy(ev.bytes.data()+off,&org,sizeof(org));
      off += sizeof(org);
      memcpy(ev.bytes.data()+off,&dir_00,sizeof(dir_00));
      off += sizeof(dir_00);
      memcpy(ev.bytes.data()+off,&dir_du,sizeof(dir_du));
      off += sizeof(dir_du);
      memcpy(ev.bytes.data()+off,&dir_dv,sizeof(dir_dv));
      off += sizeof(dir_dv);
      events.push_back(ev);
    }

    void DistributedRenderer::resize(const owl::vec2i &newSize)
    {
      Event ev;
      ev.type = Event::Type::Resize;
      ev.bytes.resize(sizeof(owl::vec2i));
      memcpy(ev.bytes.data(),&newSize,sizeof(newSize));
      events.push_back(ev);
    }

    void DistributedRenderer::screenShot(const std::string &baseName)
    {
      Event ev;
      ev.type = Event::Type::ScreenShot;
      ev.bytes = std::vector<char>(baseName.begin(),baseName.end());
      events.push_back(ev);
    }

    void DistributedRenderer::setColorMap(const std::vector<owl::vec4f> &newCM)
    {
      Event ev;
      ev.type = Event::Type::ColorMap;
      ev.bytes = std::vector<char>(newCM.size()*sizeof(owl::vec4f));
      memcpy(ev.bytes.data(),(const char *)newCM.data(),ev.bytes.size());
      events.push_back(ev);
    }

    void DistributedRenderer::processEvents()
    {
      if (commWorld.size==0)
        return;

      std::vector<int> numEventsOut(commWorld.size,(int)events.size());
      std::vector<int> numEventsIn(commWorld.size);

      MPI_Alltoall(numEventsOut.data(),1,MPI_INT,numEventsIn.data(),1,MPI_INT,
                   MPI_COMM_WORLD);

      for (size_t r=0; r<commWorld.size; ++r) {
        int numEvents = numEventsIn[r];

        for (int i=0; i<numEvents; ++i) {
          Event ev;
          uint64_t payloadSize;

          if (r == commWorld.rank) {
            ev = events[i];
            payloadSize = ev.bytes.size();
          }

          MPI_Bcast(&ev.type,sizeof(ev.type),MPI_BYTE,r,MPI_COMM_WORLD);
          MPI_Bcast(&payloadSize,1,MPI_UINT64_T,r,MPI_COMM_WORLD);

          if (payloadSize > 0) {
            std::vector<char> bytes(payloadSize);

            if (r == commWorld.rank) {
              memcpy(bytes.data(),ev.bytes.data(),ev.bytes.size());
            }

            MPI_Bcast(bytes.data(),bytes.size(),MPI_BYTE,r,MPI_COMM_WORLD);

            if (r != commWorld.rank) {
              ev.bytes.resize(payloadSize);
              memcpy(ev.bytes.data(),bytes.data(),bytes.size());
            }
          }

          if (ev.type == Event::Type::Camera) {
            size_t off=0;
            owl::vec3f org, dir_00, dir_du, dir_dv;
            memcpy(&org,ev.bytes.data()+off,sizeof(org));
            off += sizeof(org);
            memcpy(&dir_00,ev.bytes.data()+off,sizeof(dir_00));
            off += sizeof(dir_00);
            memcpy(&dir_du,ev.bytes.data()+off,sizeof(dir_du));
            off += sizeof(dir_du);
            memcpy(&dir_dv,ev.bytes.data()+off,sizeof(dir_dv));
            off += sizeof(dir_dv);

            handleCameraEvent(org,dir_00,dir_du,dir_dv);
          } else if (ev.type == Event::Type::Resize) {
            owl::vec2i newSize;
            memcpy(&newSize,ev.bytes.data(),sizeof(newSize));
            handleResizeEvent(newSize);
          } else if (ev.type == Event::Type::ScreenShot) {
            std::string baseName(ev.bytes.begin(),ev.bytes.end());
            handleScreenShotEvent(baseName);
          } else if (ev.type == Event::Type::ColorMap) {
            std::vector<owl::vec4f> newCM(ev.bytes.size()/sizeof(owl::vec4f));
            memcpy((char *)newCM.data(),ev.bytes.data(),ev.bytes.size());
            handleColorMapEvent(newCM);
          }
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);

      events.clear();
    }
  } // ::mpi
} // ::maui

