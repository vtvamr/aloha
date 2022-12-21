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

#include <algorithm>
#include <sstream>
#include <cassert>
#include <cstring>
#include <fstream>
#include <IceT.h>
#include <IceTMPI.h>
#include "brick.h"
#include "mesh.h"
#include "waikikiRenderer.h"
#include "qtOWL/ColorMaps.h"
#include "deviceCode.h"
#ifdef ALOHA_CPU
#include <owl/owl_ext.h>
#endif
#include "owl/common/math/random.h"
#include <random>
#include "qtOWL/stb/stb_image_write.h"
#include <cuda_runtime.h>

extern "C" char embedded_deviceCode[];

namespace maui {

  bool  Renderer::heatMapEnabled = false;
  float Renderer::heatMapScale = 1e-5f;
  int   Renderer::spp = 1;
  
  OWLVarDecl rayGenVars[]
  = {
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl clusterGeomVars[]
  = {
     { "clusterBuffer",  OWL_BUFPTR, OWL_OFFSETOF(ClusterGeom,clusterBuffer)},
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl triangleGeomVars[]
  = {
     { "clusterID",  OWL_UINT, OWL_OFFSETOF(TriangleGeom,clusterID)},
     { "indexBuffer",  OWL_BUFPTR, OWL_OFFSETOF(TriangleGeom,indexBuffer)},
     { "vertexBuffer", OWL_BUFPTR, OWL_OFFSETOF(TriangleGeom,vertexBuffer)},
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl brickGeomVars[]
  = {
     { "brickBuffer",  OWL_BUFPTR, OWL_OFFSETOF(BrickGeom,brickBuffer)},
     { nullptr /* sentinel to mark end of list */ }
  };

  OWLVarDecl launchParamsVars[]
  = {
     { "fbPointer",   OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams,fbPointer) },
     { "fbDepth",   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,fbDepth) },
     { "accumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,accumBuffer) },
     { "accumID",   OWL_INT, OWL_OFFSETOF(LaunchParams,accumID) },
     { "islandID",   OWL_INT, OWL_OFFSETOF(LaunchParams,islandID) },
     { "numIslands",   OWL_INT, OWL_OFFSETOF(LaunchParams,numIslands) },
     { "world",    OWL_GROUP,  OWL_OFFSETOF(LaunchParams,world)},
     { "clusters",    OWL_GROUP,  OWL_OFFSETOF(LaunchParams,clusters)},
     { "volBricks",    OWL_GROUP,  OWL_OFFSETOF(LaunchParams,volBricks)},
     { "brickBuffer",    OWL_BUFPTR,  OWL_OFFSETOF(LaunchParams,brickBuffer)},
     { "maxDepth",    OWL_FLOAT,  OWL_OFFSETOF(LaunchParams,maxDepth)},
     { "domain.lower", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,domain.lower) },
     { "domain.upper", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,domain.upper) },
     // xf data
     { "transferFunc.domain",OWL_FLOAT2, OWL_OFFSETOF(LaunchParams,transferFunc.domain) },
     { "transferFunc.texture",   OWL_USER_TYPE(cudaTextureObject_t),OWL_OFFSETOF(LaunchParams,transferFunc.texture) },
     { "transferFunc.opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,transferFunc.opacityScale) },
     // render settings
     { "render.dt",           OWL_FLOAT,   OWL_OFFSETOF(LaunchParams,render.dt) },
     { "render.spp",           OWL_INT,   OWL_OFFSETOF(LaunchParams,render.spp) },
     { "render.heatMapEnabled", OWL_INT, OWL_OFFSETOF(LaunchParams,render.heatMapEnabled) },
     { "render.heatMapScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,render.heatMapScale) },
     // camera settings
     { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.org) },
     { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_00) },
     { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_du) },
     { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.dir_dv) },
     { nullptr /* sentinel to mark end of list */ }
  };
  
  inline std::string getExt(const std::string &fileName)
  {
    int pos = fileName.rfind('.');
    if (pos == fileName.npos)
      return "";
    return fileName.substr(pos);
  }

  inline std::string myPretty(size_t n)
  {
    if (n < 1024) return "  "+prettyNumber(n);

    return "  "+prettyNumber(n)+"\t("+std::to_string(n)+")";
  }

  inline int div_up(int a, int b)
  {
    return (a + b - 1) / b;
  }

  // ==================================================================
  // Partition clusters to ranks
  // ==================================================================

  struct Partitioner
  {
    Partitioner(std::vector<Cluster> &clusters, int numRanks)
      : input(clusters)
      , numRanks(numRanks)
    {
      compositeOrder.resize(numRanks);
      std::iota(compositeOrder.begin(),compositeOrder.end(),0);
      perRank.resize(numRanks);
    }

    void partitionRoundRobin()
    {
      int numClustersPerRank = div_up(input.size(),numRanks);
      for (size_t i=0; i<input.size(); ++i)
      {
        int clusterID = i;
        int rankID = i/numClustersPerRank;
        input[i].id = clusterID;
        input[i].rank = rankID;
        perRank[rankID].push_back(clusterID);
      }
    }

    void partitionKD()
    {
      // Assign same number of clusters per rank; use split-middle heuristic
      int numClustersPerRank = input.size()/numRanks;

      struct Clusters {
        box3f bounds;
        std::vector<int> clusterIDs;
        int kdNodeID;
      };

      Clusters all;
      all.bounds = { vec3f(1e30f), vec3f(-1e30f) };
      for (size_t i=0; i<input.size(); ++i)
      {
        all.bounds.extend(input[i].domain);
        all.clusterIDs.push_back(input[i].id);
        all.kdNodeID = 0;
      }

      std::vector<Clusters> clusters;
      std::vector<Clusters> assignedClusters;

      clusters.push_back(all);

      kdTree.emplace_back(KDNode{0,0.f,INT_MAX,INT_MAX});

      while (!clusters.empty()) {
        unsigned clusterToPick = 0;

        // Pick cluster with most clusters
        int maxNumIDs = 0;
        for (unsigned i=0; i<clusters.size(); ++i)
        {
          if (clusters[i].clusterIDs.size() > maxNumIDs) {
            clusterToPick = i;
            maxNumIDs = clusters[i].clusterIDs.size();
          }
        }

        Clusters cs = clusters[clusterToPick];
        clusters.erase(clusters.begin()+clusterToPick);

        if (cs.clusterIDs.size() <= numClustersPerRank
         || assignedClusters.size() == numRanks-1) {
          int rankID = assignedClusters.size();
          kdTree[cs.kdNodeID].child1 = ~rankID;
          kdTree[cs.kdNodeID].child2 = ~rankID;
          assignedClusters.push_back(cs);

          if (assignedClusters.size() == numRanks) {
            // Bump all remaining clusters into the last one
            Clusters& last = assignedClusters.back();
            while (!clusters.empty()) {
              Clusters cs = clusters[0];
              for (auto id : cs.clusterIDs) {
                last.clusterIDs.push_back(id);
              }
              clusters.erase(clusters.begin());
            }
            break;
          }
          else
            continue;
        }

        int splitAxis = 0;
        if (cs.bounds.size()[1]>cs.bounds.size()[0]
         && cs.bounds.size()[1]>=cs.bounds.size()[2]) {
          splitAxis = 1;
        } else if (cs.bounds.size()[2]>cs.bounds.size()[0]
                && cs.bounds.size()[2]>=cs.bounds.size()[1]) {
          splitAxis = 2;
        }

        // Middle split
        float splitPlane = cs.bounds.lower[splitAxis]+cs.bounds.size()[splitAxis]*.5f;

        Clusters L,R;
        L.bounds = { vec3f(1e30f), vec3f(-1e30f) };
        R.bounds = { vec3f(1e30f), vec3f(-1e30f) };
        for (size_t i=0; i<cs.clusterIDs.size(); ++i) {
          int clusterID = cs.clusterIDs[i];
          Cluster c = input[clusterID];
          vec3f centroid = c.domain.center();
          if (centroid[splitAxis] < splitPlane) {
            L.bounds.extend(c.domain);
            L.clusterIDs.push_back(clusterID);
          } else {
            R.bounds.extend(c.domain);
            R.clusterIDs.push_back(clusterID);
          }
        }

        kdTree[cs.kdNodeID].splitAxis = splitAxis;
        kdTree[cs.kdNodeID].splitPlane = splitPlane;

        L.kdNodeID = kdTree.size();
        kdTree[cs.kdNodeID].child1 = L.kdNodeID;
        kdTree.emplace_back(KDNode{0,0.f,INT_MAX,INT_MAX});
        clusters.push_back(L);

        R.kdNodeID = kdTree.size();
        kdTree[cs.kdNodeID].child2 = R.kdNodeID;
        kdTree.emplace_back(KDNode{0,0.f,INT_MAX,INT_MAX});
        clusters.push_back(R);
      }

      for (size_t i=0; i<assignedClusters.size(); ++i) {
        for (size_t j=0; j<assignedClusters[i].clusterIDs.size(); ++j) {
          perRank[i].push_back(assignedClusters[i].clusterIDs[j]);
        }
      }

      // for (auto kd : kdTree) {
      //   std::cout << kd.splitAxis << ','
      //             << kd.splitPlane << ','
      //             << kd.child1 << ','
      //             << kd.child2 << '\n';
      // }
    }

    bool assignedTo(int clusterID, int rankID)
    {
      for (size_t i=0; i<perRank[rankID].size(); ++i) {
        if (perRank[rankID][i]==clusterID)
          return true;
      }
      return false;
    }

    void computeCompositeOrder(const vec3f &reference)
    {
      if (kdTree.empty()) {
        std::cerr << "Warning: arbitrary composite order!\n";
        return;
      }

      compositeOrder.clear();

      std::vector<int> stack;
      int addr = 0;
      stack.push_back(addr);

      while (!stack.empty()) {
        KDNode node = kdTree[addr];

        if (node.child1 < 0 && node.child2 < 0) {
          int rankID = ~node.child1;
          assert(rankID==~node.child2);
          compositeOrder.push_back(rankID);
          addr = stack.back();
          stack.pop_back();
        } else if (node.child1 == INT_MAX) {
          addr = stack.back();
          stack.pop_back();
        } else {
          if (reference[node.splitAxis] < node.splitPlane) {
            addr = node.child1;
            stack.push_back(node.child2);
          } else {
            addr = node.child2;
            stack.push_back(node.child1);
          }
        }
      }
    }

    struct KDNode {
      int splitAxis;
      float splitPlane;
      int child1,child2;
    };

    // KD tree to sort clusters into visibility order
    std::vector<KDNode> kdTree;

    // Input clusters
    std::vector<Cluster> &input;

    // Number of ranks (input)
    int numRanks;

    // Per-rank clusterIDs
    std::vector<std::vector<int>> perRank;

    // rankIDs in composite order
    std::vector<int> compositeOrder;
  };

  // ==================================================================
  // Renderer class
  // ==================================================================

  Renderer::Renderer(int commRank,
                     int commSize,
                     int numIslands,
                     const std::string& inFileName)
    : islandID(commRank/(commSize/numIslands))
    , numIslands(numIslands)
  {
    commWorld.rank = commRank;
    commWorld.size = commSize;

    commIsland.size = commSize/numIslands;
    commIsland.rank = commRank%commIsland.size;

    MPI_Comm_split(MPI_COMM_WORLD,islandID,commIsland.rank,&commIsland.comm);

    if (1) {
      // check if this worked..
      int localRank;
      int localSize;
      MPI_Comm_rank(commIsland.comm,&localRank);
      MPI_Comm_size(commIsland.comm,&localSize);
      assert(localRank==commIsland.rank&&localSize==commIsland.size);
    }

    icetComm = icetCreateMPICommunicator(MPI_COMM_WORLD);

    if (getExt(inFileName)==".tri" || getExt(inFileName)==".obj")
      mode = Mode::TriMesh;
    else if (getExt(inFileName)==".vol")
      mode = Mode::Volume;

    assert(mode==Mode::TriMesh || mode==Mode::Volume);

#ifdef ALOHA_CPU
      owl = owlContextCreate(nullptr,1);
#else
      owl = owlContextCreate(&gpuID,1);
#endif
    module = owlModuleCreate(owl,embedded_deviceCode);
    lp = owlParamsCreate(owl,sizeof(LaunchParams),launchParamsVars,-1);
    rayGen = owlRayGenCreate(owl,module,"renderFrame",sizeof(RayGen),rayGenVars,-1);

    modelBounds = { vec3f(1e30f), vec3f(-1e30f) };

    std::vector<Cluster> clusters;

    if (mode==Mode::TriMesh) {

      Mesh::SP triMesh;

      if (getExt(inFileName)==".tri") {

        // Load binary tris
        triMesh = std::make_shared<Mesh>();
        Geometry::SP geom = std::make_shared<Geometry>();
        uint64_t numClusters, numVerts;

        std::ifstream ifs(inFileName,std::ios::binary);
        ifs.read((char *)&numClusters,sizeof(numClusters));
        ifs.read((char *)&modelBounds,sizeof(modelBounds));
        ifs.read((char *)&numVerts,sizeof(numVerts));
        geom->vertex.resize(numVerts);
        ifs.read((char *)geom->vertex.data(),sizeof(vec3f)*numVerts);

        clusters.resize(numClusters);

        uint64_t clustersPos = ifs.tellg();

        for (unsigned i=0; i<numClusters; ++i) {
          uint64_t numIndices;
          box3f domainBounds;
          ifs.read((char *)&numIndices,sizeof(numIndices));
          ifs.read((char *)&domainBounds,sizeof(domainBounds));
          uint64_t cur = ifs.tellg();
          ifs.seekg(cur+sizeof(vec3i)*numIndices);
          clusters[i] = {
            (int)i, // clusterID
            -1, // rankID; we don't know this yet
            domainBounds
          };
        }

        partitioner = std::make_shared<Partitioner>(clusters,commIsland.size);
        partitioner->partitionRoundRobin();

        ifs.seekg(clustersPos);

        std::vector<unsigned> myClusters;
        size_t myNumTriangles = 0;
        for (unsigned i=0; i<numClusters; ++i) {
          uint64_t numIndices;
          box3f domainBounds;
          ifs.read((char *)&numIndices,sizeof(numIndices));
          if (partitioner->assignedTo(i,commIsland.rank)) {
            ifs.read((char *)&domainBounds,sizeof(domainBounds));
            geom->index.resize(numIndices);
            ifs.read((char *)geom->index.data(),sizeof(vec3i)*numIndices);
            triMesh->geoms.push_back(geom);
            geom = std::make_shared<Geometry>();

            myClusters.push_back(i);
            myNumTriangles += numIndices;
          } else {
            ifs.read((char *)&domainBounds,sizeof(domainBounds));
            uint64_t cur = ifs.tellg();
            ifs.seekg(cur+sizeof(vec3i)*numIndices);
          }
        }
        std::stringstream s;
        s << "Clusters assigned to (commRank|islandID): ("
          << commRank << '|' << islandID << ")\n\t";
        for (size_t i=0; i<myClusters.size(); ++i) {
          s << myClusters[i];
          if (i < myClusters.size()-1)
            s << ", ";
          else
            s << '\n';
        }
        s << "\t# clusters on (" << commRank << '|' << islandID << "): "
          << myClusters.size() << '\n';
        s << "\t# triangles on (" << commRank << '|' << islandID << "): "
          << prettyNumber(myNumTriangles) << '\n';
        std::cout << s.str();
      } else {
        // Load obj
        try {
          triMesh = Mesh::load(inFileName);
          // Construct bounds
          for (std::size_t i=0; i<triMesh->geoms.size(); ++i)
          {
            const Geometry::SP &geom = triMesh->geoms[i];
            for (const auto &v : geom->vertex) {
              modelBounds.extend(v);
            }
          }
        } catch (...) { std::cerr << "Cannot load..\n"; }
      }
#if 0
      // Remove degenerate triangles, these don't bode well with Visionaray
      size_t brokenTris = 0;
      for (size_t i=0; i<triMesh->geoms[0]->index.size(); ++i) {
        vec3f v1 = triMesh->geoms[0]->vertex[triMesh->geoms[0]->index[i].x];
        vec3f v2 = triMesh->geoms[0]->vertex[triMesh->geoms[0]->index[i].y];
        vec3f v3 = triMesh->geoms[0]->vertex[triMesh->geoms[0]->index[i].z];

        vec3f e1 = v2-v1;
        vec3f e2 = v3-v1;

        if (length(cross(e1,e2))==0.f) {
          triMesh->geoms[0]->index[i] = {-1,-1,-1};
          brokenTris++;
        }
      }

      if (brokenTris > 0) {
        std::cout << "Removing " << brokenTris << " broken triangles\n";
        auto &g = triMesh->geoms[0];
        std::cout << g->index.size() << '\n';
        g->index.erase(std::remove_if(g->index.begin(),g->index.end(),
                       [](const vec3i v) {
                         return v.x<0 || v.y<0 || v.z<0;
                       }),
                       g->index.end());
        std::cout << g->index.size() << '\n';
      }
#endif

#if 1
      // Flatten, and keep only the vertices that are used by our cluster(s)
      size_t numTriangles=0;
      for (auto& geom: triMesh->geoms) {
        numTriangles += geom->index.size();
      }

      std::vector<vec3f> ourVertices(numTriangles*3);
      size_t index=0;
      for (auto& geom: triMesh->geoms) {
        for (size_t i=0; i<geom->index.size(); ++i) {
          vec3i idx(index,index+1,index+2);
          ourVertices[idx.x] = triMesh->geoms[0]->vertex[geom->index[i].x];
          ourVertices[idx.y] = triMesh->geoms[0]->vertex[geom->index[i].y];
          ourVertices[idx.z] = triMesh->geoms[0]->vertex[geom->index[i].z];
          geom->index[i] = idx;
          index += 3;
        }
      }

      triMesh->geoms[0]->vertex = ourVertices;

      std::stringstream s;
      s << "Number of vertices on global rank " << commWorld.rank << ": "
        << triMesh->geoms[0]->vertex.size() << '\n';
      std::cout << s.str();
#endif

      // Create triangle geometry for cluster meshes
      asTriMesh.triangleGeomType = owlGeomTypeCreate(owl,
                                                     OWL_TRIANGLES,
                                                     sizeof(TriangleGeom),
                                                     triangleGeomVars, -1);
      owlGeomTypeSetClosestHit(asTriMesh.triangleGeomType, 0, module, "ModelCH");

      asTriMesh.tlasGroup = owlInstanceGroupCreate(owl, triMesh->geoms.size());

      const std::vector<vec3f> &vertex = triMesh->geoms[0]->vertex;
#ifdef ALOHA_CPU
      OWLBuffer vertexBuffer = owlBufferCreateEXT(owl, OWL_FLOAT3, vertex.size(), vertex.data());
#else
      // OWLBuffer vertexBuffer = owlDeviceBufferCreate(owl, OWL_FLOAT3, vertex.size(), vertex.data());
      OWLBuffer vertexBuffer = owlManagedMemoryBufferCreate(owl, OWL_FLOAT3, vertex.size(), vertex.data());
#endif

      uint64_t numClusters = clusters.size();
      int numClustersPerRank = numClusters/commIsland.size;

      for (std::size_t i = 0; i < triMesh->geoms.size(); ++i) {
        unsigned clusterID = commIsland.rank*numClustersPerRank+i; 
        const Geometry::SP &geom = triMesh->geoms[i];
#ifdef ALOHA_CPU
        OWLBuffer indexBuffer = owlBufferCreateEXT(owl, OWL_INT3, geom->index.size(), geom->index.data());
#else
        // OWLBuffer indexBuffer = owlDeviceBufferCreate(owl, OWL_INT3, geom->index.size(), geom->index.data());
        OWLBuffer indexBuffer = owlManagedMemoryBufferCreate(owl, OWL_INT3, geom->index.size(), geom->index.data());
#endif
        OWLGeom ogeom;
        ogeom = owlGeomCreate(owl, asTriMesh.triangleGeomType);
        owlTrianglesSetVertices(ogeom, vertexBuffer, vertex.size(), sizeof(vec3f), 0);
        owlTrianglesSetIndices(ogeom, indexBuffer, geom->index.size(), sizeof(vec3i), 0);

        OWLGroup modelGroup = owlTrianglesGeomGroupCreate(owl, 1, &ogeom);
        // OWLGroup modelGroup = owlTrianglesGeomGroupCreate(owl, 1, &ogeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        owlGroupBuildAccel(modelGroup);
        // owlParamsSetBuffer(lp,"model.vertexBuffer",vertexBuffer);
        // owlParamsSetBuffer(lp,"model.indexBuffer",indexBuffer);
        owlGeomSet1ui(ogeom,"clusterID",clusterID);
        owlGeomSetBuffer(ogeom,"vertexBuffer",vertexBuffer);
        owlGeomSetBuffer(ogeom,"indexBuffer",indexBuffer);
        // owlInstanceGroupSetChild(modelGroup, modelIndex, gi);
        // owlParamsSetGroup(lp,"model.group",modelGroup);

        owlInstanceGroupSetChild(asTriMesh.tlasGroup, i, modelGroup);

        asTriMesh.modelGroups.push_back(modelGroup);

        clusterID += numIslands;
      }

      owlParamsSetGroup(lp, "world", asTriMesh.tlasGroup);

      owlGroupBuildAccel(asTriMesh.tlasGroup);
    } else if (mode==Mode::Volume) {
      std::vector<Brick::SP> volBricks;

      uint64_t numClusters;
      box3i cellRange;

      std::ifstream ifs(inFileName,std::ios::binary);
      ifs.read((char *)&numClusters,sizeof(numClusters));
      ifs.read((char *)&asVolume.cellRange,sizeof(asVolume.cellRange));
      ifs.read((char *)&asVolume.voxelRange,sizeof(asVolume.voxelRange));
      ifs.read((char *)&modelBounds,sizeof(modelBounds));

      clusters.resize(numClusters);

      uint64_t clustersPos = ifs.tellg();

      for (unsigned i=0; i<numClusters; ++i) {
        box3i cellRange;
        box3i voxelRange;
        box3f spaceRange;
        interval<float> valueRange;
        ifs.read((char *)&cellRange,sizeof(cellRange));
        ifs.read((char *)&voxelRange,sizeof(voxelRange));
        ifs.read((char *)&spaceRange,sizeof(spaceRange));
        ifs.read((char *)&valueRange,sizeof(valueRange));
        size_t numVoxels = voxelRange.size().x * size_t(voxelRange.size().y)
                                    * voxelRange.size().z;
        uint64_t cur = ifs.tellg();
        ifs.seekg(cur+sizeof(float)*numVoxels);
        clusters[i] = {
          (int)i, // clusterID
          -1, // rankID; we don't know this yet
          spaceRange
        };
      }

      partitioner = std::make_shared<Partitioner>(clusters,commIsland.size);
      partitioner->partitionKD();

      ifs.seekg(clustersPos);

      std::vector<unsigned> myClusters;
      size_t myNumVoxels = 0;
      for (unsigned i=0; i<numClusters; ++i) {
        box3i cellRange;
        box3i voxelRange;
        box3f spaceRange;
        interval<float> valueRange;
        ifs.read((char *)&cellRange,sizeof(cellRange));
        ifs.read((char *)&voxelRange,sizeof(voxelRange));
        ifs.read((char *)&spaceRange,sizeof(spaceRange));
        ifs.read((char *)&valueRange,sizeof(valueRange));
        size_t numVoxels = voxelRange.size().x * size_t(voxelRange.size().y)
                                    * voxelRange.size().z;
        if (partitioner->assignedTo(i,commIsland.rank)) {
          Brick::SP brick = std::make_shared<Brick>();
          brick->cellRange    = cellRange;
          brick->voxelRange   = voxelRange;
          brick->spaceRange   = spaceRange;
          brick->valueRange   = valueRange;
          brick->voxels.resize(numVoxels);

          volBricks.push_back(brick);

          ifs.read((char *)brick->voxels.data(),sizeof(float)*numVoxels);
          myClusters.push_back(i);
          myNumVoxels += numVoxels;
        } else {
          uint64_t cur = ifs.tellg();
          ifs.seekg(cur+sizeof(float)*numVoxels);
        }
      }
      std::stringstream s;
      s << "Clusters assigned to (commRank|islandID): ("
        << commRank << '|' << islandID << ")\n\t";
      for (size_t i=0; i<myClusters.size(); ++i) {
        s << myClusters[i];
        if (i < myClusters.size()-1)
          s << ", ";
        else
          s << '\n';
      }
      s << "\t# clusters on (" << commRank << '|' << islandID << "): "
        << myClusters.size() << '\n';
      s << "\t# cells on (" << commRank << '|' << islandID << "): "
        << prettyNumber(myNumVoxels) << '\n';
      std::cout << s.str();

      owlBuildPrograms(owl);


      // Create user geometry for clusters (only for debugging!)
      asVolume.brickGeomType = owlGeomTypeCreate(owl,
                                                 OWL_GEOM_USER,
                                                 sizeof(BrickGeom),
                                                 brickGeomVars, -1);
      owlGeomTypeSetBoundsProg(asVolume.brickGeomType, module, "VolBrickBounds");
      owlGeomTypeSetIntersectProg(asVolume.brickGeomType, 0, module, "VolBrickIsect");
      owlGeomTypeSetClosestHit(asVolume.brickGeomType, 0, module, "VolBrickCH");

      OWLGeom brickGeom = owlGeomCreate(owl, asVolume.brickGeomType);
      owlGeomSetPrimCount(brickGeom, volBricks.size());

      std::vector<VolBrick> bricks(volBricks.size());
      for (size_t i=0; i<volBricks.size(); ++i) {
        bricks[i].cellRange = volBricks[i]->cellRange;
        bricks[i].voxelRange = volBricks[i]->voxelRange;
        bricks[i].spaceRange = volBricks[i]->spaceRange;
        bricks[i].valueRange = volBricks[i]->valueRange;
        vec3i texSize = volBricks[i]->voxelRange.upper-volBricks[i]->voxelRange.lower;
#ifdef ALOHA_CPU
        OWLTexture volTex
          = owlTexture3DCreateEXT(owl,
                                  OWL_TEXEL_FORMAT_R32F,
                                  texSize.x,texSize.y,texSize.z,
                                  volBricks[i]->voxels.data(),
                                  OWL_TEXTURE_LINEAR,
                                  OWL_TEXTURE_CLAMP);
        bricks[i].texture = owlTextureGetObject(volTex,0);
#else

        cudaTextureObject_t volumeTexture;
        cudaResourceDesc res_desc = {};
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

        // texture<float, 3, cudaReadModeElementType> 
        cudaArray_t   voxelArray;
        cudaMalloc3DArray(&voxelArray,
                          &channel_desc,
                          make_cudaExtent(texSize.x,
                                          texSize.y,
                                          texSize.z));
        
        cudaMemcpy3DParms copyParams = {0};
        cudaExtent volumeSize = make_cudaExtent(texSize.x,
                                                texSize.y,
                                                texSize.z);
        copyParams.srcPtr
          = make_cudaPitchedPtr((void *)volBricks[i]->voxels.data(),
                                volumeSize.width * sizeof(float),
                                volumeSize.width,
                                volumeSize.height);
        copyParams.dstArray = voxelArray;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
        
        cudaResourceDesc            texRes;
        memset(&texRes,0,sizeof(cudaResourceDesc));
        
        texRes.resType            = cudaResourceTypeArray;
        texRes.res.array.array    = voxelArray;
        
        cudaTextureDesc             texDescr;
        memset(&texDescr,0,sizeof(cudaTextureDesc));
        
        texDescr.normalizedCoords = true; // access with normalized texture coordinates
        texDescr.filterMode       = cudaFilterModeLinear; // linear interpolation
        // wrap texture coordinates
        texDescr.addressMode[0] = cudaAddressModeClamp;//Wrap;
        texDescr.addressMode[1] = cudaAddressModeClamp;//Wrap;
        texDescr.addressMode[2] = cudaAddressModeClamp;//Wrap;
        texDescr.sRGB                = 0;

        // texDescr.addressMode[0]      = cudaAddressModeBorder;
        // texDescr.addressMode[1]      = cudaAddressModeBorder;
        texDescr.filterMode          = cudaFilterModeLinear;
        texDescr.normalizedCoords    = 1;
        texDescr.maxAnisotropy       = 1;
        texDescr.maxMipmapLevelClamp = 0;
        texDescr.minMipmapLevelClamp = 0;
        texDescr.mipmapFilterMode    = cudaFilterModePoint;
        texDescr.borderColor[0]      = 0.0f;
        texDescr.borderColor[1]      = 0.0f;
        texDescr.borderColor[2]      = 0.0f;
        texDescr.borderColor[3]      = 0.0f;
        texDescr.sRGB                = 0;
        
        texDescr.readMode = cudaReadModeElementType;
        
        cudaCreateTextureObject(&volumeTexture, &texRes, &texDescr, NULL);
        bricks[i].texture = volumeTexture;
#endif
      }
#ifdef ALOHA_CPU
      OWLBuffer brickBuffer = owlBufferCreateEXT(owl, OWL_USER_TYPE(VolBrick{}), bricks.size(), bricks.data());
#else
      OWLBuffer brickBuffer = owlDeviceBufferCreate(owl, OWL_USER_TYPE(VolBrick{}), bricks.size(), bricks.data());
#endif
      owlGeomSetBuffer(brickGeom,"brickBuffer",brickBuffer);

      owlParamsSetBuffer(lp,"brickBuffer",brickBuffer);

      owlBuildPrograms(owl);

      asVolume.blasGroup = owlUserGeomGroupCreate(owl, 1, &brickGeom);
      owlGroupBuildAccel(asVolume.blasGroup);

      asVolume.tlasGroup = owlInstanceGroupCreate(owl, 1);
      owlInstanceGroupSetChild(asVolume.tlasGroup, 0, asVolume.blasGroup);

      owlParamsSetGroup(lp, "volBricks", asVolume.tlasGroup);

      owlGroupBuildAccel(asVolume.tlasGroup);
    }


    // Create user geometry for clusters (only for debugging!)
    clusterGeomType = owlGeomTypeCreate(owl,
                                        OWL_GEOM_USER,
                                        sizeof(ClusterGeom),
                                        clusterGeomVars, -1);
    owlGeomTypeSetBoundsProg(clusterGeomType, module, "ClusterBounds");
    owlGeomTypeSetIntersectProg(clusterGeomType, 0, module, "ClusterIsect");
    owlGeomTypeSetClosestHit(clusterGeomType, 0, module, "ClusterCH");

    OWLGeom clusterGeom = owlGeomCreate(owl, clusterGeomType);
    owlGeomSetPrimCount(clusterGeom, clusters.size());
#ifdef ALOHA_CPU
    OWLBuffer clusterBuffer = owlBufferCreateEXT(owl, OWL_USER_TYPE(Cluster{}), clusters.size(), clusters.data());
#else
    OWLBuffer clusterBuffer = owlDeviceBufferCreate(owl, OWL_USER_TYPE(Cluster{}), clusters.size(), clusters.data());
#endif
    owlGeomSetBuffer(clusterGeom,"clusterBuffer",clusterBuffer);

    owlBuildPrograms(owl);

    blasClusters = owlUserGeomGroupCreate(owl, 1, &clusterGeom);
    owlGroupBuildAccel(blasClusters);

    tlasClusters = owlInstanceGroupCreate(owl, 1);
    owlInstanceGroupSetChild(tlasClusters, 0, blasClusters);

    owlParamsSetGroup(lp, "clusters", tlasClusters);

    owlBuildPipeline(owl);
    owlBuildSBT(owl);


    owlParamsSet3f(lp,"domain.lower",
                   modelBounds.lower.x,
                   modelBounds.lower.y,
                   modelBounds.lower.z);
    owlParamsSet3f(lp,"domain.upper",
                   modelBounds.upper.x,
                   modelBounds.upper.y,
                   modelBounds.upper.z);

    owlParamsSet1i(lp,"islandID",islandID);
    owlParamsSet1i(lp,"numIslands",numIslands);

    owlBuildSBT(owl);
  }

  Renderer::~Renderer()
  {
    icetDestroyMPICommunicator(icetComm);
  }

  void Renderer::handleCameraEvent(const vec3f &org,
                                   const vec3f &dir_00,
                                   const vec3f &dir_du,
                                   const vec3f &dir_dv)
  {
    float sceneRadius = length(modelBounds.center()+modelBounds.upper);
    float maxDepth = length(org+modelBounds.center())+sceneRadius;
    owlParamsSet1f(lp,"maxDepth", maxDepth);

    owlParamsSet3f(lp,"camera.org",   org.x,org.y,org.z);
    owlParamsSet3f(lp,"camera.dir_00",dir_00.x,dir_00.y,dir_00.z);
    owlParamsSet3f(lp,"camera.dir_du",dir_du.x,dir_du.y,dir_du.z);
    owlParamsSet3f(lp,"camera.dir_dv",dir_dv.x,dir_dv.y,dir_dv.z);

    if (mode==Mode::Volume)
      partitioner->computeCompositeOrder(org);

    accumID = 0;
  }

  void Renderer::handleResizeEvent(const vec2i &newSize)
  {
    if (newSize != this->fbSize) {
      if (!accumBuffer)
#ifdef ALOHA_CPU
        accumBuffer = owlBufferCreateEXT(owl,OWL_FLOAT4,1,nullptr);
#else
        accumBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT4,1,nullptr);
#endif
      owlBufferResize(accumBuffer,newSize.x*newSize.y);
      owlParamsSetBuffer(lp,"accumBuffer",accumBuffer);
      if (!depthBuffer)
#ifdef ALOHA_CPU
        depthBuffer = owlBufferCreateEXT(owl,OWL_FLOAT,1,nullptr);
#else
        depthBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT,1,nullptr);
#endif
      owlBufferResize(depthBuffer,newSize.x*newSize.y);
      owlParamsSetBuffer(lp,"fbDepth",depthBuffer);
      this->fbSize = newSize;

      // Also reset IceT
      if (icetCtx == NULL) {
        icetCtx = icetCreateContext(icetComm);
        icetSetContext(icetCtx);
      }

      icetResetTiles();
      icetAddTile(0, 0, fbSize.x, fbSize.y, displayRank);

      if (mode==Mode::TriMesh) {
        icetDisable(ICET_COMPOSITE_ONE_BUFFER); // w/ depth
        // icetEnable(ICET_COMPOSITE_ONE_BUFFER);
        icetStrategy(ICET_STRATEGY_REDUCE);
        icetCompositeMode(ICET_COMPOSITE_MODE_Z_BUFFER);
        icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
        icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);
      } else if (mode==Mode::Volume) {
        icetEnable(ICET_COMPOSITE_ONE_BUFFER);
        icetStrategy(ICET_STRATEGY_SEQUENTIAL);
        icetCompositeMode(ICET_COMPOSITE_MODE_BLEND);
        icetEnable(ICET_ORDERED_COMPOSITE);
        icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
        icetSetDepthFormat(ICET_IMAGE_DEPTH_NONE);
      }
    }
    owlParamsSetPointer(lp,"fbPointer",fbPointer);
  }

  void Renderer::handleScreenShotEvent(const std::string &baseName)
  {
    const uint32_t *fb
      = (const uint32_t*)fbPointer;
     
    if (fb == nullptr)
      return;

    std::vector<uint32_t> pixels;
    for (int y=0;y<fbSize.y;y++) {
      const uint32_t *line = fb + (fbSize.y-1-y)*fbSize.x;
      for (int x=0;x<fbSize.x;x++) {
        pixels.push_back(line[x] | (0xff << 24));
      }
    }
    std::stringstream str;
    str << baseName << "_rank" << commIsland.rank << "_island" << islandID << ".png";
    std::string fileName = str.str();
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    std::cout << "#owl.viewer: frame buffer written to " << fileName << std::endl;
  }

  void Renderer::handleColorMapEvent(const std::vector<vec4f> &newCM)
  {
    asVolume.colorMap = newCM;
    if (!asVolume.colorMapBuffer)
#ifdef ALOHA_CPU
      asVolume.colorMapBuffer = owlBufferCreateEXT(owl,OWL_FLOAT4,
                                                   newCM.size(),nullptr);
#else
      asVolume.colorMapBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT4,
                                                      newCM.size(),nullptr);
#endif

    owlBufferUpload(asVolume.colorMapBuffer,newCM.data());
    
    if (asVolume.colorMapTexture != 0) {
      cudaDestroyTextureObject(asVolume.colorMapTexture);
      asVolume.colorMapTexture = 0;
    }

    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc
      = cudaCreateChannelDesc<float4>();

    // cudaArray_t   voxelArray;
    if (asVolume.colorMapArray == 0) {
      cudaMallocArray(&asVolume.colorMapArray,
                      &channel_desc,
                      newCM.size(),1);
    }
    
    int pitch = newCM.size()*sizeof(newCM[0]);
#ifdef ALOHA_CPU
    cudaMemcpy2DToArray(asVolume.colorMapArray,
                        /* offset */0,0,
                        newCM.data(),
                        pitch,pitch,1,
                        cudaMemcpyDefault);
#else
    cudaMemcpy2DToArray(asVolume.colorMapArray,
                        /* offset */0,0,
                        newCM.data(),
                        pitch,pitch,1,
                        cudaMemcpyHostToDevice);
#endif

    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = asVolume.colorMapArray;
    
    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = cudaAddressModeBorder;
    tex_desc.addressMode[1]      = cudaAddressModeBorder;
    tex_desc.filterMode          = cudaFilterModeLinear;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 0.0f;
    tex_desc.borderColor[1]      = 0.0f;
    tex_desc.borderColor[2]      = 0.0f;
    tex_desc.borderColor[3]      = 0.0f;
    tex_desc.sRGB                = 0;
    cudaCreateTextureObject(&asVolume.colorMapTexture, &res_desc, &tex_desc,
                            nullptr);

    owlParamsSetRaw(lp,"transferFunc.texture",&asVolume.colorMapTexture);

    accumID = 0;
  }

  void Renderer::render(uint32_t *fbPointer)
  {
    this->fbPointer = fbPointer; // store so we can take screen shots..

    processEvents();

    owlParamsSet1i(lp,"accumID",accumID);
    accumID++;
    owlParamsSet1f(lp,"render.dt",2.f);
    owlParamsSet1i(lp,"render.spp",max(spp,1));
    owlParamsSet1i(lp,"render.heatMapEnabled",heatMapEnabled);
    owlParamsSet1f(lp,"render.heatMapScale",heatMapScale);

    owlLaunch2D(rayGen,fbSize.x,fbSize.y,lp);

    // Composite with IceT
    icetSetContext(icetCtx);

    IceTFloat bg[4] = { 1.0, 1.0, 1.0, 1.0 };

    IceTDouble mv[16], proj[16];
    int viewport[4] = { 0,0,fbSize.x,fbSize.y };
    for (int i=0; i<16; ++i) {
      mv[i]=(i%5==0)?1.:0.;
      proj[i]=(i%5==0)?1.:0.;
    }

    // Set compositing order for DVR; this gets recomputed whenever the camera changes
    if (mode==Mode::Volume) {
      std::vector<int> compositeOrder;
      for (int i=0; i<numIslands; ++i) {
        for (size_t j=0; j<partitioner->compositeOrder.size(); ++j) {
          int rankID = partitioner->compositeOrder[j];
          compositeOrder.push_back(i*commIsland.size+rankID);
        }
      }
      icetCompositeOrder(compositeOrder.data());
    }

#ifdef ALOHA_CPU
    if (mode==Mode::Volume) {
      icetImg = icetCompositeImage(fbPointer, NULL, viewport, proj, mv, bg);
    } else {
      icetImg = icetCompositeImage(fbPointer, (float *)owlBufferGetPointer(depthBuffer,0), viewport, proj, mv, bg);
    }
    if (!icetImageIsNull(icetImg)) {
       IceTUByte *color = icetImageGetColorub(icetImg);
       memcpy(fbPointer,(void *)color,fbSize.x*fbSize.y*sizeof(uint32_t));
    }
#else
    std::vector<uint32_t> hFbPointer(fbSize.x*fbSize.y);
    cudaMemcpy(hFbPointer.data(),fbPointer,fbSize.x*fbSize.y*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    if (mode==Mode::Volume) {
      icetImg = icetCompositeImage(hFbPointer.data(), NULL, viewport, proj, mv, bg);
    } else {
      std::vector<float> hDepthBuffer(fbSize.x*fbSize.y);
      cudaMemcpy(hDepthBuffer.data(),owlBufferGetPointer(depthBuffer,0),fbSize.x*fbSize.y*sizeof(float),
                 cudaMemcpyDeviceToHost);
      icetImg = icetCompositeImage(hFbPointer.data(), hDepthBuffer.data(), viewport, proj, mv, bg);
    }
    if (!icetImageIsNull(icetImg)) {
       IceTUByte *color = icetImageGetColorub(icetImg);
       cudaMemcpy(fbPointer,(void *)color,fbSize.x*fbSize.y*sizeof(uint32_t),cudaMemcpyHostToDevice);
    }
#endif
  }
}
