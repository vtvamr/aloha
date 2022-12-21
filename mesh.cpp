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

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#ifdef USE_MINI
#include <miniScene/Scene.h>
#include <miniScene/Serialized.h>
#endif

#include "mesh.h"

namespace maui {

  inline std::string getExt(const std::string &fileName)
  {
    int pos = fileName.rfind('.');
    if (pos == fileName.npos)
      return "";
    return fileName.substr(pos);
  }

  Mesh::SP Mesh::loadOBJ(std::string objFileName) {
    Mesh::SP res = std::make_shared<Mesh>();

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    std::string modelDir = objFileName.substr(0, objFileName.rfind('/') + 1);

    bool readOK
      = tinyobj::LoadObj(&attrib,
                        &shapes,
                        &materials,
                        &err,
                        &err,
                        objFileName.c_str(),
                        modelDir.c_str(),
                        /* triangulate */true,
                        /* default vertex colors fallback*/ false);

    if (!readOK)
      throw std::runtime_error("Could not read OBJ model from " + objFileName + " : " + err);

    for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
      tinyobj::shape_t& shape = shapes[shapeID];
      Geometry::SP geom = std::make_shared<Geometry>();
      // Just lazily copy _all_ the vertices into this geom..
      for (std::size_t i=0; i<attrib.vertices.size(); i+=3) {
        float *v = attrib.vertices.data() + i;
        geom->vertex.push_back({v[0],v[1],v[2]});
      }
      for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
        if (shape.mesh.num_face_vertices[faceID] != 3)
          throw std::runtime_error("not properly tessellated"); // while this would actually be rather easy...
        tinyobj::index_t idx0 = shape.mesh.indices[3*faceID];
        tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
        tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
        geom->index.push_back({idx0.vertex_index,
                               idx1.vertex_index,
                               idx2.vertex_index});
      }
      res->geoms.push_back(geom);
    }

    return res;
  }

  Mesh::SP Mesh::loadMini(std::string miniFileName) {
#ifdef USE_MINI
    std::cout << "Loading mini scene...!\n";
    mini::Scene::SP scene = mini::Scene::load(miniFileName);
    std::cout << "Done!\n";

    std::cout << "Serializing mini scene...!\n";
    mini::SerializedScene serialized(scene.get());
    std::cout << "Done!\n";

    Mesh::SP res = std::make_shared<Mesh>();

    size_t numUniqueMeshes = 0;
    size_t numUniqueTriangles = 0;
    size_t numUniqueVertices = 0;
    
    Geometry::SP geom = std::make_shared<Geometry>();
    for (auto mesh : serialized.meshes.list) {
      size_t offV = geom->vertex.size();
      size_t offI = geom->index.size();
      geom->vertex.resize(geom->vertex.size()+mesh->vertices.size());
      geom->index.resize(geom->index.size()+mesh->indices.size());
      memcpy(geom->vertex.data()+offV,mesh->vertices.data(),mesh->vertices.size()*sizeof(vec3f));
      for (size_t i=0; i<mesh->indices.size(); ++i) {
        geom->index[offI+i] = mesh->indices[i]+vec3i((int)offV);
      }

      numUniqueMeshes++;
      numUniqueTriangles += mesh->indices.size();
      numUniqueVertices  += mesh->vertices.size();
      std::cout << "Mesh no. " << numUniqueMeshes << ", #vertices: " << mesh->vertices.size()
                << ", #indices: " << mesh->indices.size() << '\n';
    }
    res->geoms.push_back(geom);
    std::cout << "num *unique* meshes\t: "    << (numUniqueMeshes) << std::endl;
    std::cout << "num *unique* triangles\t: " << (numUniqueTriangles) << std::endl;
    std::cout << "num *unique* vertices\t: "  << (numUniqueVertices) << std::endl;
    return res;
#else
    return NULL;
#endif
  }

  Mesh::SP Mesh::load(std::string fileName) {
    if (getExt(fileName)==".obj") {
      return loadOBJ(fileName);
    } else if (getExt(fileName)==".mini") {
      return loadMini(fileName);
    }
    return NULL;
  }
}
