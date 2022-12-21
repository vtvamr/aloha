#include <algorithm>
#include <string>
#include <fstream>
#include <stdint.h>
#include <owl/owl_host.h>
#include "owl/common/math/box.h"
#include "mesh.h"

using namespace owl;
using namespace owl::common;

namespace maui {

  struct {
    std::string inFileName = "";
    std::string outFileName = "chopExtract.tri";
    int clusterID = -1;
  } cmdline;

  std::string myPretty(size_t n)
  {
    if (n < 1024) return "  "+prettyNumber(n);

    return "  "+prettyNumber(n)+"\t("+std::to_string(n)+")";
  }

  void usage(const std::string &err)
  {
    if (err != "")
      std::cout << OWL_TERMINAL_RED << "\nFatal error: " << err
                << OWL_TERMINAL_DEFAULT << std::endl << std::endl;

    std::cout << "Usage: ./chopExtract chopFile.tri -o output.tri -cid <clusterID>" << std::endl;
    std::cout << std::endl;
    exit(1);
  }

  extern "C" int main(int argc, char **argv)
  {
    for (int i=1;i<argc;i++) {
      const std::string arg = argv[i];
      if (arg[0] != '-') {
        cmdline.inFileName = arg;
      }
      else if (arg == "-o") {
        cmdline.outFileName = argv[++i];
      }
      else if (arg == "-cid") {
        cmdline.clusterID = std::atoi(argv[++i]);
      }
      else
        usage("unknown cmdline arg '"+arg+"'");
    }

    if (cmdline.inFileName == "")
      usage("no filename specified");

    if (cmdline.clusterID == -1)
      usage("no clusterID specified");

    Geometry::SP geom = std::make_shared<Geometry>();

    uint64_t numClusters, numVerts;
    box3f modelBounds;
    std::ifstream ifs(cmdline.inFileName,std::ios::binary);
    ifs.read((char *)&numClusters,sizeof(numClusters));
    ifs.read((char *)&modelBounds,sizeof(modelBounds));
    ifs.read((char *)&numVerts,sizeof(numVerts));
    geom->vertex.resize(numVerts);
    ifs.read((char *)geom->vertex.data(),sizeof(vec3f)*numVerts);

    box3f domainBounds;
    for (unsigned i=0; i<numClusters; ++i) {
      uint64_t numIndices;
      ifs.read((char *)&numIndices,sizeof(numIndices));
      if (i != (unsigned)cmdline.clusterID) {
        ifs.read((char *)&domainBounds,sizeof(domainBounds));
        uint64_t cur = ifs.tellg();
        ifs.seekg(cur+sizeof(vec3i)*numIndices);
      } else {
        ifs.read((char *)&domainBounds,sizeof(domainBounds));
        geom->index.resize(numIndices);
        ifs.read((char *)geom->index.data(),sizeof(vec3i)*numIndices);
        break;
      }
    }

    int minIndex = INT_MAX;
    int maxIndex = -1;

    for (size_t i=0; i<geom->index.size(); ++i) {
      minIndex = std::min(minIndex,geom->index[i].x);
      minIndex = std::min(minIndex,geom->index[i].y);
      minIndex = std::min(minIndex,geom->index[i].z);

      maxIndex = std::max(maxIndex,geom->index[i].x);
      maxIndex = std::max(maxIndex,geom->index[i].y);
      maxIndex = std::max(maxIndex,geom->index[i].z);
    }

    for (size_t i=0; i<geom->index.size(); ++i) {
      geom->index[i].x -= minIndex;
      geom->index[i].y -= minIndex;
      geom->index[i].z -= minIndex;
    }

    // Write out
    numClusters = 1;
    modelBounds = domainBounds; // TODO: make this optional
    numVerts = maxIndex-minIndex+1;

    std::ofstream ofs(cmdline.outFileName,std::ios::binary);
    ofs.write((const char *)&numClusters,sizeof(numClusters));
    ofs.write((const char *)&modelBounds,sizeof(modelBounds));
    ofs.write((const char *)&numVerts,sizeof(numVerts));
    ofs.write((const char *)(geom->vertex.data()+minIndex),sizeof(vec3f)*numVerts);

    uint64_t numIndices = geom->index.size();
    ofs.write((const char *)&numIndices,sizeof(numIndices));
    ofs.write((const char *)&domainBounds,sizeof(domainBounds));
    ofs.write((const char *)geom->index.data(),sizeof(vec3i)*numIndices);
  }
}


