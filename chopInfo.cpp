#include <string>
#include <fstream>
#include <stdint.h>
#include <owl/owl_host.h>
#include "owl/common/math/box.h"

using namespace owl;
using namespace owl::common;

namespace maui {

  struct {
    std::string inFileName = "";
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

    std::cout << "Usage: ./chopInfo chopFile.tri" << std::endl;
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
      else
        usage("unknown cmdline arg '"+arg+"'");
    }

    if (cmdline.inFileName == "")
      usage("no filename specified");

    uint64_t numClusters, numVerts;
    box3f modelBounds;
    std::ifstream ifs(cmdline.inFileName,std::ios::binary);
    ifs.read((char *)&numClusters,sizeof(numClusters));
    ifs.read((char *)&modelBounds,sizeof(modelBounds));
    ifs.read((char *)&numVerts,sizeof(numVerts));
    uint64_t cur = ifs.tellg();
    ifs.seekg(cur+sizeof(vec3f)*numVerts);

    std::cout << "Model with:\n\t# clusters: " << numClusters << '\n'
              << "\t# vertices: " << numVerts << '\n'
              << "\t model bounds: " << modelBounds << "\n\n";

    for (unsigned i=0; i<numClusters; ++i) {
      uint64_t numIndices;
      box3f domainBounds;
      ifs.read((char *)&numIndices,sizeof(numIndices));
      ifs.read((char *)&domainBounds,sizeof(domainBounds));
      uint64_t cur = ifs.tellg();
      ifs.seekg(cur+sizeof(vec3i)*numIndices);

      std::cout << "Cluster #" << i << ": " << prettyNumber(numIndices) << " triangles,"
                << " domain bounds: " << domainBounds << '\n';
    }
  }
}


