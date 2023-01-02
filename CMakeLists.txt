
project(waikikiViewer)

cmake_minimum_required(VERSION 3.1.3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Build all libraries as shared libraries instead of static" ON)

option(ALOHA_CPU "Build the project to run on the CPU" ON)
if(ALOHA_CPU)
  set(owl_dir ${PROJECT_SOURCE_DIR}/submodules/fakeOwl)
  add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
  include_directories(${owl_dir}/include)
  include(${owl_dir}/cmake/configure_fake_owl.cmake)
  fake_owl_compile_and_embed(
    embedded_deviceCode deviceCode.cpp
    )
set(CUTEEOWL_USE_CUDA OFF CACHE BOOL "Build with CUDA frame buffer implementation" FORCE)
else()
  enable_language(CUDA)
  set(owl_dir ${CMAKE_CURRENT_SOURCE_DIR}/submodules/owl)
  add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
  include_directories(${OWL_INCLUDES})
  
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
  include(embed_ptx)
  embed_ptx(
    OUTPUT_TARGET
      embedded_deviceCode
    EMBEDDED_SYMBOL_NAMES
      embedded_deviceCode
    PTX_LINK_LIBRARIES
      owl::owl
    SOURCES
      deviceCode.cu
  )
set(CUTEEOWL_USE_CUDA ON CACHE BOOL "Build with CUDA frame buffer implementation" FORCE)
endif()
add_subdirectory(${PROJECT_SOURCE_DIR}/submodules/cuteeOwl)
include_directories(${PROJECT_SOURCE_DIR}/submodules/cuteeOwl)
include_directories(${QT_OWL_INCLUDES})
find_package(Qt5Widgets REQUIRED)
find_package(OpenGL REQUIRED)
set(CMAKE_AUTOMOC ON)

# ------------------------------------------------------------------
# waikikiViewer
# ------------------------------------------------------------------
find_package(MPI COMPONENTS C REQUIRED)
include_directories(${MPI_C_INCLUDE_PATH})

# add_subdirectory(submodules/IceT)
find_package(IceT REQUIRED)
include_directories(${ICET_INCLUDE_DIRS})

add_executable(waikikiRenderer
  waikikiRenderer.cpp
  viewer.cpp
  DistributedRenderer.cpp
  mesh.cpp
  Headless.cpp
  )
if(ALOHA_CPU)
  target_sources(waikikiRenderer PRIVATE ${embedded_deviceCode})
endif()

target_link_directories(waikikiRenderer PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/submodules/cuteeOwl/qtOWL)
if(NOT ALOHA_CPU)
  target_link_libraries(waikikiRenderer embedded_deviceCode)
endif()
target_link_libraries(waikikiRenderer
      owl::owl
      qtOWL
      ${ICET_CORE_LIBS}
      ${ICET_MPI_LIBS}
      ${OPENGL_LIBRARIES}
      ${MPI_C_LIBRARIES}
      ${MPI_C_LINK_FLAGS}
      ${ICET_LIBARIES})
qt5_use_modules(waikikiRenderer Widgets)
set_target_properties(waikikiRenderer PROPERTIES MACOSX_BUNDLE YES)
if(ALOHA_CPU)
  target_compile_options(waikikiRenderer PUBLIC -DALOHA_CPU=1)
endif()
# QT_OWL_LINK(waikikiRenderer)

# ------------------------------------------------------------------
# chopSuey - chop triangle models into pieces
# ------------------------------------------------------------------
set(mini_dir submodules/miniScene/miniScene)
add_library(miniScene STATIC
  ${mini_dir}/LocalScene.cpp
  ${mini_dir}/SplitSpecs.cpp
  ${mini_dir}/Scene.cpp
  ${mini_dir}/Serialized.cpp
  )
target_link_libraries(miniScene owl::owl)
target_include_directories(miniScene PUBLIC submodules/miniScene)
add_executable(chopSuey chopSuey.cpp mesh.cpp FileMapping.cpp)
target_link_libraries(chopSuey miniScene owl::owl)
target_compile_options(chopSuey PUBLIC -DUSE_MINI=1)


# ------------------------------------------------------------------
# chopInfo - print info about split files
# ------------------------------------------------------------------
add_executable(chopInfo chopInfo.cpp)
target_link_libraries(chopInfo owl::owl)


# ------------------------------------------------------------------
# chopExtract - take .tri file as input and extract cluster(s)
# ------------------------------------------------------------------
add_executable(chopExtract chopExtract.cpp)
target_link_libraries(chopExtract owl::owl)



