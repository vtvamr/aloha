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

#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace maui {

  class FileMapping {
    void *mapping;
    size_t num_bytes;
#ifdef _WIN32
    HANDLE file;
    HANDLE mapping_handle;
#else
    int file;
#endif

  public:
    // Map the file into memory
    FileMapping(const std::string &fname);
    FileMapping(FileMapping &&fm);
    ~FileMapping();
    FileMapping& operator=(FileMapping &&fm);

    FileMapping(const FileMapping &) = delete;
    FileMapping& operator=(const FileMapping&) = delete;

    const uint8_t* data() const;
    size_t nbytes() const;
  };

  template<typename T>
  class BasicStringView {
    const T *ptr;
    size_t count;

    public:
    BasicStringView() : ptr(nullptr), count(0) {}

    /* Create a typed view into a string. The count is in
     * number of elements of T in the view.
     */
    BasicStringView(const T* ptr, size_t count)
        : ptr(ptr), count(count)
    {}
    const T& operator[](const size_t i) const {
        return ptr[i];
    }
    const T* data() const {
        return ptr;
    }
    size_t size() const {
        return count;
    }
    const T* cbegin() const {
        return ptr;
    }
    const T* cend() const {
        return ptr + count;
    }
  };

  typedef BasicStringView<char> StringView;

  struct MappedFile {
    MappedFile() = default;
    MappedFile(const std::string &fname);

    MappedFile& operator=(MappedFile &&) = delete;

    MappedFile(const MappedFile &) = delete;
    MappedFile& operator=(const FileMapping&) = delete;

    size_t tellg() const;
    void seek(size_t p);
    size_t read(char *buf, size_t len);

    FileMapping fm;
    StringView view;
    size_t pos = 0;
  };

} // ::maui
