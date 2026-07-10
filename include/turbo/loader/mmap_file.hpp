#pragma once
#include <iostream>
#include <string>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace turbo {

class MmapFile {
private:
    void* data_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;

public:
    // Delete copy constructor and assignment to prevent double-freeing memory
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    // Constructor: Opens the file and maps it into virtual memory
    explicit MmapFile(const std::string& filepath) {
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        struct stat sb;
        if (fstat(fd_, &sb) == -1) {
            close(fd_);
            throw std::runtime_error("Failed to stat file: " + filepath);
        }
        size_ = sb.st_size;

        data_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap file: " + filepath);
        }
    }

    // Destructor: Safely unmaps memory and closes the file descriptor
    ~MmapFile() {
        if (data_ != MAP_FAILED && data_ != nullptr) {
            munmap(data_, size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }

    // Accessors
    const void* data() const { return data_; }
    size_t size() const { return size_; }
};

} // namespace turbo
