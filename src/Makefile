FILENAME = FluidSim

CC = gcc
CXX = g++
NVCC = nvcc

# Include path
IDIR = include/vcpkg/installed/x64-linux/include
IDIR2 = include/utils
IDIR_CUDA = include/lib
BASE_DIR = $(shell pwd)

IMGUI_DIR = include/imgui

# Libraries path
LDIR = include/vcpkg/installed/x64-linux/lib

# Build directory
BUILD_DIR = build

# Compiler flags
NVCCFLAGS = -g -O3 -rdc=true -std=c++17 -arch=sm_75 -I$(IDIR) -I$(IDIR_CUDA) -I$(BASE_DIR)
CXXFLAGS = -O3 -std=c++17 -MMD -MP -I$(IDIR) -I$(IDIR2) -I$(IDIR_CUDA) -I${IMGUI_DIR} -I$(BASE_DIR)

# Linker flags
LDFLAGS = -L$(LDIR) -lglfw3 -lglad -lassimp -lz -lminizip -lkubazip -lpoly2tri -lpolyclipping -ldraco -lpugixml 

VCPKG_SRC = include/vcpkg/buildtrees/glad/x64-linux-rel/src/

# Source files
CUDA_SOURCES = $(IDIR_CUDA)/cuda_utils.cu Grid.cu Particle.cu 
CPP_SOURCES = FluidSim.cpp \
              $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp \
              $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp \
              $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

# Object files with build directory prefixes
CUDA_OBJECTS = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
CPP_OBJECTS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))

# Dependency files
DEPS = $(CPP_OBJECTS:.o=.d)

TARGET = $(FILENAME).out

.PHONY: all
all: directories $(TARGET)

# Create build directories
.PHONY: directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$(IDIR_CUDA)
	@mkdir -p $(BUILD_DIR)/$(IMGUI_DIR)/backends

# Compile CUDA files
$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Compile C++ files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CXXFLAGS) -c $< -o $@

# Link everything into the final executable
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_OBJECTS) $(CPP_OBJECTS) $(LDFLAGS) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)