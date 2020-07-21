CC := g++
TARGET := main
CPPFLAGS :=-std=c++14 -MMD -MP
LDFLAGS  := -pthread
LDLIBS   := 

# Basler Pylon
CPPFLAGS   += $(shell $(PYLON_ROOT)/bin/pylon-config --cflags)
LDFLAGS    += $(shell $(PYLON_ROOT)/bin/pylon-config --libs-rpath)
LDLIBS     += $(shell $(PYLON_ROOT)/bin/pylon-config --libs)

# OpenCV
CPPFLAGS += `pkg-config --libs opencv`
LDLIBS += `pkg-config --libs opencv`

SRC_DIR := src
OBJ_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY : all
all: $(TARGET)

.PHONY : debug
debug: CPPFLAGS += -DDEBUG -g
debug: clean
debug: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CPPFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

.PHONY : clean
clean:
	rm $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS)