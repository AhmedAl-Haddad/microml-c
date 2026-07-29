CC      = gcc
CFLAGS  = -Wall -Wextra -Isrc/include
TARGET  = ml
BUILD   = build

SRCS    = src/main.c \
          src/arena.c \
          src/platform/linux.c \
          src/platform/win32.c

OBJS    = $(patsubst src/%.c, $(BUILD)/%.o, $(SRCS))

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile — create the output directory before compiling
$(BUILD)/%.o: src/%.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Remove build artifacts
clean:
	rm -rf $(BUILD) $(TARGET)

.PHONY: all clean
