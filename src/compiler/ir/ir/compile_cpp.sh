#!/bin/sh
protoc --cpp_out=. basic.proto
protoc --cpp_out=. data.proto
protoc --cpp_out=. code.proto
protoc --cpp_out=. mapping.proto
protoc --cpp_out=. task.proto
protoc --cpp_out=. asm.proto
protoc --cpp_out=. msg.proto
protoc --cpp_out=. paint.proto
