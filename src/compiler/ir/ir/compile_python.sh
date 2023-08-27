#!/bin/sh
protoc --python_out=. basic.proto
protoc --python_out=. data.proto
protoc --python_out=. code.proto
protoc --python_out=. mapping.proto
protoc --python_out=. task.proto
protoc --python_out=. asm.proto
protoc --python_out=. paint.proto
protoc --python_out=. msg.proto
