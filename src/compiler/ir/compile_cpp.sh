#!/bin/sh
protoc --cpp_out=. basic.proto
protoc --cpp_out=. task.proto
