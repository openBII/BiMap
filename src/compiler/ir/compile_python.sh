#!/bin/sh
protoc --python_out=. basic.proto
protoc --python_out=. task.proto
