// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "src/runtime/io_streamer/server.h"

int main(){
    iostreamer::IOstreamerServer server("7000");
    server.start();
}