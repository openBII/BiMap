#ifndef __TOP_HPP__
#define __TOP_HPP__

#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <pybind11/pybind11.h>

#include "module.hpp"
#include "config_utils.hpp"
#include "network.hpp"
#include "flit.hpp"
#include "buffer_state.hpp"
#include "stats.hpp"
#include "routefunc.hpp"
#include "outputset.hpp"
#include "booksim_config.hpp"
#include "lockfree_queue.hpp"

class booksim
{
private:
    bool sync;
    int subnets;
    vector<Network *> net;
    BookSimConfig config;
public:
    LockFreePktQueue i_fifo;
    LockFreePktQueue o_fifo;
    booksim(){}
    ~booksim(){}
    void init(char* config_file, bool sync);
    void inject(int src, int dst, int t_inject, int pkg_size);
    int eject();
    void eject_all_print();
    bool run_async();
    bool run_sync();
    void end();
};

#endif