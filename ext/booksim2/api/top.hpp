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
    void inject_comp_air(   int type,   float data, int t_inject,
                            int src,    int iter_tag, int pkg_size,
                            int x_0,    int y_0,    int op_0, 
                            int x_1,    int y_1,    int op_1, 
                            int x_2,    int y_2,    int op_2, 
                            int x_3,    int y_3,    int op_3 );
    int eject();
    std::string eject_all_print();
    bool run_async();
    bool run_sync();
    void end();
};

#endif