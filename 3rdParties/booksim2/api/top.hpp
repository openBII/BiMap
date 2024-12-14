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

class booksim
{
private:
    int subnets;
    vector<Network *> net;
    BookSimConfig config;
public:
    booksim(){}
    ~booksim(){}
    void prepare(char* config_file);
    bool run();
    void end();
};

#endif