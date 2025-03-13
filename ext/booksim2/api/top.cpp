#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <pybind11/pybind11.h>
#include <thread>
#include <unistd.h>

#include "booksim.hpp"
#include "routefunc.hpp"
#include "traffic.hpp"
#include "booksim_config.hpp"
#include "trafficmanager.hpp"
#include "random_utils.hpp"
#include "network.hpp"
#include "injection.hpp"
#include "power_module.hpp"
#include "top.hpp"

namespace py = pybind11;

TrafficManager* trafficManager = NULL;
int GetSimTime()
{ 
    return trafficManager->getTime(); 
}
class Stats;
Stats * GetStats(const std::string & name)
{
    Stats* test =  trafficManager->getStats(name);
    return test;
}

/* printing activity factor */
bool gPrintActivity;
int  gK, gN, gC;

// generate nocviewer trace
bool gTrace;
int gNodes;
ostream* gWatchOut;

/////////////////////////////////////////////////////////////////////////////

void booksim::end()
{
    /// Analysis
    if (this->sync) {
        trafficManager->RunDrain();
        trafficManager->Report();
    } else {
        trafficManager->Report();
        pthread_exit(nullptr);
    }
    for (int i = 0; i < this->subnets; ++i) {
        if (config.GetInt("sim_power") > 0) {
            Power_Module pnet(this->net[i], config);
            pnet.run();
        }
        delete this->net[i];
    }
    delete trafficManager;
    trafficManager = NULL;
}

bool booksim::run_async()
{
    trafficManager->RunAlways();
    return true;
}

bool booksim::run_sync()
{
    trafficManager->RunOnce();
    return true;
}

void booksim::init(char* config_file, bool sync)
{
    // Initialize the Simulator
    this->sync = sync;
    config.ParseFile(config_file);
    InitializeRoutingMap(config);
    gPrintActivity = (config.GetInt("print_activity") > 0);
    gTrace = (config.GetInt("viewer_trace") > 0);
    string watch_out_file = config.GetStr("watch_out");
    if (watch_out_file == "") {
        gWatchOut = NULL;
    } else if (watch_out_file == "-") {
        gWatchOut = &cout;
    } else {
        gWatchOut = new ofstream(watch_out_file.c_str());
    }    
    this->subnets = config.GetInt("subnets");
    this->net.resize(this->subnets);
    for (int i = 0; i < this->subnets; ++i) {
        ostringstream name;
        name << "network_" << i;
        this->net[i] = Network::New( config, name.str() );
    }
    assert(trafficManager == NULL);
    trafficManager = TrafficManager::New(config, this->net, this);
    trafficManager->RunInit();

    // Run the simulator with a seperate thread
    if (!sync) {
        // Async ...
        std::thread sim_thread(&booksim::run_async, this);
        sim_thread.detach();
        std::cout << "Init end, simulation thread detach ..." << std::endl;
    } else {
        // Sync ...
        trafficManager->RunReady();
    }
}

void booksim::inject(int src, int dst, int t_inject, int pkg_size)
{
    this->i_fifo.enqueue(src, dst, t_inject, -1, pkg_size);
}

void booksim::inject_comp_air(  int type,   float data, int t_inject,
                                int src,    int iter_tag, int pkg_size,
                                int x_0,    int y_0,    int op_0, 
                                int x_1,    int y_1,    int op_1, 
                                int x_2,    int y_2,    int op_2, 
                                int x_3,    int y_3,    int op_3 )
{
    int edge_len = config.GetInt("k");
    // 0: scalar and 4: cover
    int t_i = t_inject;
    int t_e = -1;
    int dst = src + x_0 + y_0 * edge_len;
    // cout << "[inject_comp_air] pkt " << src << " -> " << dst << endl;
    if (dst < edge_len * edge_len && dst >= 0) {
        pkt comp_pkt(t_i, t_e, type, data, src, dst, pkg_size, iter_tag,
                x_0, y_0, op_0, x_1, y_1, op_1, x_2, y_2, op_2, x_3, y_3, op_3);
        this->i_fifo.enqueue_pkt(comp_pkt);
    } else {
        printf("Dst (%d) out-of range!!\n", dst);
    }
}

int booksim::eject()
{
    return this->o_fifo.dequeue();
}

std::string booksim::eject_all_print()
{
    std::string res = "";
    while (!this->o_fifo.is_empty()) {
        pkt p = this->o_fifo.dequeue_pkt();
        res += "Ti: " + std::to_string(p.t_inject) + " Te: " + std::to_string(p.t_eject) + " Src: " + std::to_string(p.addr_src) + " Dst: " + std::to_string(p.addr_dst) + " # ";
    }
    return res;
}

/////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(booksim2, m)
{
    m.doc() = "Booksim2 Python API (T-NoC Version)";
    py::class_<LockFreePktQueue>(m, "LockFreePktQueue")
        .def(py::init<>())
        .def("enqueue", &LockFreePktQueue::enqueue)
        .def("dequeue", &LockFreePktQueue::dequeue);
    py::class_<booksim>(m, "booksim")
        .def(py::init<>())
        .def("run_async", &booksim::run_async)
        .def("run_sync", &booksim::run_sync)
        .def("init", &booksim::init)
        .def("inject_comp_air", &booksim::inject_comp_air)
        .def("eject_all_print", &booksim::eject_all_print)
        .def("inject", &booksim::inject)
        .def("eject", &booksim::eject)
        .def("end", &booksim::end);
}
