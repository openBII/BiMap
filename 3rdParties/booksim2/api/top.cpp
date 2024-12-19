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
Stats * GetStats( const std::string & name )
{
    Stats* test =  trafficManager->getStats(name);
    return test;
}

/* printing activity factor */
bool gPrintActivity;
int gK; //radix
int gN; //dimension
int gC; //concentration

// generate nocviewer trace
bool gTrace;
int gNodes;
ostream* gWatchOut;

/////////////////////////////////////////////////////////////////////////////

void booksim::end()
{
    ///Power analysis
    trafficManager->Report();
    pthread_exit(nullptr); // Threads shutdown!
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

bool booksim::run()
{
    trafficManager->TransRun();
    return true;
}

void booksim::init(char* config_file)
{
    // Initialize the Simulator
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
    trafficManager = TrafficManager::New(config, this->net);
    trafficManager->RunInit();

    // Run the simulator with a seperate thread
    std::thread sim_thread(&booksim::run, this);
    sim_thread.detach();
    std::cout << "Init end, simulation thread detach ..." << std::endl;
}

void booksim::inject(int src, int dst, int t_inject)
{
    this->i_fifo.enqueue(src, dst, t_inject, -1);
}

int booksim::eject()
{
    return this->o_fifo.dequeue();
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
        .def("run", &booksim::run)
        .def("init", &booksim::init)
        .def("inject", &booksim::inject)
        .def("eject", &booksim::eject)
        .def("end", &booksim::end);
}
