#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <pybind11/pybind11.h>

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

int add (int i, int j) { return i + j; }

TrafficManager* trafficManager = NULL;
int GetSimTime() { return trafficManager->getTime(); }
class Stats;
Stats * GetStats( const std::string & name ) {
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
    trafficManager->Run_Until_Eject();
    trafficManager->Report();
    return true;
}

void booksim::prepare(char* config_file)
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
    trafficManager->Run_Init();
}

PYBIND11_MODULE(booksim2, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function which adds two numbers", py::arg("i"), py::arg("j"));
    py::class_<booksim>(m, "booksim")
        .def(py::init<>())
        .def("run", &booksim::run)
        .def("prepare", &booksim::prepare)
        .def("end", &booksim::end);
}
