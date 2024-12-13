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

namespace py = pybind11;

int add (int i, int j) { return i + j; }

TrafficManager* trafficManager = NULL;

int GetSimTime() { return trafficManager->getTime(); }

class Stats;
Stats * GetStats( const std::string & name ) {
    Stats* test =  trafficManager->getStats(name);
    if (test == 0) {
        cout << "warning statistics " << name << " not found" << endl;
    }
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

bool Simulate( BookSimConfig const & config ) 
{
    vector<Network *> net;
    int subnets = config.GetInt("subnets");
    /*
     * To include a new network, must register the network here
     * add an else if statement with the name of the network
     */
    net.resize(subnets);
    for (int i = 0; i < subnets; ++i) {
        ostringstream name;
        name << "network_" << i;
        net[i] = Network::New( config, name.str() );
    }
    assert(trafficManager == NULL);
    trafficManager = TrafficManager::New( config, net ) ;
    double total_time; /* Amount of time we've run */
    struct timeval start_time, end_time; /* Time before/after user code */
    total_time = 0.0;
    gettimeofday(&start_time, NULL);
    bool result = trafficManager->Run() ;
    gettimeofday(&end_time, NULL);
    total_time = ((double)(end_time.tv_sec) + (double)(end_time.tv_usec)/1000000.0)
                - ((double)(start_time.tv_sec) + (double)(start_time.tv_usec)/1000000.0);
    cout << "Total run time " << total_time << endl;
    for (int i=0; i<subnets; ++i) {
        ///Power analysis
        if(config.GetInt("sim_power") > 0){
            Power_Module pnet(net[i], config);
            pnet.run();
        }
        delete net[i];
    }
    delete trafficManager;
    trafficManager = NULL;
    return result;
}

int run_sim( char* config_file )
{
    BookSimConfig config;
    config.ParseFile(config_file);

    InitializeRoutingMap( config );
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
    bool result = Simulate(config);
    return result ? -1 : 0;
}

PYBIND11_MODULE(booksim2, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function which adds two numbers", py::arg("i"), py::arg("j"));
    m.def("run_sim", &run_sim, "run booksim, simulator", py::arg("config_file"));
}
