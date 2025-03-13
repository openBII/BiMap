// $Id$

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <sstream>

#include "simulatetrafficmanager.hpp"
#include "compair.hpp"

SimulateTrafficManager::SimulateTrafficManager( const Configuration &config, 
						const vector<Network *> & net, booksim* bs_ptr )
  : TrafficManager(config, net), _overall_runtime(0)
{
  _sample_period = config.GetInt( "sample_period" );    // Default: 1000
  _max_samples    = config.GetInt( "max_samples" );     // Default: 10
  _warmup_periods = config.GetInt( "warmup_periods" );  // Default: 3

  vector<string> workload = config.GetStrArray("workload");
  workload.resize(_classes, workload.back());
  _workload.resize(_classes);
  _bs_ptr = bs_ptr;
  for(int c = 0; c < _classes; ++c) {
    Workload * wl = Workload::New(workload[c], _nodes, &config, bs_ptr);
    assert(wl);
    _workload[c] = wl;
  }
}

SimulateTrafficManager::~SimulateTrafficManager( )
{
  for(int c = 0; c < _classes; ++c) {
    delete _workload[c];
  }
}

void SimulateTrafficManager::_Inject( )
{
  for(int c = 0; c < _classes; ++c) {
    Workload * const wl = _workload[c]; // *-workload
    while(!wl->empty()) {
      int const source = wl->source();
      if(_partial_packets[c][source].empty()) {
        ++_requests_outstanding[c][source];
        ++_packet_seq_no[c][source];
        int const dest = wl->dest();
        int const size = wl->size();
        int const time = (_include_queuing == 1) ? wl->time() : _time;
        int const pid = _GeneratePacketCompAir(source, dest, size, c, time, wl->ca_info());
        wl->inject(pid);
      } else {
	      wl->defer();
      }
    }
    wl->advanceTime(); // _refill() in this ...
  }
}

void SimulateTrafficManager::_RetirePacket( Flit * head, Flit * tail )
{
  TrafficManager::_RetirePacket( head, tail );
  _workload[head->cl]->retire(head->pid);
}

void SimulateTrafficManager::_ResetSim( )
{
  TrafficManager::_ResetSim( );
  for(int c = 0; c < _classes; ++c) {
    _workload[c]->reset();
  }
}

bool SimulateTrafficManager::_SingleSim( )
{
  _sim_state = warming_up;
  if(_warmup_periods > 0) {
    cout << "[SimulateTrafficManager] Warming up..." << endl;
    while(_time < _warmup_periods * _sample_period) {
      _StepSim(); 
      if((_time % _sample_period) == 0) {
        UpdateStats();
        DisplayStats();
      }
    }
    _ClearStats();
    cout << "Warmup ends after " << _warmup_periods * _sample_period << " cycles." << endl;
  }
  _sim_state = running;  
  cout << "Beginning measurements..." << endl;
  while(!_Completed() && 
      ((_max_samples < 0) || 
      (_time < (_warmup_periods + _max_samples) * _sample_period))) {
      _StepSim();
    if((_time % _sample_period) == 0) {
      UpdateStats();
      DisplayStats();
    }
  }
  cout << "Completed measurements after " << _time << " cycles." << endl;
  _sim_state = draining;
  _drain_time = _time;
  return 1;
}

// Function to perform a single simulation stage
bool SimulateTrafficManager::_SingleSim_Stage()
{
  // Print the start time of the measurements and whether the simulation is completed
#ifdef TRACK_EJECT
  cout << "Measurements starting @" << _time  << " " <<  _Completed() << endl;
#endif
  // If the simulation is completed, refill the workload for each class that requires measurement
  if (_Completed()) {
    for (int c = 0; c < _classes; ++c) {
      if (_measure_stats[c]) _workload[c]->refill();
    }
  }
  
  // Continue the simulation until it is completed or the maximum number of samples is reached
  while(!_Completed() && 
        ((_max_samples < 0) || (_time < (_max_samples) * _sample_period)))
  {
    // Perform a single step of the simulation and check if any flits were ejected
    bool ejected = _StepSim();
    
    // Update and display the simulation statistics at regular intervals
    if ((_time % _sample_period) == 0) {
      UpdateStats();
      DisplayStats();
    }
    
    // If any flits were ejected, break the loop
    if (ejected) { break; }
  }
  
  // Print the end time of the measurements
#ifdef TRACK_EJECT
  cout << "Measurements ending @" << _time << endl;
#endif

  // Return true to indicate that the simulation stage was completed
  return 1;
}


bool SimulateTrafficManager::_Completed( )
{
  for(int c = 0; c < _classes; ++c) {
    if(_measure_stats[c] &&
       (!_workload[c]->completed() || !_measured_in_flight_flits[c].empty())) {
      return false;
    }
  }
  return true;
}

void SimulateTrafficManager::_UpdateOverallStats()
{
  TrafficManager::_UpdateOverallStats();
  _overall_runtime += (_drain_time - _reset_time);
}
  
string SimulateTrafficManager::_OverallStatsHeaderCSV() const
{
  ostringstream os;
  os << TrafficManager::_OverallStatsHeaderCSV()
     << ',' << "runtime";
  return os.str();
}

string SimulateTrafficManager::_OverallClassStatsCSV(int c) const
{
  ostringstream os;
  os << TrafficManager::_OverallClassStatsCSV(c)
     << ',' << (double)_overall_runtime / (double)_total_sims;
  return os.str();
}

void SimulateTrafficManager::_DisplayClassStats(int c, ostream & os) const
{
  _workload[c]->printStats(os);
  TrafficManager::_DisplayClassStats(c, os);
}

void SimulateTrafficManager::_DisplayOverallClassStats(int c, ostream & os) const
{
  TrafficManager::_DisplayOverallClassStats(c, os);
  os << "Overall workload runtime = " << (double)_overall_runtime / (double)_total_sims
     << " (" << _total_sims << " samples)" << endl;
}

bool SimulateTrafficManager::_StepSim( )
{
  // Deadlock Detection
  bool flits_in_flight = false;

  // Merged Flits Detection
  vector<int> merged_flits_vec;
  for ( int subnet = 0; subnet < _subnets; ++subnet ) {
    vector<int> merged_flits_local = _net[subnet]->GetMergedFlits();
    for ( auto item : merged_flits_local )  merged_flits_vec.push_back(item);
  }

  // Eliminate merged flits from _total_in_flight_flits & _measured_in_flight_flits
  for (int c = 0; c < _classes; ++c) {
    bool class_not_empty = !_total_in_flight_flits[c].empty();
    if (class_not_empty) {
      for ( auto merged_flit_id : merged_flits_vec ) {
        if (_total_in_flight_flits[c].find(merged_flit_id) != _total_in_flight_flits[c].end()) {
          // cout << "[SimulateTrafficManager] Class " << c << " try to merge " << merged_flit_id << endl;
          if(_total_in_flight_flits[c][merged_flit_id]->record) {
            _measured_in_flight_flits[c].erase(merged_flit_id);
          }
          _total_in_flight_flits[c].erase(merged_flit_id);
        }
      }
    }
    flits_in_flight |= !_total_in_flight_flits[c].empty();
  }
  if ( flits_in_flight && (_deadlock_timer++ >= _deadlock_warn_timeout) ) {
    _deadlock_timer = 0;
    cout << "WARNING: Possible network deadlock." << endl;
  }

  bool ejected = false;
  vector< map<int, Flit *> > flits(_subnets);
  // For all subnets ...
  for ( int subnet = 0; subnet < _subnets; ++subnet ) {
    // For all nodes ...
    for ( int n = 0; n < _nodes; ++n ) {

      // Flit Eject
      Flit* const f = _net[subnet]->ReadFlit( n );
      if (f) {
        if (f->watch) {
          *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Ejecting flit " << f->id
              << " (packet " << f->pid << ")" 
              << " from VC " << f->vc
              << "." << endl;
        }
      #ifdef TRACK_EJECT
        if (true) {
          cout << f->ctime << "-" 
              << GetSimTime() << " | "
              << "subnet " << subnet << " | "
              << "node " << n << " | "
              << "Ejecting flit " << f->id
              << " (packet " << f->pid << ")" 
              << " from VC " << f->vc
              << " as tail " << f->tail << endl;
        }
      #endif
        flits[subnet].insert(make_pair(n, f)); // < node, flits >
        if ((_sim_state == warming_up) || (_sim_state == running)) {
          ++_accepted_flits[f->cl][n];
          if (f->tail) {
            ++_accepted_packets[f->cl][n];
            // Eject to Output FIFO
            _bs_ptr->o_fifo.enqueue(f->src, f->dest, f->ctime, GetSimTime(), -1);
            ejected = true;
          }
        }
      }
      // Credit-Based Flow Control
      Credit * const c = _net[subnet]->ReadCredit( n );
      if ( c ) {
        #ifdef TRACK_FLOWS
        for(set<int>::const_iterator iter = c->vc.begin(); iter != c->vc.end(); ++iter) {
          int const vc = *iter;
          assert(!_outstanding_classes[n][subnet][vc].empty());
          int cl = _outstanding_classes[n][subnet][vc].front();
          _outstanding_classes[n][subnet][vc].pop();
          assert(_outstanding_credits[cl][subnet][n] > 0);
          --_outstanding_credits[cl][subnet][n];
        }
        #endif
        _buf_states[n][subnet]->ProcessCredit(c);
        c->Free();
      }
    }
    _net[subnet]->ReadInputs( );
  }
  
  // Inject New Generated Flits
  for ( int subnet = 0; subnet < _subnets; ++subnet ) {
    for ( auto item : _net[subnet]->GetGeneratedFlits() ) {
      
      int source = item.source;
      int dest = item.dest;
      int size = item.size;
      int cl = item.cl;
      int time = item.time;
      comp_air_info ca_info = item.ca_info;
      int const pid = _GeneratePacketCompAir(source, dest, size, cl, time, ca_info);
      // printf("[INJ] Inject flit %d (%d->%d) at %d (x,y,op)=(%d,%d,%d) => w[%d]\n", pid, source, dest, time, 
      //                                                               ca_info.x_0, ca_info.y_0, ca_info.op_0, cl);
    }
  }

  // Inject pkts
  if ( !_empty_network ) { _Inject(); }

  // Inject flits
  for(int subnet = 0; subnet < _subnets; ++subnet) {
    for(int n = 0; n < _nodes; ++n) {
      Flit * f = NULL;
      BufferState * const dest_buf = _buf_states[n][subnet];
      int const last_class = _last_class[n][subnet];
      int class_limit = _classes;

      if (_hold_switch_for_packet) {
        list<Flit *> const & pp = _partial_packets[last_class][n];
        if(!pp.empty() && !pp.front()->head && 
          !dest_buf->IsFullFor(pp.front()->vc)) {
          f = pp.front();
          assert(f->vc == _last_vc[n][subnet][last_class]);
          // if we're holding the connection, we don't need to check that class 
          // again in the for loop
          --class_limit;
        }
      }

      for (int i = 1; i <= class_limit; ++i) {
        int const c = (last_class + i) % _classes;
        if(_subnet[c] != subnet) { continue; }
        list<Flit *> const & pp = _partial_packets[c][n];
        if(pp.empty()) { continue; }
        Flit * const cf = pp.front();
        assert(cf);
        assert(cf->cl == c);
        if (f && (f->pri >= cf->pri)) {
          continue;
        }
        if (cf->head && cf->vc == -1) { 
          // Find first available VC
          OutputSet route_set;
          _rf(NULL, cf, -1, &route_set, true);
          set<OutputSet::sSetElement> const & os = route_set.GetSet();
          assert(os.size() == 1);
          OutputSet::sSetElement const & se = *os.begin();
          assert(se.output_port == -1);
          int vcBegin = se.vc_start;
          int vcEnd = se.vc_end;
          int vc_count = vcEnd - vcBegin + 1;
          if(_noq) {
            assert(_lookahead_routing);
            const FlitChannel * inject = _net[subnet]->GetInject(n);
            const Router * router = inject->GetSink();
            assert(router);
            int in_channel = inject->GetSinkPort();

            // NOTE: Because the lookahead is not for injection, but for the 
            // first hop, we have to temporarily set cf's VC to be non-negative 
            // in order to avoid seting of an assertion in the routing function.
            cf->vc = vcBegin;
            _rf(router, cf, in_channel, &cf->la_route_set, false);
            cf->vc = -1;

            if (cf->watch) {
              *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Generating lookahead routing info for flit " << cf->id
              << " (NOQ)." << endl;
            }
            set<OutputSet::sSetElement> const sl = cf->la_route_set.GetSet();
            assert(sl.size() == 1);
            int next_output = sl.begin()->output_port;
            vc_count /= router->NumOutputs();
            vcBegin += next_output * vc_count;
            vcEnd = vcBegin + vc_count - 1;
            assert(vcBegin >= se.vc_start && vcBegin <= se.vc_end);
            assert(vcEnd >= se.vc_start && vcEnd <= se.vc_end);
            assert(vcBegin <= vcEnd);
          }
          if(cf->watch) {
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                << "Finding output VC for flit " << cf->id
                << ":" << endl;
          }
          for(int i = 1; i <= vc_count; ++i) {
            int const lvc = _last_vc[n][subnet][c];
            int const vc =  (lvc < vcBegin || lvc > vcEnd) ? 
                vcBegin : 
                (vcBegin + (lvc - vcBegin + i) % vc_count);
            assert((vc >= vcBegin) && (vc <= vcEnd));
            if (!dest_buf->IsAvailableFor(vc)) {
              if (cf->watch) {
                *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                    << "  Output VC " << vc << " is busy." << endl;
              }
            } else {
              if (dest_buf->IsFullFor(vc)) {
                if (cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                      << "  Output VC " << vc << " is full." << endl;
                }
              } else {
                if(cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                      << "  Selected output VC " << vc << "." << endl;
                }
                cf->vc = vc;
                break;
              }
            }
          }
        }
        if (cf->vc == -1) {
          if (cf->watch) {
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                << "No output VC found for flit " << cf->id
                << "." << endl;
          }
        } else {
          if (dest_buf->IsFullFor(cf->vc)) {
            if (cf->watch) {
              *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                << "Selected output VC " << cf->vc
                << " is full for flit " << cf->id
                << "." << endl;
            }
          } else {
            f = cf;
          }
        }
      }
      if (f) {
        int const c = f->cl;     
        if(f->head) {
          if (_lookahead_routing) {
            if(!_noq) {
              const FlitChannel * inject = _net[subnet]->GetInject(n);
              const Router * router = inject->GetSink();
              assert(router);
              int in_channel = inject->GetSinkPort();
              _rf(router, f, in_channel, &f->la_route_set, false);
              if (f->watch) {
                *gWatchOut << GetSimTime() << " | "
                    << "node" << n << " | "
                    << "Generating lookahead routing info for flit " << f->id
                    << "." << endl;
              }
            } else if (f->watch) {
              *gWatchOut << GetSimTime() << " | "
                << "node" << n << " | "
                << "Already generated lookahead routing info for flit " << f->id
                << " (NOQ)." << endl;
            }
          } else {
            f->la_route_set.Clear();
          }
          dest_buf->TakeBuffer(f->vc);
          _last_vc[n][subnet][c] = f->vc;
        }
        _last_class[n][subnet] = c;
        _partial_packets[c][n].pop_front();

#ifdef TRACK_FLOWS
        ++_outstanding_credits[c][subnet][n];
        _outstanding_classes[n][subnet][f->vc].push(c);
#endif

        dest_buf->SendingFlit(f);
        if (_pri_type == network_age_based) {
          f->pri = numeric_limits<int>::max() - _time;
          assert(f->pri >= 0);
        }
        if (f->watch) {
          *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Injecting flit " << f->id
              << " into subnet " << subnet
              << " at time " << _time
              << " with priority " << f->pri
              << " (packet " << f->pid
              << ", class = " << c
              << ", src = " << f->src 
              << ", dest = " << f->dest
              << ")." << endl;
          *gWatchOut << *f;
        }
        f->itime = _time;
        // Pass VC "back"
        if (!_partial_packets[c][n].empty() && !f->tail) {
          Flit* const nf = _partial_packets[c][n].front();
          nf->vc = f->vc;
        }
        if ((_sim_state == warming_up) || (_sim_state == running)) {
          ++_sent_flits[c][n];
          if(f->head) {
            ++_sent_packets[c][n];
          }
	      }
      #ifdef TRACK_FLOWS
	      ++_injected_flits[c][n];
      #endif
	      _net[subnet]->WriteFlit(f, n);
      }	
    }
  }

  // Inject Credit
  for (int subnet = 0; subnet < _subnets; ++subnet) {
    for (int n = 0; n < _nodes; ++n) {
      map<int, Flit *>::const_iterator iter = flits[subnet].find(n);
      if(iter != flits[subnet].end()) {
        Flit * const f = iter->second;
        f->atime = _time;
        if (f->watch) {
          *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Injecting credit for VC " << f->vc 
              << " into subnet " << subnet 
              << "." << endl;
        }
        Credit * const c = Credit::New();
        c->vc.insert(f->vc);
        _net[subnet]->WriteCredit(c, n);
      #ifdef TRACK_FLOWS
        ++_ejected_flits[f->cl][n];
      #endif
        _RetireFlit(f, n);
      }
    }
    flits[subnet].clear();
    _net[subnet]->Evaluate( );
    _net[subnet]->WriteOutputs( );
  }
  ++_time;
  assert(_time);
  if (gTrace) cout << "TIME " << _time << endl;
  return ejected;
}

int SimulateTrafficManager::_GeneratePacketCompAir( int source, int dest, int size, int cl, int time, comp_air_info ca_info )
{
  assert(size > 0);
  assert((source >= 0) && (source < _nodes));
  assert((dest >= 0) && (dest < _nodes));

  int pid = _cur_pid++;
  assert(_cur_pid);

  bool watch = gWatchOut && (_packets_to_watch.count(pid) > 0);

  if(watch) {
    *gWatchOut << GetSimTime() << " | "
	       << "node" << source << " | "
	       << "Enqueuing packet " << pid
	       << " at time " << time
	       << "." << endl;
  }

  Workload * const wl = _workload[cl];
  #ifdef TRACK_EJECT
    cout << GetSimTime() << " | "
	       << "node " << source << " | "
         << "comp_air type " <<  wl->ca_info().type << " | "
         << "comp_air data " <<  wl->ca_info().data << " | "
	       << "Enqueuing packet " << pid
	       << " at time " << time
	       << "." << endl;
  #endif
  
  bool record = (((_sim_state == running) ||
		  ((_sim_state == draining) && (time < _drain_time))) &&
		 _measure_stats[cl]);

  for ( int i = 0; i < size; ++i ) {

    int id = _cur_id++;
    assert(_cur_id);

    Flit * f = Flit::New();
    f->id = id;
    f->pid = pid;
    f->watch = watch | (gWatchOut && (_flits_to_watch.count(f->id) > 0));
    f->src = source;
    f->dest = dest;
    f->last_dest = dest;
    f->ctime = time;
    f->record = record;
    f->cl = cl;
    f->head = (i == 0);
    f->tail = (i == (size-1));
    f->vc  = -1;
    f->ca_info = ca_info;

    switch(_pri_type) {
      case class_based:
        f->pri = _class_priority[cl];
        break;
      case age_based:
        f->pri = numeric_limits<int>::max() - time;
        assert(f->pri >= 0);
        break;
      case sequence_based:
        f->pri = numeric_limits<int>::max() - _packet_seq_no[cl][source];
        break;
      default:
        f->pri = 0;
    }
    assert(f->pri >= 0);

    _total_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    if(record) {
      _measured_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    }
    
    if(gTrace) {
      cout<<"New Flit "<<f->src<<endl;
    }

    if(f->watch) { 
      *gWatchOut << GetSimTime() << " | "
		  << "node" << source << " | "
		  << "Enqueuing flit " << f->id
		  << " (packet " << f->pid
		  << ") at time " << time
		  << "." << endl;
    }

    _partial_packets[cl][source].push_back(f);
  }
  return pid;
}