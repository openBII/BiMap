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

/*flit.cpp
 *
 *flit struct is a flit, carries all the control signals that a flit needs
 *Add additional signals as necessary. Flits has no concept of length
 *it is a singluar object.
 *
 *When adding objects make sure to set a default value in this constructor
 */

#include "booksim.hpp"
#include "workload.hpp"
#include "flit.hpp"

stack<Flit *> Flit::_all;
stack<Flit *> Flit::_free;

ostream& operator<<( ostream& os, const Flit& f )
{
  os << "  Flit ID: " << f.id << " (" << &f << ")" 
     << " Packet ID: " << f.pid
     << " Class: " << f.cl 
     << " Head: " << f.head
     << " Tail: " << f.tail << endl;
  os << "  Source: " << f.src << "  Dest: " << f.dest << " Intm: "<<f.intm<<endl;
  os << "  Creation time: " << f.ctime << " Injection time: " << f.itime << " Arrival time: " << f.atime << " Phase: "<<f.ph<< endl;
  os << "  VC: " << f.vc << endl;
  return os;
}

Flit::Flit() 
{  
  Reset();
}  

void Flit::Reset() 
{  
  vc        = -1 ;
  cl        = -1 ;
  head      = false ;
  tail      = false ;
  ctime     = -1 ;
  itime     = -1 ;
  atime     = -1 ;
  id        = -1 ;
  pid       = -1 ;
  hops      = 0 ;
  watch     = false ;
  record    = false ;
  intm = 0;
  src = -1;
  dest = -1;
  pri = 0;
  intm =-1;
  ph = -1;
  data = 0;
}  

Flit * Flit::New() {
  Flit * f;
  if(_free.empty()) {
    f = new Flit;
    _all.push(f);
  } else {
    f = _free.top();
    f->Reset();
    _free.pop();
  }
  return f;
}

void Flit::UpdateDest(int node_id) {
  if (node_id == this->dest && this->ca_info.type >= 0) {
    int matched_router_id = MatchedDestID(node_id);
    if (matched_router_id >= 0) {
      if (matched_router_id == 0) {

        if (this->ca_info.op_1 >= 0) {
          int new_dest = this->dest + this->ca_info.x_1 + this->ca_info.y_1 * this->ca_info.edge_len;
        #ifdef TRACK_COMP_AIR
          printf("[RRC] Flit %d: Update dest at node_id %d go %d <0>\n", this->id, node_id, new_dest);
        #endif
          this->last_dest = this->dest;
          this->dest = new_dest;
        } else if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
        #ifdef TRACK_COMP_AIR
          printf("[RRC] Flit %d: Update dest at node_id %d dst %d <1>\n", this->id, node_id, new_dest);
        #endif
          this->last_dest = this->dest;
          this->dest = new_dest;
          this->ca_info.iter_tag -= 1;
        }

      } else if (matched_router_id == 1) {
        
        if (this->ca_info.op_2 >= 0) {
          int new_dest = this->dest + this->ca_info.x_2 + this->ca_info.y_2 * this->ca_info.edge_len;
        #ifdef TRACK_COMP_AIR
          printf("[RRC] Flit %d: Update dest at node_id %d go %d <2>\n", this->id, node_id, new_dest);
        #endif
          this->last_dest = this->dest;
          this->dest = new_dest;
        } else if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
        #ifdef TRACK_COMP_AIR
          printf("[RRC] Flit %d: Update dest at node_id %d dst %d <3>\n", this->id, node_id, new_dest);
        #endif
          this->last_dest = this->dest;
          this->dest = new_dest;
          this->ca_info.iter_tag -= 1;
        }

      } else if (matched_router_id == 2) {

        if (this->ca_info.op_3 >= 0) {
          int new_dest = this->dest + this->ca_info.x_3 + this->ca_info.y_3 * this->ca_info.edge_len;
        #ifdef TRACK_COMP_AIR
          printf("[RRC] Flit %d: Update dest at node_id %d go %d <4>\n", this->id, node_id, new_dest);
        #endif
          this->last_dest = this->dest;
          this->dest = new_dest;
        } else if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
        #ifdef TRACK_COMP_AIR
          printf("[RRC] Flit %d: Update dest at node_id %d dst %d <5>\n", this->id, node_id, new_dest);
        #endif
          this->last_dest = this->dest;
          this->dest = new_dest;
          this->ca_info.iter_tag -= 1;
        }

      } else if (matched_router_id == 3) {

        if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
        #ifdef TRACK_COMP_AIR
          printf("[RRC] Flit %d: Update dest at node_id %d dst %d <6>\n", this->id, node_id, new_dest);
        #endif
          this->last_dest = this->dest;
          this->dest = new_dest;
          this->ca_info.iter_tag -= 1;
        }
      }
    }
  }
}

int Flit::GetUpdatedDest(int node_id) const {
  int matched_router_id = MatchedDestID(node_id);
  if (node_id == this->dest && this->ca_info.type >= 0) {
    if (matched_router_id >= 0) {
      if (matched_router_id == 0) {
        if (this->ca_info.op_1 >= 0) {
          int new_dest = this->dest + this->ca_info.x_1 + this->ca_info.y_1 * this->ca_info.edge_len;
          return new_dest;
        } else if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
          return new_dest;
        }
      } else if (matched_router_id == 1) {
        
        if (this->ca_info.op_2 >= 0) {
          int new_dest = this->dest + this->ca_info.x_2 + this->ca_info.y_2 * this->ca_info.edge_len;
          return new_dest;
        } else if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
          return new_dest;
        }

      } else if (matched_router_id == 2) {

        if (this->ca_info.op_3 >= 0) {
          int new_dest = this->dest + this->ca_info.x_3 + this->ca_info.y_3 * this->ca_info.edge_len;
          return new_dest;
        } else if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
          return new_dest;
        }

      } else if (matched_router_id == 3) {

        if (this->ca_info.iter_tag > 0) {
          int new_dest = this->src + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
          return new_dest;
        }
      }
    }
  }
  // printf("[GetUpdatedDest] Flit %d: %d -> keep for %d <m:%d>\n", this->id, this->dest, node_id, matched_router_id);
  return -1;
}

int Flit::MatchedDestID(int node_id) const {
  int matched_router_id = -1;
  int base_router = this->src;
  if (base_router + this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len == node_id) {
    matched_router_id = 0;
  } else {
    base_router += this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
    if (base_router + this->ca_info.x_1 + this->ca_info.y_1 * this->ca_info.edge_len == node_id) {
      matched_router_id = 1;
    } else {
      base_router += this->ca_info.x_1 + this->ca_info.y_1 * this->ca_info.edge_len;
      if (base_router + this->ca_info.x_2 + this->ca_info.y_2 * this->ca_info.edge_len == node_id) {
        matched_router_id = 2;
      } else {
        base_router += this->ca_info.x_2 + this->ca_info.y_2 * this->ca_info.edge_len;
        if (base_router + this->ca_info.x_3 + this->ca_info.y_3 * this->ca_info.edge_len == node_id) {
          matched_router_id = 3;
        } else {
          matched_router_id = -1;
        }
      } 
    }
  }
  return matched_router_id;
}

int Flit::LastPos() const {
  if (this->ca_info.type >= 0 && this->ca_info.iter_tag == 0) {
    int base_pos = this->src;
    if (this->ca_info.op_0 >= 0) {
      base_pos += this->ca_info.x_0 + this->ca_info.y_0 * this->ca_info.edge_len;
      if (this->ca_info.op_1 >= 0) {
        base_pos += this->ca_info.x_1 + this->ca_info.y_1 * this->ca_info.edge_len;
        if (this->ca_info.op_2 >= 0) {
          base_pos += this->ca_info.x_2 + this->ca_info.y_2 * this->ca_info.edge_len;
          if (this->ca_info.op_3 >= 0) {
            base_pos += this->ca_info.x_3 + this->ca_info.y_3 * this->ca_info.edge_len;
            return base_pos;
          } else {
            return base_pos;
          }
        } else {
          return base_pos;
        }
      } else {
        return base_pos;
      }
    } else {
      return -1;
    }
  } else {
    return -1;
  }
}

void Flit::Free() {
  _free.push(this);
}

void Flit::FreeAll() {
  while(!_all.empty()) {
    delete _all.top();
    _all.pop();
  }
}
