// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#pragma once
#include <string>
#include <mutex>
#include <json/json.h>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <list>
#include <fstream>

#include "spdlog/spdlog.h"
#include "src/compiler/ir/asm.pb.h"
#include "src/runtime/io_streamer/interface.h"
#include "src/runtime/io_streamer/client.h"

using namespace std;

// template<typename T>
// class Singleton{
// public:
//     static T& get_instance(){
//         static T instance;
//         return instance;
//     }
//     virtual ~Singleton(){}
//     Singleton(const Singleton&)=delete;
//     Singleton& operator =(const Singleton&)=delete;
// protected:
//     Singleton(){}

// };

class Iiostreamer
{
protected:
    IChipArray *chipArray;
    Json::Value root;

public:
    virtual void bind(IChipArray *chipArray, string info_path) = 0;
    virtual void inject(int step_group_id, int phase_group_id, int phase_id) = 0;
};

class IOstreamer : public Iiostreamer
{
public:
    // 在每个phase的开始打入数据
    iostreamer::SingleThreadClient client;
    map<int, list<codeIR::Config>> phase2Blocks;

    void divide()
    {
        ifstream is("style.png", ifstream::in | ios::binary);

        is.seekg(0, is.end);
        int length = is.tellg();
        is.seekg(0, is.beg);
        char * buffer = new char[length];
        is.read(buffer, length);
    }


    
    void inject(int phase_id)
    {
        // auto map = root["data_map"][to_string(step_group_id)][to_string(phase_group_id)];
        // auto keys = map.getMemberNames();
        // for (auto iter = keys.begin(); iter!=keys.end(); ++iter){
        //     auto phase_blocks = map[*iter][phase_id];
        //     for (auto block : phase_blocks){
        //         string filename = block["data"].asString();
        //         int start = block["start"].asInt();
        //         auto array =  root["core_map"][*iter];
        //         int chipx = array[0].asInt();
        //         int chipy = array[1].asInt();
        //         int corex = array[2].asInt();
        //         int corey = array[3].asInt();
        //         inject(chipx, chipy, corex, corey, start, filename);
        //         spdlog::get("console")->debug("data injected: sg{} pg{} p{} {}{}{}{} addr:{}{}", step_group_id, phase_group_id, phase_id, chipx, chipy, corex, corey, start, filename.c_str());
        //     }
        // }
        for (auto block : phase2Blocks[phase_id])
        {
            inject(block.chip_IDX(), block.chip_IDY(), block.core_IDX, block.core_IDY(), block.start());
        }
    }

    void inject(int chipx, int chipy, int corex, int corey, int start)
    {
        iostreamer::Request req;
        req.set_addr(start);
        client.bind(chipArray->getChip(chipx, chipy)->getCore(corex, corey));
        client.do_request(req);
        // for (int i = 0; i < (fsize+3)/4; ++i){
        //     cout<<"Chip("<<chipx<<","<<chipy<<") core("<<corex<<","<<corey<<")"<<hex<<setw(4)<<" addr:0x"<<start+i<<"\t: "<<setw(8)<<setfill('0')<<(chipArray->getChip(chipx, chipy)->getCore(corex, corey)->getMemory(start+i))<<dec<<endl;
        // }
    }
    void bind(IChipArray *chipArray, string info_path)
    {
        this->chipArray = chipArray;

        ifstream ifs(info_path);
        Json::Reader reader;
        if (!reader.parse(ifs, root, false))
        {
            cout << "reader parse error: " << strerror(errno) << endl;
            assert(false);
        }
    }

    bool parseConfigFile(const string &filename)
    {
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        fstream input(filename, ios::in | ios::binary);
        codeIR::Config config;

        if (!input)
        {
            cout << "Open case proto file error" << endl;
            return -1;
        }
        if (filename.substr(filename.rfind("."), filename.length()) == "txt"){
            //asm.txt
            if (!config.ParseFromString(&input))
            {
                cout << "Parse proto file error" << endl;
                return -1;
            }
        }else{
            //asm
            if (!config.ParseFromIstream(&input))
            {
                cout << "Parse proto file error" << endl;
                return -1;
            }
        }

        
        
        auto ioblocks = config.ioblocks();
        for (auto block : ioblocks)
        {
            for (auto phase : block.phases())
            {
                if (phase2Blocks.end() == phase2Blocks.find(phase))
                {
                    phase2Blocks[phase] = list<int>();
                }
                phase2Blocks[phase].push_back(block);
            }
        }
    }
}
}
}
;
