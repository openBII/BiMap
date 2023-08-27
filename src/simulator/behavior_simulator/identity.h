// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef IDENTITY_H
#define IDENTITY_H

#include <assert.h>
#include <functional>
#include <string>

using namespace std;

#define COREINCHIPROW 16
#define COREINCHIPCOLUMN 10

class IDUtil;

class ID
{
 public:
    // enum ID_TYPE：ID的类型
    enum ID_TYPE {
        CHIP_ARRAY,
        CHIP,
        CORE,
        RESOURCE,
        DATA_BLOCK,
        FPGA,
        INVALID,
    };

    // ID的缺省构造函数，构造出的ID类型为INVALID
    ID() : _type(INVALID), _id("") {}

    // make_chip_array_id：调用该函数，是唯一生成CHIP_ARRAY类型ID的方式
    static ID make_chip_array_id(const string &id) {
        return ID(CHIP_ARRAY, id);
    }
    // make_chip_id：调用该函数，是唯一生成CHIP类型ID的方式
    static ID make_chip_id(ID chip_array_id, uint32_t x, uint32_t y) {
        assert(chip_array_id._type == CHIP_ARRAY);
        return ID(CHIP, to_string(x) + "_" + to_string(y)) + chip_array_id;
    }
    // make_core_id：调用该函数，是唯一生成CORE类型ID的方式
    static ID make_core_id(ID chip_id, uint32_t x, uint32_t y) {
        assert(chip_id._type == CHIP);
        return ID(CORE, to_string(x) + "_" + to_string(y)) + chip_id;
    }
    // make_resource_id：调用该函数，是唯一生成RESOURCE类型ID的方式
    static ID make_resource_id(ID core_id, const string &id) {
        assert(core_id._type == CORE);
        return ID(RESOURCE, id) + core_id;
    }

    static ID make_data_block_id(ID core_id, const string &id) {
        assert(core_id._type == CORE);
        return ID(DATA_BLOCK, id) + core_id;
    }

    static ID make_fpga_id() { return ID(FPGA, "FPGA"); }

 private:
    // ID类的私有构造函数，确保外部只能使用ID类提供的构造方式，来保证ID系统的正确运行
    ID(ID_TYPE id_type, const string &id) : _type(id_type), _id(id) {}

    ID operator+(const ID &id) {
        _id += "." + id._id;
        return *this;
    }

 public:
    //==操作符重载：判断另一个ID的ID类型与该ID的类型是否相同
    bool operator==(const ID &id) const {
        return _id.compare(id._id) == 0 ? _type == id._type : false;
    }

    //<操作符重载：用于数据结构map使用
    bool operator<(const ID &id) const { return _id < id._id; }

    // get_chip_array_id：获取该ID所属的chip_array的ID
    ID get_chip_array_id() const {
        if (_type == CHIP_ARRAY) {
            return *this;
        } else if (_type == CHIP) {
            return ID(ID::CHIP_ARRAY, _id.substr(_id.find_first_of('.') + 1));
        } else if (_type == CORE) {
            string chip_id_str = _id.substr(_id.find_first_of('.') + 1);
            return ID(ID::CHIP_ARRAY,
                      chip_id_str.substr(chip_id_str.find_first_of('.') + 1));
        } else if (_type == RESOURCE) {
            string core_id_str = _id.substr(_id.find_first_of('.') + 1);
            string chip_id_str =
                core_id_str.substr(core_id_str.find_first_of('.') + 1);
            return ID(ID::CHIP_ARRAY,
                      chip_id_str.substr(chip_id_str.find_first_of('.') + 1));
        }
        return ID(INVALID, "");
    }
    // get_chip_id：获取该ID所属的chip的ID，当本ID类型不等于RESOURCE和CORE和CHIP时，则返回无效ID
    ID get_chip_id() const {
        if (_type == CHIP) {
            return *this;
        } else if (_type == CORE) {
            return ID(ID::CHIP, _id.substr(_id.find_first_of('.') + 1));
        } else if (_type == RESOURCE) {
            string core_id_str = _id.substr(_id.find_first_of('.') + 1);
            return ID(ID::CHIP,
                      core_id_str.substr(core_id_str.find_first_of('.') + 1));
        }
        return ID(INVALID, "");
    }
    // get_core_id：获取该ID所属的core的ID，当本ID类型不等于RESOURCE和CORE时，则返回无效ID
    ID get_core_id() const {
        if (_type == CORE) {
            return *this;
        } else if (_type == RESOURCE || _type == DATA_BLOCK) {
            return ID(ID::CORE, _id.substr(_id.find_first_of('.') + 1));
        }
        return ID(INVALID, "");
    }
    // get_resource_id：获取本ID所属的resource的ID，当本ID类型不等于RESOURCE时，则返回无效ID
    ID get_resource_id() const {
        if (_type == RESOURCE) {
            return *this;
        }
        return ID(INVALID, "");
    }

    // is_chip_array：判断本ID是否是chip_array的ID
    bool is_chip_array() const { return _type == CHIP_ARRAY; }
    // is_chip：判断本ID是否是is_chip的ID
    bool is_chip() const { return _type == CHIP; }
    // is_core：判断本ID是否是is_core的ID
    bool is_core() const { return _type == CORE; }
    // is_resource：判断本ID是否是is_resource的ID
    bool is_resource() const { return _type == RESOURCE; }
    // is_packet_header：判断本ID是否是一个包头数据块的ID
    bool is_packet_header() const { return _id.find("packet_") == 0; }

    // get_module_id_str：获取本ID所属类型所对应部分的id字符串
    string get_module_id_str() const {
        return _id.substr(0, _id.find_first_of('.'));
    }

    // get_id_str：获取本ID的完整id字符串
    string get_id_str() const { return _id; }

    // valid：判断ID对象是否是一个有效的ID
    bool valid() const { return _type != INVALID; }

    // get_core_xy：通过ID获取core的坐标，仅当ID类型为CORE或RESOURCE时可以调用，其他类型调用会assert报错
    pair<uint32_t, uint32_t> get_core_xy() const;

    // get_chip_xy：通过ID获取chip的坐标，仅当ID类型为CHIP或CORE或RESOURCE时可以调用，其他类型调用会assert报错
    pair<uint32_t, uint32_t> get_chip_xy() const;

 private:
    //_type：id类型
    ID_TYPE _type;
    //_id：id字符串
    string _id;
    friend IDUtil;
};

// 重载打印id函数
inline ostream &operator<<(ostream &os, const ID &id) {
    os << id.get_id_str();
    return os;
}

// 获取ID的哈希值，用于数据结构unordered_map等
struct Hash_ID {
    std::size_t operator()(const ID &id) const {
        return hash<string>()(id.get_id_str());
    }
};

// ID类的辅助类
class IDUtil
{
 public:
    //
    static ID find_offset_core_id(ID core_id, int32_t dx, int32_t dy);
};

#endif  // IDENTIfind_offset_core_id：通过ID，以及给定的偏移量，获取到指定core坐标的ID，仅当core_id的类型等于CORE时有效，其余会assert报错TY_H
