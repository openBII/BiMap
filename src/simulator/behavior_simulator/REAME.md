`Context`类负责持有Memory和Noc以及原语的高层调度
```c++
class Context
{
public:
    shared_ptr<uint8_t> read(const ID &core_id, size_t address, size_t length) const;

    void execute(const ID &core_id, const shared_ptr<Primitive> &pi, uint32_t phase_num);

    void init_data_block(const DataBlock &block);

private:
    unique_ptr<VirtualMemory> _memory;
    unique_ptr<NoC> _network;
};
```

```c++
class Packet
{
public:
    struct Head
    {
        ID source_id;                    //数据包的源core的ID
        ID destination_id;               //数据包的目的core的ID
        bool broadcast_or_relay;         //是否多播或中继，即物理包头的Q位
        ROUTER::PACKET_TYPE packet_type; //包头的类型，多包/单包，即物理包头的T位
        size_t offset;                   //接收存储位置的偏移量，即物理包头的A位
        bool stop;                       //是否停止，即物理包头的P位
        uint32_t recv_end_phase;         //接收端在第几个phase接收
    };

    Head get_head() const;
    DataBlock get_data() const;
    Packet(Head header, DataBlock data);

private:
    Head _header;
    DataBlock _data;
};

class DataPacketUtil
{
public:
    //pack：将DataBlock根据router_parameter封装为Packet
    static vector<Packet> pack(const vector<DataBlock> &data, const shared_ptr<const Prim09_Parameter> &router_parameter);
    //pack：对Packet根据router_parameter重新封装，在多播或中继是调用，修改Packet中源坐标和目的坐标
    static vector<Packet> repack(const vector<Packet> &packets, const shared_ptr<const Prim09_Parameter> &router_parameter);
    //unpack：对Packet根据router_parameter解封装为DataBlock
    static vector<DataBlock> unpack(const vector<Packet> &packets, const shared_ptr<const Prim09_Parameter> &router_parameter);

protected:
    //calc_address：对于一对多包头类型，计算每个包中的偏移量A字段
    static size_t calc_address(size_t packet_num, size_t start, size_t const_num, size_t offset);
};
```



```c++
class IDUtil;
class ID
{
public:
    //enum ID_TYPE：ID的类型
    enum ID_TYPE
    {
        CHIP_ARRAY,
        CHIP,
        CORE,
        RESOURCE,
        INVALID,
    };

    //ID的缺省构造函数，构造出的ID类型为INVALID
    ID() : _type(INVALID), _id("") {}
    //ID 生成
    make_chip_array_id(id);
    make_chip_id(chip_array_id, x, y);
    make_core_id(chip_id, x, y);
    make_resource_id(core_id, id);

private:
    //ID类的私有构造函数，确保外部只能使用ID类提供的构造方式，来保证ID系统的正确运行
    ID(id_type, id);
    ID operator+(id);

public:
    //==操作符重载：判断另一个ID的ID类型与该ID的类型是否相同
    bool operator==(id);
    //<操作符重载：用于数据结构map使用
    bool operator<(id);
    //get_chip_array_id：获取该ID所属的chip_array的ID
    ID get_xxx_id();    
    //is_chip_array：判断本ID是否是chip_array的ID
    bool is_xxx();
    //get_module_id_str：获取本ID所属类型所对应部分的id字符串
    get_module_id_str();
    get_id_str();
    bool valid();

    //get_core_xy：通过ID获取core的坐标，仅当ID类型为CORE或RESOURCE时可以调用，其他类型调用会assert报错
    get_core_xy() const;   
    get_chip_xy() const;

private:
    //_type：id类型
    ID_TYPE _type;
    //_id：id字符串
    string _id;
    friend IDUtil;
};

//重载打印id函数
inline ostream &operator<<(ostream &os, const ID &id)
{
    os << id.get_id_str();
    return os;
}

//获取ID的哈希值，用于数据结构unordered_map等
struct Hash_ID
{
    std::size_t operator()(const ID &id) const
    {
        return hash<string>()(id.get_id_str());
    }
};

//ID类的辅助类
class IDUtil
{
public:
    //
    static ID find_offset_core_id(ID core_id, int32_t dx, int32_t dy);
};

//find_offset_core_id：通过ID，以及给定的偏移量，获取到指定core坐标的ID，仅当core_id的类型等于CORE时有效，其余会assert报错TY_H

```