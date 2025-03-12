#ifndef _COMP_AIR_HPP
#define _COMP_AIR_HPP

#define EDGE_LEN 8

struct comp_air_info 
{
public:
    int type;       // 0: scalar, 1: reduce, 2: p2p, 3: broadcast, 4: Cover, -1: Normal
    float data;     // data (BF16 in reality)
    int iter_tag;   // 0 ~ 15
    int edge_len;
    int x_0; int y_0; int op_0; // x_offset, y_offset, op (0: data, 1: iter || 0: +=, 1: -=, 2: *=, 3: /=)
    int x_1; int y_1; int op_1; // x_offset, y_offset, op (0: data, 1: iter || 0: +=, 1: -=, 2: *=, 3: /=)
    int x_2; int y_2; int op_2; // x_offset, y_offset, op (0: data, 1: iter || 0: +=, 1: -=, 2: *=, 3: /=)
    int x_3; int y_3; int op_3; // x_offset, y_offset, op (0: data, 1: iter || 0: +=, 1: -=, 2: *=, 3: /=)

    comp_air_info() 
    {
        this->type = -1;
        this->data = -1;
        this->iter_tag = -1;
        this->x_0 = -1; this->y_0 = -1; this->op_0 = -1;
        this->x_1 = -1; this->y_1 = -1; this->op_1 = -1;
        this->x_2 = -1; this->y_2 = -1; this->op_2 = -1;
        this->x_3 = -1; this->y_3 = -1; this->op_3 = -1;
        this->edge_len = EDGE_LEN;
    }

    void set(
        int type, float data, int iter_tag,
        int x_0, int y_0, int op_0,
        int x_1, int y_1, int op_1,
        int x_2, int y_2, int op_2,
        int x_3, int y_3, int op_3)
    {
        this->type = type;
        this->data = data;
        this->iter_tag = iter_tag;
        this->x_0 = x_0; this->y_0 = y_0; this->op_0 = op_0;
        this->x_1 = x_1; this->y_1 = y_1; this->op_1 = op_1;
        this->x_2 = x_2; this->y_2 = y_2; this->op_2 = op_2;
        this->x_3 = x_3; this->y_3 = y_3; this->op_3 = op_3;
    }
};


#endif