#ifndef __LOCKFREE_QUEUE_HPP__
#define __LOCKFREE_QUEUE_HPP__

#include <atomic>  
#include <memory>  
#include "compair.hpp"

template <typename T>  
class LockFreeLinkedListQueue {  
private:  
    struct Node {  
        std::shared_ptr<T> data;  
        std::atomic<Node*> next;  
        Node(T new_data) : data(std::make_shared<T>(new_data)), next(nullptr) {}  
    };  
  
    std::atomic<Node*> head;  
    std::atomic<Node*> tail;  
  
public:  
    LockFreeLinkedListQueue() : head(new Node(T())), tail(head.load()) {}  
  
    ~LockFreeLinkedListQueue() {  
        Node* curr = head.load();  
        while (curr) {  
            Node* toDelete = curr;  
            curr = curr->next.load();  
            delete toDelete;  
        }  
    }  
  
    bool enqueue(T new_value) {  
        Node* new_node = new Node(new_value);  
        while (true) {  
            Node* old_tail = tail.load();  
            Node* next = old_tail->next.load();  
            if (old_tail == tail.load()) {  
                if (next == nullptr) {  
                    if (old_tail->next.compare_exchange_strong(next, new_node)) {  
                        tail.compare_exchange_strong(old_tail, new_node);  
                        return true;  
                    }  
                } else {  
                    tail.compare_exchange_strong(old_tail, next);  
                }  
            }  
        }  
        return false;  
    }  
  
    bool dequeue(T& value) {  
        while (true) {  
            Node* old_head = head.load();  
            Node* next = old_head->next.load();  
            if (old_head == head.load()) {  
                if (next == nullptr) {  
                    return false; // Queue is empty  
                }  
                if (head.compare_exchange_strong(old_head, next)) {  
                    value = *next->data;  
                    delete old_head;  
                    return true;  
                }  
            }  
        }  
    }  
};

struct pkt
{
    int addr_src;   // Address of src node
    int addr_dst;   // Address of dst node
    int t_inject;   // Timestamp of injecting
    int t_eject;    // Timestamp of ejecting
    int pkg_size;   // Size of packet
    struct comp_air_info ca_info;

    pkt(){
        this->addr_src = -1;
        this->addr_dst = -1;
        this->t_inject = -1;
        this->t_eject  = -1;
        this->pkg_size = 1;
    }
    pkt(int src, int dst, int t_i, int t_e){
        this->addr_src = src;
        this->addr_dst = dst;
        this->t_inject = t_i;
        this->t_eject  = t_e;
        this->pkg_size = 1;
    }
    pkt(int src, int dst, int t_i, int t_e, int pkg_size){
        this->addr_src = src;
        this->addr_dst = dst;
        this->t_inject = t_i;
        this->t_eject  = t_e;
        this->pkg_size = pkg_size;
    }
    pkt(int t_i,    int t_e, int type,
        float data, int src, int dst,
        int pkg_size, int iter_tag,
        int x_0,    int y_0, int op_0,
        int x_1,    int y_1, int op_1,
        int x_2,    int y_2, int op_2,
        int x_3,    int y_3, int op_3) {
        this->addr_src = src;
        this->addr_dst = dst;
        this->t_inject = t_i;
        this->t_eject  = t_e;
        this->pkg_size = pkg_size;
        this->ca_info.set(type, data, iter_tag, x_0, y_0, op_0, 
                            x_1, y_1, op_1, x_2, y_2, op_2, x_3, y_3, op_3);
    }
};

class LockFreePktQueue {  
private:  
    struct Node {  
        std::shared_ptr<pkt> data;  
        std::atomic<Node*> next;  
        Node(pkt new_data) : data(std::make_shared<pkt>(new_data)), next(nullptr) {}  
    };  
  
    std::atomic<Node*> head;  
    std::atomic<Node*> tail;  
  
public:  
    LockFreePktQueue() : head(new Node(pkt())), tail(head.load()) {}  
  
    ~LockFreePktQueue() {  
        Node* curr = head.load();  
        while (curr) {  
            Node* toDelete = curr;  
            curr = curr->next.load();  
            delete toDelete;  
        }  
    }  

    bool is_empty() {
        Node* old_head = head.load();
        if (old_head->next.load() == nullptr) {
            return true;
        } else {
            return false;
        }
    }
  
    bool enqueue(int src, int dst, int t_inject, int t_eject, int pkg_size) {
        pkt new_value(src, dst, t_inject, t_eject, pkg_size);
        Node* new_node = new Node(new_value);  
        while (true) {  
            Node* old_tail = tail.load();  
            Node* next = old_tail->next.load();  
            if (old_tail == tail.load()) {  
                if (next == nullptr) {  
                    if (old_tail->next.compare_exchange_strong(next, new_node)) {  
                        tail.compare_exchange_strong(old_tail, new_node);  
                        return true;  
                    }  
                } else {  
                    tail.compare_exchange_strong(old_tail, next);  
                }  
            }  
        }  
        return false;  
    }

    bool enqueue_pkt(pkt new_value) {
        Node* new_node = new Node(new_value);  
        while (true) {  
            Node* old_tail = tail.load();  
            Node* next = old_tail->next.load();  
            if (old_tail == tail.load()) {  
                if (next == nullptr) {  
                    if (old_tail->next.compare_exchange_strong(next, new_node)) {  
                        tail.compare_exchange_strong(old_tail, new_node);  
                        return true;  
                    }  
                } else {  
                    tail.compare_exchange_strong(old_tail, next);  
                }  
            }  
        }  
        return false;  
    }
  
    int dequeue() {  
        while (true) {  
            Node* old_head = head.load();  
            Node* next = old_head->next.load();  
            if (old_head == head.load()) {  
                if (next == nullptr) {  
                    return -1;
                }  
                if (head.compare_exchange_strong(old_head, next)) {  
                    pkt value = *next->data;  
                    delete old_head;  
                    return value.t_eject;  
                }  
            }  
        }  
    } 

    pkt dequeue_pkt() {  
        while (true) {  
            Node* old_head = head.load();  
            Node* next = old_head->next.load();  
            if (old_head == head.load()) {  
                if (next == nullptr) {
                    pkt null_pkt;
                    return null_pkt;
                }  
                if (head.compare_exchange_strong(old_head, next)) {  
                    pkt value = *next->data;  
                    return value;  
                }  
            }  
        }  
    }  
};

#endif