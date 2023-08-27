// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef UTIL_H
#define UTIL_H

#include <ctime>

enum P2mode { AVGPOOL, VADD };
inline long long getTimeNs() {
    struct timespec ts;
    //    clock_gettime(CLOCK_REALTIME,&ts);
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

template <typename... args>
inline void _deleter(void *p, uint32_t n, args... arg) {
    for (uint32_t i = 0; i < n; i++) {
        _deleter(reinterpret_cast<void **>(p)[i], arg...);
    }
    delete[] p;
}

template <> inline void _deleter(void *p, uint32_t n) { delete[] p; }

template <typename... args> inline void deleter(void *p, args... arg) {
    _deleter(p, arg...);
}

template <typename data> inline data *_new_array(uint32_t cur) {
    return new data[cur]();
}

template <typename data, typename... args>
inline data *_new_array(uint32_t cur, args... _args) {
    data **p = new data *[cur];
    for (uint32_t i = 0; i < cur; i++) {
        p[i] = _new_array<data>(_args...);
    }
    return (data *)p;
}

template <typename data, typename... args>
inline shared_ptr<data> new_array(args... _args) {
    data *p = _new_array<data>(_args...);

    return shared_ptr<data>(p,
                            bind(deleter<args...>, placeholders::_1, _args...));
}

/*
** example:
** shared_ptr<uint32_t> p = new_array<uint32_t>(2, 2, 100, 100, 100);
** create 5 dimensions matrix:2*2*100*100*100 ,elements type:uint32_t
*/

template <class T> class ArrayBase
{
 public:
    ArrayBase() : m_array(nullptr), m_strides(nullptr) {}
    ArrayBase(T *array, int *strides) : m_array(array), m_strides(strides) {}
    ArrayBase(initializer_list<int> dims) {
        int nDim = dims.size();
        m_strides = new int[nDim];
        int i = 0;
        m_size = 1;
        for (auto ri = dims.end() - 1; ri != dims.begin() - 1; --ri) {
            m_size *= *ri;
            m_strides[i++] = m_size;
        }
        m_array = new T[m_size];
        memset(m_array, 0, m_size * sizeof(int32_t));
    }
    ArrayBase(initializer_list<int> dims, initializer_list<T> data)
        : ArrayBase<T>(dims) {
        m_array = new T[m_size];
        int i = 0;
        for (auto &x : data) {
            m_array[i++] = x;
        }
    }
    T *raw() { return m_array; }
    int *m_strides;
    int m_size;
    T *m_array;
};

template <class T, int N> class SubArray : public ArrayBase<T>
{
 public:
    SubArray(T *array, int *strides)
        : ArrayBase<T>(array, strides),
          subArray(new SubArray<T, N - 1>(this->m_array, this->m_strides)) {}
    SubArray<T, N - 1> *subArray;
    SubArray<T, N - 1> &operator[](int i) {
        subArray->m_array = this->m_array + i * this->m_strides[N - 2];
        return *subArray;
    }
    ~SubArray() {
        delete subArray;
        subArray = nullptr;
    }
};
template <class T> class SubArray<T, 1> : public ArrayBase<T>
{
 public:
    SubArray(T *array, int *strides) : ArrayBase<T>(array, strides) {}
    T &operator[](int i) { return *(this->m_array + i); }
};
template <class T, int N> class Array : public ArrayBase<T>
{
 public:
    SubArray<T, N> *subArray;
    Array(initializer_list<int> dims)
        : ArrayBase<T>(dims),
          subArray(new SubArray<T, N>(this->m_array, this->m_strides)) {}
    Array(initializer_list<int> dims, initializer_list<int> array)
        : ArrayBase<T>(dims, array),
          subArray(new SubArray<T, N>(this->m_array, this->m_strides)) {}
    Array() {}
    Array(Array<T, N> &&other) {
        this->subArray = other.subArray;
        this->m_array = other.m_array;
        this->m_strides = other.m_strides;
        this->m_size = other.m_size;
        other.m_strides = nullptr;
        other.subArray = nullptr;
        other.m_array = nullptr;
    }
    Array<T, N> &operator=(Array<T, N> &&other) {
        if (this != &other) {
            delete this->subArray;
            delete this->m_array;
            delete this->m_strides;
            this->subArray = other.subArray;
            this->m_array = other.m_array;

            this->m_strides = other.m_strides;
            this->m_size = other.m_size;
            other.m_strides = nullptr;
            other.m_array = nullptr;
            other.subArray = nullptr;
        }
        return *this;
    }
    SubArray<T, N - 1> &operator[](int i) { return (*subArray)[i]; }
    ~Array() {
        delete[] this->m_array;
        delete[] this->m_strides;
        delete this->subArray;
    }
};
template <class T> class Array<T, 1> : public ArrayBase<T>
{
 public:
    Array(initializer_list<int> dims) : ArrayBase<T>(dims) {}
    Array(initializer_list<int> dims, initializer_list<int> array)
        : ArrayBase<T>(dims, array) {}
    Array() {}
    Array(Array<T, 1> &&other) {
        this->m_array = other.m_array;
        this->m_strides = other.m_strides;
        this->m_size = other.m_size;
        other.m_strides = nullptr;
        other.m_array = nullptr;
    }
    Array<T, 1> &operator=(Array<T, 1> &&other) {
        if (this != &other) {
            delete this->m_array;
            delete this->m_strides;
            this->m_array = other.m_array;
            this->m_strides = other.m_strides;
            this->m_size = other.m_size;
            other.m_strides = nullptr;
            other.m_array = nullptr;
        }
        return *this;
    }
    T &operator[](int i) { return *(this->m_array + i); }
    ~Array() {
        delete[] this->m_array;
        delete[] this->m_strides;
    }
};
#endif  // UTIL_H
