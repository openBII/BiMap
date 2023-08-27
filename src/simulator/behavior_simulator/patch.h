// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PATCH_H
#define PATCH_H

#include <algorithm>
#include <memory>

// template< class T, class U >
// std::shared_ptr<T> reinterpret_pointer_cast( const std::shared_ptr<U>& r )
// noexcept
// {
// 	    auto p = reinterpret_cast<typename
// std::shared_ptr<T>::element_type*>(r.get()); 		return std::shared_ptr<T>(r, p);
// }

enum device_t { CPU, GPU };

extern device_t DEVICE;
int64_t sign_cast_64_32(int64_t sum);
int32_t int2_t(int32_t x);
int32_t transType(int32_t &data, const int32_t &type, const int32_t &j);
#endif
