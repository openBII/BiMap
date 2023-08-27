// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef SPACE_H
#define SPACE_H

#include "src/simulator/behavior_simulator/context.h"
#include "src/simulator/behavior_simulator/identity.h"
#include <memory>

class Context;
class Space
{
 public:
    Space(const ID &id, shared_ptr<Context> ctx) : _id(id), context(ctx) {
        context->addIdentity(id, (Space *)this);
    }

    const ID get_id() const { return _id; }
    virtual ~Space();
    virtual void execute() = 0;
    shared_ptr<Context> get_context() const { return context; }

 protected:
    shared_ptr<Context> context;

 private:
    ID _id;
};

inline Space::~Space() {}

#endif  // SPACE_H
