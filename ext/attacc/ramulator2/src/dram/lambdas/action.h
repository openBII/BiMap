#ifndef RAMULATOR_DRAM_LAMBDAS_ACTION_H
#define RAMULATOR_DRAM_LAMBDAS_ACTION_H

#include <spdlog/spdlog.h>

#include "dram/node.h"

namespace Ramulator {
namespace Lambdas {

template<class>
inline constexpr bool false_v = false;

namespace Action {
namespace Bank {
  template <class T>
  void ACT(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    node->m_state = T::m_states["Opened"];
    node->m_row_state[target_id] = T::m_states["Opened"];
  };

  template <class T>
  void PRE(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    node->m_state = T::m_states["Closed"];
    node->m_row_state.clear();
  };

  template <class T>
  void ACTAB(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    // For HBM3
    if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 4) {
      typename T::Node* channel = node->m_parent_node->m_parent_node->m_parent_node->m_parent_node;
      for (auto pc : channel->m_child_nodes) {
        for (auto rank : pc->m_child_nodes) {
          for (auto bg : rank->m_child_nodes) {
            for (auto bank : bg->m_child_nodes) {
              bank->m_state = T::m_states["Opened"];
              bank->m_row_state[target_id] = T::m_states["Opened"];
            }
          }
        }
      }
    } else {
      static_assert(
        false_v<T>,
        "[Action::Bank] Unsupported organization. Please write your own ACTAB function."
      );
    }
  };

  template <class T>
  void ACTSB(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    // For HBM3
    if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 4) {
      typename T::Node* ch = node->m_parent_node->m_parent_node->m_parent_node->m_parent_node;
      for (auto pch : ch->m_child_nodes) {
        for (auto rank : pch->m_child_nodes) {
          for (auto bg : rank->m_child_nodes) {
            for (auto bank : bg->m_child_nodes) {
              if (bank->m_node_id == node->m_node_id) {
                bank->m_state = T::m_states["Opened"];
                bank->m_row_state[target_id] = T::m_states["Opened"];
              }
            }
          }
        }
      }
    } else {
      static_assert(
        false_v<T>,
        "[Action::Bank] Unsupported organization. Please write your own ACTSB function."
      );
    }
  };

  template <class T>
  void ACTPB(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    // For HBM3
    if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 4) {
      typename T::Node* ch = node->m_parent_node->m_parent_node->m_parent_node->m_parent_node;
      for (auto pch : ch->m_child_nodes) {
        for (auto rank : pch->m_child_nodes) {
          if (rank->m_node_id == node->m_parent_node->m_parent_node->m_node_id) {
            for (auto bg : rank->m_child_nodes) {
              if (bg->m_node_id == node->m_parent_node->m_node_id) {
                for (auto bank : bg->m_child_nodes) {
                  if (bank->m_node_id == node->m_node_id) {
                    bank->m_state = T::m_states["Opened"];
                    bank->m_row_state[target_id] = T::m_states["Opened"];
                  }
                }
              }
            }
          }
        }
      }
    } else {
      static_assert(
        false_v<T>,
        "[Action::Bank] Unsupported organization. Please write your own ACTPB function."
      );
    }
  };

  template <class T>
  void PRESB(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    // For HBM3
    if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 4) {
      typename T::Node* ch = node->m_parent_node->m_parent_node->m_parent_node->m_parent_node;
      for (auto pch : ch->m_child_nodes) {
        for (auto rank : pch->m_child_nodes) {
          if (rank->m_node_id == node->m_parent_node->m_parent_node->m_node_id) {
            for (auto bg : rank->m_child_nodes) {
              if (bg->m_node_id == node->m_parent_node->m_node_id) {
                for (auto bank : bg->m_child_nodes) {
                  if (bank->m_node_id == node->m_node_id) {
                    bank->m_state = T::m_states["Closed"];
                    bank->m_row_state.clear();
                  }
                }
              }
            }
          }
        }
      }
    } else {
      static_assert(
        false_v<T>,
        "[Action::Bank] Unsupported organization. Please write your own PREPB function."
      );
    }
  };

  template <class T>
  void PREsb(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    typename T::Node* rank = node->m_parent_node->m_parent_node;
    for (auto bg : rank->m_child_nodes) {
      for (auto bank : bg->m_child_nodes) {
        if (bank->m_node_id == node->m_node_id) {
          bank->m_state = T::m_states["Closed"];
          bank->m_row_state.clear();
        }
      }
    }
  };  

  template <class T>
  void PREPB(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    // For HBM3
    if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 4) {
      typename T::Node* ch = node->m_parent_node->m_parent_node->m_parent_node->m_parent_node;
      for (auto pch : ch->m_child_nodes) {
        for (auto rank : pch->m_child_nodes) {
          if (rank->m_node_id == node->m_parent_node->m_parent_node->m_node_id) {
            for (auto bg : rank->m_child_nodes) {
              if (bg->m_node_id == node->m_parent_node->m_node_id) {
                for (auto bank : bg->m_child_nodes) {
                  if (bank->m_node_id == node->m_node_id) {
                    bank->m_state = T::m_states["Closed"];
                    bank->m_row_state.clear();
                  }
                }
              }
            }
          }
        }
      }
    } else {
      static_assert(
        false_v<T>,
        "[Action::Bank] Unsupported organization. Please write your own ACTPB function."
      );
    }
  };

}       // namespace Bank

namespace BankGroup {
  template <class T>
  void PREsb(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    typename T::Node* rank = node->m_parent_node;
    for (auto bg : rank->m_child_nodes) {
      for (auto bank : bg->m_child_nodes) {
        if (bank->m_node_id == target_id) {
          bank->m_state = T::m_states["Closed"];
          bank->m_row_state.clear();
        }
      }
    }
  };  

  template <class T>
  void SameBankActions(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    AddrVec_t same_bank_addr(T::m_levels.size(), -1);
    same_bank_addr[T::m_levels["bank"]] = target_id;

    typename T::Node* rank = node->m_parent_node;
    for (auto bg : rank->m_child_nodes) {
      for (auto bank : bg->m_child_nodes) {
        if (bank->m_node_id == target_id) {
          bank->update_timing(cmd, same_bank_addr, clk);
        }
      }
    }
  }

  template <class T>
  void PIMSameBankActions(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    AddrVec_t same_bank_addr(T::m_levels.size(), -1);
    same_bank_addr[T::m_levels["bank"]] = target_id;

    typename T::Node* ch = node->m_parent_node->m_parent_node->m_parent_node;
    for (auto pch : ch->m_child_nodes) {
      for (auto rank : pch->m_child_nodes) {
        for (auto bg : rank->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            if (bank->m_node_id == target_id) {
              bank->update_timing(cmd, same_bank_addr, clk);
            }
          }
        }
      }
    }
  }


  template <class T>
  void PIMPerBankActions(typename T::Node* node, int cmd, int target_id, Clk_t clk) { // pCH Broadcast
    AddrVec_t same_bank_addr(T::m_levels.size(), -1);
    same_bank_addr[T::m_levels["bank"]] = target_id;

    typename T::Node* ch = node->m_parent_node->m_parent_node->m_parent_node;
    for (auto pch : ch->m_child_nodes) {
      for (auto rank : pch->m_child_nodes) {
        if (rank->m_node_id == node->m_parent_node->m_parent_node->m_node_id) {
          for (auto bg : rank->m_child_nodes) {
            if (bg->m_node_id == node->m_parent_node->m_node_id) {
              for (auto bank : bg->m_child_nodes) {
                if (bank->m_node_id == target_id) {
                  bank->update_timing(cmd, same_bank_addr, clk);
                }
              }
            }
          }
        }
      }
    }
  }

}       // namespace BankGroup


namespace Rank {
  template <class T>
  void PREA(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    if constexpr (T::m_levels["bank"] - T::m_levels["rank"] == 1) {
      for (auto bank : node->m_child_nodes) {
        bank->m_state = T::m_states["Closed"];
        bank->m_row_state.clear();
      }
    } else if constexpr (T::m_levels["bank"] - T::m_levels["rank"] == 2) {
      for (auto bg : node->m_child_nodes) {
        for (auto bank : bg->m_child_nodes) {
          bank->m_state = T::m_states["Closed"];
          bank->m_row_state.clear();
        }
      }
    } else {
      static_assert(
        false_v<T>, 
        "[Action::Rank] Unsupported organization. Please write your own PREA function."
      );
    }
  };

  template <class T>
  void PREsb(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    for (auto bg : node->m_child_nodes) {
      for (auto bank : bg->m_child_nodes) {
        if (bank->m_node_id == target_id) {
          bank->m_state = T::m_states["Closed"];
          bank->m_row_state.clear();
        }
      }
    }
  };

}       // namespace Rank

namespace Channel {
  // TODO: Make these nicer...
  template <class T>
  void PREA(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 2) {
      for (auto bg : node->m_child_nodes) {
        for (auto bank : bg->m_child_nodes) {
          bank->m_state = T::m_states["Closed"];
          bank->m_row_state.clear();
        }
      }
    } else if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 3) {
      for (auto pc : node->m_child_nodes) {
        for (auto bg : pc->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            bank->m_state = T::m_states["Closed"];
            bank->m_row_state.clear();
          }
        }
      }
    } else if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 4) {
      for (auto pc : node->m_child_nodes) {
        for (auto rank : pc->m_child_nodes) {
          for (auto bg : rank->m_child_nodes) {
            for (auto bank : bg->m_child_nodes) {
              bank->m_state = T::m_states["Closed"];
              bank->m_row_state.clear();
            }
          }
        }
      }
    } else {
      static_assert(
        false_v<T>, 
        "[Action::Rank] Unsupported organization. Please write your own PREA function."
      );
    }
  };
}      // namespace Channel

}       // namespace Action
}       // namespace Lambdas
};      // namespace Ramulator

#endif  // RAMULATOR_DRAM_LAMBDAS_ACTION_H