// include/aim/interfaces/safety_supervisor.hpp
#pragma once

#include "aim/core/actuator.hpp"
#include "aim/core/safety.hpp"
#include "aim/core/time.hpp"

namespace aim {

/// @brief Abstraction for runtime safety monitoring, command validation, and fail-closed e-stop.
class ISafetySupervisor {
public:
    virtual ~ISafetySupervisor() = default;
    virtual bool check_actuation_safety(const ActuationCommand& command, MonotonicNs now_ns) noexcept = 0;
    virtual void trigger_emergency_stop(SafetyReason reason = SafetyReason::emergency_stop_triggered) noexcept = 0;
    [[nodiscard]] virtual bool is_latched() const noexcept = 0;
    virtual bool try_reset(const ResetToken& token) noexcept = 0;
};

} // namespace aim
