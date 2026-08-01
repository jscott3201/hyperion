//! B6: the `/control` ops surface (06 §Surfaces). Health, stats, reload (409
//! conflict during a generation, 501 idle — honest stub), shutdown.
//!
//! The stats counters are `Arc<AtomicU64>` so the request handlers increment
//! them lock-free. `reload` is 409 if a generation is in flight (the
//! single-flight permit is held); 501 Not Implemented when idle — PR B ships
//! the 409 guard, the actual hot-swap of a loaded model is a later slice (one
//! model at startup is the PR B reality, so there's nothing to swap to).
//! `shutdown` sets a flag + initiates graceful drain.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde::Serialize;

use crate::tool_call::ToolCallStats;

/// The shared ops state: readiness + counters + shutdown flag. Cloned cheaply
/// (all `Arc`) into each handler. The request handlers bump the counters; the
/// `/control/*` handlers read them.
#[derive(Clone)]
pub struct ControlState {
    inner: Arc<Inner>,
}

struct Inner {
    ready: AtomicBool,
    model_id: String,
    /// True while a generation holds the single-flight permit (the 409 guard).
    in_flight: AtomicBool,
    requests_total: AtomicU64,
    tokens_total: AtomicU64,
    governor_rejections: AtomicU64,
    near_tie_events: AtomicU64,
    single_flight_rejections: AtomicU64,
    tool_calls_parsed: AtomicU64,
    tool_calls_wellformed: AtomicU64,
    tool_calls_repaired: AtomicU64,
    tool_calls_deduped: AtomicU64,
    tool_call_candidate_overflows: AtomicU64,
    tool_call_limit_exceeded: AtomicU64,
    shutdown: AtomicBool,
}

impl ControlState {
    /// Create the ops state. `model_id` is echoed by `/control/health` +
    /// `/control/stats`. Starts `ready=false`; the server flips it true after
    /// the model loads.
    #[must_use]
    pub fn new(model_id: &str) -> Self {
        Self {
            inner: Arc::new(Inner {
                ready: AtomicBool::new(false),
                model_id: model_id.to_string(),
                in_flight: AtomicBool::new(false),
                requests_total: AtomicU64::new(0),
                tokens_total: AtomicU64::new(0),
                governor_rejections: AtomicU64::new(0),
                near_tie_events: AtomicU64::new(0),
                single_flight_rejections: AtomicU64::new(0),
                tool_calls_parsed: AtomicU64::new(0),
                tool_calls_wellformed: AtomicU64::new(0),
                tool_calls_repaired: AtomicU64::new(0),
                tool_calls_deduped: AtomicU64::new(0),
                tool_call_candidate_overflows: AtomicU64::new(0),
                tool_call_limit_exceeded: AtomicU64::new(0),
                shutdown: AtomicBool::new(false),
            }),
        }
    }

    /// Mark the model loaded (ready to serve).
    pub fn set_ready(&self, ready: bool) {
        self.inner.ready.store(ready, Ordering::SeqCst);
    }

    /// Whether the model is loaded + ready.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.ready.load(Ordering::SeqCst)
    }

    /// Mark a generation in flight (the single-flight permit acquired). The
    /// 409 reload guard checks this.
    pub fn set_in_flight(&self, in_flight: bool) {
        self.inner.in_flight.store(in_flight, Ordering::SeqCst);
    }

    /// Whether a generation is in flight.
    #[must_use]
    pub fn is_in_flight(&self) -> bool {
        self.inner.in_flight.load(Ordering::SeqCst)
    }

    /// Bump the request counter (one per accepted request).
    pub fn inc_requests(&self) {
        self.inner.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Add `n` to the token counter (prompt + completion tokens).
    pub fn add_tokens(&self, n: u64) {
        self.inner.tokens_total.fetch_add(n, Ordering::Relaxed);
    }

    /// Bump the governor-rejection counter (a 529, pre- or mid-stream).
    pub fn inc_governor_rejections(&self) {
        self.inner
            .governor_rejections
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the near-tie-event counter (sampler telemetry).
    pub fn inc_near_tie(&self) {
        self.inner.near_tie_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the single-flight-rejection counter (a 429).
    pub fn inc_single_flight_rejections(&self) {
        self.inner
            .single_flight_rejections
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Aggregate one completed request parser snapshot exactly once. Every
    /// counter saturates independently so lifetime telemetry cannot wrap.
    pub fn add_tool_call_stats(&self, stats: ToolCallStats) {
        saturating_atomic_add(&self.inner.tool_calls_parsed, stats.parsed);
        saturating_atomic_add(&self.inner.tool_calls_wellformed, stats.wellformed);
        saturating_atomic_add(&self.inner.tool_calls_repaired, stats.repaired);
        saturating_atomic_add(&self.inner.tool_calls_deduped, stats.deduped);
        saturating_atomic_add(
            &self.inner.tool_call_candidate_overflows,
            stats.candidate_overflows,
        );
        saturating_atomic_add(
            &self.inner.tool_call_limit_exceeded,
            stats.call_limit_exceeded,
        );
    }

    /// Request a graceful shutdown. Returns true if this call set the flag
    /// (the first shutdown request); false if one was already in flight.
    pub fn request_shutdown(&self) -> bool {
        self.inner
            .shutdown
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Whether shutdown has been requested.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.inner.shutdown.load(Ordering::SeqCst)
    }
}

/// The `/control/health` body.
#[derive(Serialize)]
pub struct Health {
    pub ready: bool,
    pub model: String,
    pub governor: String,
}

/// The `/control/stats` body.
#[derive(Serialize)]
pub struct Stats {
    pub requests_total: u64,
    pub requests_in_flight: bool,
    pub tokens_total: u64,
    pub governor_rejections: u64,
    pub near_tie_events: u64,
    pub single_flight_rejections: u64,
    pub tool_calls_parsed: u64,
    pub tool_calls_wellformed: u64,
    pub tool_calls_repaired: u64,
    pub tool_calls_deduped: u64,
    pub tool_call_candidate_overflows: u64,
    pub tool_call_limit_exceeded: u64,
}

impl ControlState {
    /// Build the `/control/health` body.
    #[must_use]
    pub fn health(&self) -> Health {
        Health {
            ready: self.is_ready(),
            model: self.inner.model_id.clone(),
            governor: if self.is_in_flight() {
                "busy".to_string()
            } else {
                "idle".to_string()
            },
        }
    }

    /// Build the `/control/stats` body.
    #[must_use]
    pub fn stats(&self) -> Stats {
        Stats {
            requests_total: self.inner.requests_total.load(Ordering::Relaxed),
            requests_in_flight: self.is_in_flight(),
            tokens_total: self.inner.tokens_total.load(Ordering::Relaxed),
            governor_rejections: self.inner.governor_rejections.load(Ordering::Relaxed),
            near_tie_events: self.inner.near_tie_events.load(Ordering::Relaxed),
            single_flight_rejections: self.inner.single_flight_rejections.load(Ordering::Relaxed),
            tool_calls_parsed: self.inner.tool_calls_parsed.load(Ordering::Relaxed),
            tool_calls_wellformed: self.inner.tool_calls_wellformed.load(Ordering::Relaxed),
            tool_calls_repaired: self.inner.tool_calls_repaired.load(Ordering::Relaxed),
            tool_calls_deduped: self.inner.tool_calls_deduped.load(Ordering::Relaxed),
            tool_call_candidate_overflows: self
                .inner
                .tool_call_candidate_overflows
                .load(Ordering::Relaxed),
            tool_call_limit_exceeded: self.inner.tool_call_limit_exceeded.load(Ordering::Relaxed),
        }
    }

    /// The reload verdict: 409 if a generation is in flight, 501 (Not
    /// Implemented) if idle. PR B ships the 409 guard; the actual hot-swap is
    /// a later slice (one model at startup → nothing to swap to).
    #[must_use]
    pub fn reload_verdict(&self) -> ReloadVerdict {
        if self.is_in_flight() {
            ReloadVerdict::Conflict
        } else {
            ReloadVerdict::NotImplemented
        }
    }
}

fn saturating_atomic_add(counter: &AtomicU64, amount: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(amount))
    });
}

/// The `/control/reload` result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReloadVerdict {
    /// 409 — a generation is in flight; reload would lose it.
    Conflict,
    /// 501 — idle, but hot-swap isn't implemented in PR B.
    NotImplemented,
}

impl ReloadVerdict {
    /// The HTTP status code.
    #[must_use]
    pub const fn status(self) -> u16 {
        match self {
            Self::Conflict => 409,
            Self::NotImplemented => 501,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_reflects_readiness() {
        let s = ControlState::new("gemma4-12b");
        assert!(!s.health().ready);
        s.set_ready(true);
        assert!(s.health().ready);
        assert_eq!(s.health().model, "gemma4-12b");
    }

    #[test]
    fn stats_counters_increment() {
        let s = ControlState::new("m");
        s.inc_requests();
        s.inc_requests();
        s.add_tokens(42);
        s.inc_governor_rejections();
        s.inc_near_tie();
        s.inc_single_flight_rejections();
        s.add_tool_call_stats(ToolCallStats {
            parsed: 3,
            wellformed: 2,
            repaired: 1,
            deduped: 1,
            candidate_overflows: 4,
            call_limit_exceeded: 5,
        });
        let stats = s.stats();
        assert_eq!(stats.requests_total, 2);
        assert_eq!(stats.tokens_total, 42);
        assert_eq!(stats.governor_rejections, 1);
        assert_eq!(stats.near_tie_events, 1);
        assert_eq!(stats.single_flight_rejections, 1);
        assert_eq!(stats.tool_calls_parsed, 3);
        assert_eq!(stats.tool_calls_wellformed, 2);
        assert_eq!(stats.tool_calls_repaired, 1);
        assert_eq!(stats.tool_calls_deduped, 1);
        assert_eq!(stats.tool_call_candidate_overflows, 4);
        assert_eq!(stats.tool_call_limit_exceeded, 5);
        assert!(!stats.requests_in_flight);
    }

    #[test]
    fn reload_returns_409_during_generation() {
        let s = ControlState::new("m");
        s.set_in_flight(true);
        assert_eq!(s.reload_verdict(), ReloadVerdict::Conflict);
        assert_eq!(s.reload_verdict().status(), 409);
    }

    #[test]
    fn parser_stats_saturate_without_wrapping() {
        let s = ControlState::new("m");
        s.inner.tool_calls_parsed.store(u64::MAX, Ordering::Relaxed);
        s.add_tool_call_stats(ToolCallStats {
            parsed: 1,
            ..ToolCallStats::default()
        });
        assert_eq!(s.stats().tool_calls_parsed, u64::MAX);
    }

    #[test]
    fn reload_returns_501_when_idle() {
        let s = ControlState::new("m");
        assert_eq!(s.reload_verdict(), ReloadVerdict::NotImplemented);
        assert_eq!(s.reload_verdict().status(), 501);
    }

    #[test]
    fn shutdown_flag_is_one_shot() {
        let s = ControlState::new("m");
        assert!(s.request_shutdown(), "first shutdown sets the flag");
        assert!(!s.request_shutdown(), "second shutdown is a no-op");
        assert!(s.is_shutdown());
    }

    #[test]
    fn in_flight_toggles() {
        let s = ControlState::new("m");
        assert!(!s.is_in_flight());
        s.set_in_flight(true);
        assert!(s.is_in_flight());
        assert_eq!(s.health().governor, "busy");
        s.set_in_flight(false);
        assert_eq!(s.health().governor, "idle");
    }
}
