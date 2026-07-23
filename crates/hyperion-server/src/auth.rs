//! B3: the bind-gate + constant-time bearer auth (06 §Surfaces, the bonsai
//! "localhost-first, fail-closed" rule).
//!
//! **Bind-gate**: a non-loopback bind address REQUIRES a bearer token or the
//! server refuses to start (exit 78, config error). Loopback binds with no
//! token run unauthenticated — localhost-first.
//!
//! **Bearer middleware**: when a token is configured, every `/v1/*` and
//! `/control/*` request must carry `Authorization: Bearer <token>`, compared
//! in **constant time** via `subtle::ConstantTimeEq` (never `==` on a secret —
//! the bonsai 401 rule). A missing/wrong token → 401 with the dialect-correct
//! error envelope ([`crate::dialect::ErrorEnvelope`]).
//!
//! The constant-time compare is on the token bytes; a wrong-length token
//! short-circuits to 401 (the length is not a secret — the token value is).

use std::net::IpAddr;
use std::str::FromStr;

use subtle::ConstantTimeEq;

/// The bind-gate verdict: may the server start with this `(addr, token)` pair?
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindGate {
    /// Loopback, no token — OK (localhost-first).
    Loopback,
    /// Non-loopback with a token — OK (auth enforced).
    Authenticated,
}

/// Parse a bind decision. Returns `Ok(BindGate)` if the `(addr, token)` pair is
/// allowed to start; `Err(reason)` if a non-loopback bind has no token
/// (fail-closed). `addr` is the resolved IP (not the host string).
///
/// # Errors
/// Returns a human-facing reason string when a non-loopback bind is missing a
/// token (the caller exits with 78 + this reason).
pub fn bind_gate(addr: IpAddr, token: Option<&str>) -> Result<BindGate, String> {
    let is_loopback = addr.is_loopback();
    match (is_loopback, token) {
        (true, _) => Ok(BindGate::Loopback),
        (false, Some(_)) => Ok(BindGate::Authenticated),
        (false, None) => Err(format!(
            "non-loopback bind {addr} requires --token (bearer auth); refusing to start (fail-closed)"
        )),
    }
}

/// Parse a `host:port` string into the resolved `IpAddr` for the bind-gate.
/// Returns `Err` if the host doesn't resolve to an IP. The caller passes the
/// already-resolved `IpAddr` to [`bind_gate`].
///
/// # Errors
/// Returns a message if the string isn't `host:port` with a parseable IP.
pub fn parse_bind_addr(s: &str) -> Result<(IpAddr, u16), String> {
    let (host, port) = s
        .rsplit_once(':')
        .ok_or_else(|| format!("--addr '{s}' is not host:port"))?;
    let port: u16 = port
        .parse()
        .map_err(|_| format!("--addr '{s}' has a non-numeric port"))?;
    // Accept a bare IP for the host; a hostname would need resolution (PR B is
    // localhost-first, so an IP is the expected form).
    let addr = IpAddr::from_str(host).map_err(|e| format!("--addr host '{host}': {e}"))?;
    Ok((addr, port))
}

/// A bearer-auth failure (missing or mismatched token). Carries a stable
/// reason so the 401 envelope can render a message without leaking whether
/// the header was missing vs wrong (both surface the same opaque reason).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthError;

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bearer token missing or invalid")
    }
}

impl std::error::Error for AuthError {}

/// Verify a request's bearer token against the configured one in constant
/// time. Returns `Ok(())` on a match, `Err(AuthError)` on a mismatch/missing
/// header. `None`-configured token means auth is off (loopback) — always `Ok`.
///
/// The compare is constant-time on the token bytes via `subtle::ConstantTimeEq`.
/// A wrong-length token short-circuits to `Err` (length isn't a secret); the
/// byte-level compare is CT so a partial-match attacker learns nothing about
/// the configured token's value.
pub fn verify_bearer(configured: Option<&str>, header: Option<&str>) -> Result<(), AuthError> {
    let Some(configured) = configured else {
        return Ok(()); // auth off (loopback)
    };
    let Some(provided) = header.and_then(|h| h.strip_prefix("Bearer ")) else {
        return Err(AuthError);
    };
    let a = configured.as_bytes();
    let b = provided.as_bytes();
    if a.len() != b.len() {
        return Err(AuthError); // length mismatch — not a secret
    }
    // ct_eq returns 0 on equal; subtle masks to a 0/1 Choice.
    if a.ct_eq(b).into() {
        Ok(())
    } else {
        Err(AuthError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn loopback_allows_no_token() {
        let addr = IpAddr::V4(Ipv4Addr::LOCALHOST);
        assert_eq!(bind_gate(addr, None).unwrap(), BindGate::Loopback);
        // a token on loopback is allowed too (just unused)
        assert_eq!(bind_gate(addr, Some("secret")).unwrap(), BindGate::Loopback);
    }

    #[test]
    fn non_loopback_requires_token() {
        let addr = IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0));
        assert!(
            bind_gate(addr, None).is_err(),
            "0.0.0.0 without a token must refuse"
        );
        assert_eq!(
            bind_gate(addr, Some("secret")).unwrap(),
            BindGate::Authenticated
        );
    }

    #[test]
    fn ipv6_loopback_allows_no_token() {
        let addr = IpAddr::V6(Ipv6Addr::LOCALHOST);
        assert_eq!(bind_gate(addr, None).unwrap(), BindGate::Loopback);
    }

    #[test]
    fn parse_bind_addr_accepts_ipv4_loopback() {
        let (addr, port) = parse_bind_addr("127.0.0.1:8080").unwrap();
        assert_eq!(addr, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(port, 8080);
    }

    #[test]
    fn parse_bind_addr_rejects_garbage() {
        assert!(parse_bind_addr("not-an-addr").is_err());
        assert!(parse_bind_addr("127.0.0.1:notaport").is_err());
        assert!(
            parse_bind_addr("localhost:8080").is_err(),
            "hostnames not resolved in PR B"
        );
    }

    #[test]
    fn verify_bearer_accepts_correct_token() {
        assert!(verify_bearer(Some("s3cret"), Some("Bearer s3cret")).is_ok());
    }

    #[test]
    fn verify_bearer_rejects_wrong_token() {
        assert!(verify_bearer(Some("s3cret"), Some("Bearer wrong")).is_err());
    }

    #[test]
    fn verify_bearer_rejects_missing_header() {
        assert!(verify_bearer(Some("s3cret"), None).is_err());
        assert!(verify_bearer(Some("s3cret"), Some("Bearerless")).is_err());
    }

    #[test]
    fn verify_bearer_rejects_wrong_length() {
        // A wrong-length token short-circuits (length isn't a secret).
        assert!(verify_bearer(Some("s3cret"), Some("Bearer s3")).is_err());
        assert!(verify_bearer(Some("s3cret"), Some("Bearer s3cret-extra")).is_err());
    }

    #[test]
    fn verify_bearer_no_configured_token_is_open() {
        // Loopback: no configured token → auth off → always Ok.
        assert!(verify_bearer(None, None).is_ok());
        assert!(verify_bearer(None, Some("Bearer anything")).is_ok());
    }

    #[test]
    fn verify_bearer_rejects_prefix_only() {
        // "Bearer " with no token after it.
        assert!(verify_bearer(Some("s3cret"), Some("Bearer ")).is_err());
    }
}
