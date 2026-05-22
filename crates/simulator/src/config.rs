// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{fmt, ops::Range, time::Duration};

use serde::{Deserialize, Serialize};

use dag::config::ImportExport;
use replica::config::{LoadGeneratorConfig, ReplicaParameters};

/// Either a single simulation or a suite of simulations to run sequentially.
///
/// The untagged representation lets one YAML file be either a mapping (single
/// config, as before) or a top-level sequence of configs (suite).
#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum SimulationMode {
    Suite(Vec<SimulationConfig>),
    Single(SimulationConfig),
}

impl SimulationMode {
    pub fn into_configs(self) -> Vec<SimulationConfig> {
        match self {
            SimulationMode::Single(config) => vec![config],
            SimulationMode::Suite(configs) => configs,
        }
    }
}

impl ImportExport for SimulationMode {}

#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default = "defaults::committee_size")]
    pub committee_size: usize,
    #[serde(default = "defaults::latency_min_ms")]
    pub latency_min_ms: u64,
    #[serde(default = "defaults::latency_max_ms")]
    pub latency_max_ms: u64,
    #[serde(default)]
    pub topology: NetworkTopology,
    #[serde(default = "defaults::duration_secs")]
    pub duration_secs: u64,
    #[serde(default)]
    pub rng_seed: u64,
    #[serde(default)]
    pub replica_parameters: ReplicaParameters,
    #[serde(default = "defaults::load_generator")]
    pub load_generator: Option<LoadGeneratorConfig>,
    /// Optional oscillating scenario. When non-empty, the runner walks these
    /// phases in order — reconfiguring the network and sampling per-replica
    /// latency between phases — instead of applying a single static topology.
    /// The top-level `topology`, latency, and `duration_secs` fields are then
    /// ignored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phases: Vec<SimulationPhase>,
}

/// One stage of an oscillating simulation: a network condition held for a
/// fixed duration, with per-replica latency sampled at a regular interval.
#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationPhase {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub topology: NetworkTopology,
    /// Latency range applied to every link not touching a slow validator.
    #[serde(default = "defaults::latency_min_ms")]
    pub latency_min_ms: u64,
    #[serde(default = "defaults::latency_max_ms")]
    pub latency_max_ms: u64,
    /// Indices of "slow" validators. A link is degraded (uses the slow latency
    /// range) when either of its endpoints is in this set. Empty means a
    /// uniform network — every link uses `latency_min/max_ms`. Listing `f`
    /// validators models the realistic case where a fast quorum of `2f+1`
    /// keeps consensus progressing and only the leader's speed varies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slow_validators: Vec<usize>,
    /// Latency range for links touching a slow validator. Ignored when
    /// `slow_validators` is empty.
    #[serde(default = "defaults::slow_latency_min_ms")]
    pub slow_latency_min_ms: u64,
    #[serde(default = "defaults::slow_latency_max_ms")]
    pub slow_latency_max_ms: u64,
    /// Validators that do *not* receive a slow validator's blocks in time.
    /// Only the `slow_validator -> slow_blind` *direction* is degraded; the
    /// reverse stays fast, so the slow validator keeps receiving blocks and
    /// producing its own on schedule. Its leader block then reaches just a
    /// partial quorum — too few for a direct commit, too many for a direct
    /// skip — leaving the leader *undecided* and stalling the commit pipeline.
    /// Empty means every direction touching a slow validator is degraded (the
    /// leader then reaches no one and is cleanly skipped instead).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slow_blind: Vec<usize>,
    pub duration_secs: u64,
    #[serde(default = "defaults::sample_interval_secs")]
    pub sample_interval_secs: u64,
}

impl Default for SimulationPhase {
    fn default() -> Self {
        Self {
            name: None,
            topology: NetworkTopology::default(),
            latency_min_ms: defaults::latency_min_ms(),
            latency_max_ms: defaults::latency_max_ms(),
            slow_validators: Vec::new(),
            slow_latency_min_ms: defaults::slow_latency_min_ms(),
            slow_latency_max_ms: defaults::slow_latency_max_ms(),
            slow_blind: Vec::new(),
            duration_secs: 0,
            sample_interval_secs: defaults::sample_interval_secs(),
        }
    }
}

impl SimulationPhase {
    /// One-way latency range for messages sent **from** `from` **to** `to`.
    ///
    /// The degraded range is applied *directionally*:
    /// * With `slow_blind` empty, any direction touching a slow validator is
    ///   degraded — the slow validator is fully isolated.
    /// * Otherwise only the `slow_validator -> slow_blind` direction is
    ///   degraded. The reverse (`slow_blind -> slow_validator`) stays fast, so
    ///   a slow validator still *receives* blocks promptly and keeps producing
    ///   its own on schedule; only its outgoing leader block is withheld from
    ///   the blind validators. That leaves the leader on a partial quorum —
    ///   too few votes to commit, too few blames to skip — i.e. *undecided*.
    ///   A *symmetric* slowdown would instead stall the slow validator itself:
    ///   it could not gather a quorum to build its leader block on time, so
    ///   that block would land late everywhere and be cleanly skipped.
    ///
    /// Directions between two slow validators stay fast — they still reach
    /// each other.
    pub fn link_latency_range(&self, from: usize, to: usize) -> Range<Duration> {
        let from_slow = self.slow_validators.contains(&from);
        let to_slow = self.slow_validators.contains(&to);
        let degraded = if !(from_slow || to_slow) {
            false
        } else if self.slow_blind.is_empty() {
            true
        } else {
            from_slow && self.slow_blind.contains(&to)
        };
        if degraded {
            Self::range(self.slow_latency_min_ms, self.slow_latency_max_ms)
        } else {
            Self::range(self.latency_min_ms, self.latency_max_ms)
        }
    }

    fn range(min_ms: u64, max_ms: u64) -> Range<Duration> {
        assert!(
            min_ms <= max_ms,
            "latency min ({min_ms}) must not exceed max ({max_ms})"
        );
        Duration::from_millis(min_ms)..Duration::from_millis(max_ms)
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs(self.duration_secs)
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            name: None,
            committee_size: defaults::committee_size(),
            latency_min_ms: defaults::latency_min_ms(),
            latency_max_ms: defaults::latency_max_ms(),
            topology: NetworkTopology::default(),
            duration_secs: defaults::duration_secs(),
            rng_seed: 0,
            replica_parameters: ReplicaParameters::default(),
            load_generator: Some(LoadGeneratorConfig::new_for_test()),
            phases: Vec::new(),
        }
    }
}

impl SimulationConfig {
    pub fn latency_range(&self) -> Range<Duration> {
        assert!(
            self.latency_min_ms <= self.latency_max_ms,
            "latency_min_ms ({}) must not exceed latency_max_ms ({})",
            self.latency_min_ms,
            self.latency_max_ms
        );
        let min = Duration::from_millis(self.latency_min_ms);
        let max = Duration::from_millis(self.latency_max_ms);
        min..max
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs(self.duration_secs)
    }
}

impl ImportExport for SimulationConfig {}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub enum NetworkTopology {
    #[default]
    FullMesh,
    OneDown(usize),
    Partition(Vec<Vec<usize>>),
    Star(usize),
}

impl fmt::Display for NetworkTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullMesh => f.write_str("full mesh"),
            Self::OneDown(index) => write!(f, "one down ({index})"),
            Self::Star(center) => write!(f, "star (center={center})"),
            Self::Partition(groups) => {
                f.write_str("partition (")?;
                for (i, group) in groups.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    f.write_str("[")?;
                    for (j, index) in group.iter().enumerate() {
                        if j > 0 {
                            f.write_str(",")?;
                        }
                        write!(f, "{index}")?;
                    }
                    f.write_str("]")?;
                }
                f.write_str(")")
            }
        }
    }
}

mod defaults {
    use replica::config::LoadGeneratorConfig;

    pub fn committee_size() -> usize {
        10
    }
    pub fn latency_min_ms() -> u64 {
        50
    }
    pub fn latency_max_ms() -> u64 {
        100
    }
    pub fn duration_secs() -> u64 {
        20
    }
    pub fn sample_interval_secs() -> u64 {
        2
    }
    pub fn slow_latency_min_ms() -> u64 {
        1500
    }
    pub fn slow_latency_max_ms() -> u64 {
        2000
    }
    pub fn load_generator() -> Option<LoadGeneratorConfig> {
        Some(LoadGeneratorConfig::new_for_test())
    }
}
