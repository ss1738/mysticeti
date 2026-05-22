// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive leader-scheduling experiment.
//!
//! Runs the same oscillating 3-phase scenario — healthy network, sustained
//! high latency, healthy again — under three consensus configurations:
//!
//! * `adaptive`  — the leader scheduler is on; it should track the network,
//!   behaving like a low leader count when latency is good and ramping up
//!   when it degrades.
//! * `fixed_1`   — a single leader per round. Cheap when the network is
//!   healthy, but a single slow leader stalls progress under high latency.
//! * `fixed_5`   — five leaders per round. Robust under high latency, but
//!   pays redundant work when the network is healthy.
//!
//! Each run writes a `time_s,validator,p90_ms,avg_ms,phase` CSV under
//! `results/<protocol>/<variant>/`. `scripts/plot_adaptive_scheduling_normalized.py`
//! pools those CSVs into the per-protocol comparison plot. The expected story:
//! adaptive beats `fixed_1` in the healthy phases and beats `fixed_5` in the
//! degraded phase, because it adapts to whichever regime it is in.
//!
//! These are long-running experiments, not CI checks, so they are
//! `#[ignore]`d by default. Run them via `scripts/run_protocol.sh <protocol>`
//! (which passes `--ignored`).

use std::{
    fs,
    io::Write,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use consensus::protocol::ConsensusProtocol;
use replica::{
    config::ReplicaParameters,
    result::{LatencySample, Outcome},
};
use simulator::{SimulationConfig, SimulationPhase, SimulationRunner};

/// Mysticeti protocol selection with explicit scheduler settings.
fn mysticeti(leader_count: usize, adaptive_scheduler: bool) -> ConsensusProtocol {
    ConsensusProtocol::Mysticeti {
        leader_count: NonZeroUsize::new(leader_count).expect("leader_count must be non-zero"),
        adaptive_scheduler,
        max_leaders_per_round: NonZeroUsize::new(5).unwrap(),
        scheduler_interval: 25,
    }
}

/// Blue Bottle protocol selection with explicit scheduler settings.
fn blue_bottle(leader_count: usize, adaptive_scheduler: bool) -> ConsensusProtocol {
    ConsensusProtocol::BlueBottle {
        leader_count: NonZeroUsize::new(leader_count).expect("leader_count must be non-zero"),
        adaptive_scheduler,
        max_leaders_per_round: NonZeroUsize::new(5).unwrap(),
        scheduler_interval: 25,
    }
}

/// Nemo-Nemo protocol selection with explicit scheduler settings.
fn nemo_nemo(leader_count: usize, adaptive_scheduler: bool) -> ConsensusProtocol {
    ConsensusProtocol::NemoNemo {
        leader_count: NonZeroUsize::new(leader_count).expect("leader_count must be non-zero"),
        adaptive_scheduler,
        max_leaders_per_round: NonZeroUsize::new(5).unwrap(),
        scheduler_interval: 25,
    }
}

/// Mahi-Mahi protocol selection with explicit wave length and scheduler settings.
fn mahi_mahi(leader_count: usize, wave_length: u64, adaptive_scheduler: bool) -> ConsensusProtocol {
    ConsensusProtocol::MahiMahi {
        leader_count: NonZeroUsize::new(leader_count).expect("leader_count must be non-zero"),
        wave_length,
        adaptive_scheduler,
        max_leaders_per_round: NonZeroUsize::new(5).unwrap(),
        scheduler_interval: 25,
    }
}

/// Committee size for the experiment: 3f+1 with f = 3.
const COMMITTEE_SIZE: usize = 10;
/// The `f` validators whose blocks propagate slowly in the degraded phase.
const SLOW_VALIDATORS: [usize; 3] = [7, 8, 9];
/// Validators kept "blind" to slow-validator blocks in the degraded phase. A
/// slow validator's leader block then reaches only the 5 non-blind validators
/// — enough votes to avoid a direct skip, too few for a direct commit — so the
/// leader is left *undecided*.
const BLIND_VALIDATORS: [usize; 2] = [3, 4];

/// Directory holding the CSVs for one protocol/variant pair:
/// `results/<protocol>/<variant>/`.
fn results_dir(protocol: &str, variant: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("results")
        .join(protocol)
        .join(variant)
}

/// Per-run index used to pick the rotation offset and RNG seed. Honors the
/// `RUN_INDEX` environment variable when set (e.g. by a parallel driver
/// script); otherwise it is the count of CSVs already written for this
/// protocol/variant, so sequential `cargo test` invocations auto-increment
/// with no setup — each run gets a fresh rotation and seed.
fn run_index(protocol: &str, variant: &str) -> usize {
    if let Some(k) = std::env::var("RUN_INDEX").ok().and_then(|v| v.parse().ok()) {
        return k;
    }
    fs::read_dir(results_dir(protocol, variant))
        .map(|entries| entries.filter_map(|e| e.ok()).count())
        .unwrap_or(0)
}

/// Rotate a validator set by `k` positions within the committee. Rotating the
/// slow and blind sets together by the same offset is a rigid relabelling: it
/// preserves the 5-supporter undecided mechanism exactly while shifting which
/// rounds of the fixed round-robin carry a slow leader. Pooling runs with
/// distinct `k` therefore decorrelates the round-robin's 10-round cycle.
fn rotate(base: &[usize], k: usize) -> Vec<usize> {
    base.iter().map(|v| (v + k) % COMMITTEE_SIZE).collect()
}

/// Healthy → degraded → healthy, all full-mesh.
///
/// In the degraded phase each slow validator's links to `BLIND_VALIDATORS` are
/// slow, so its leader block reaches only a partial quorum (5 of 10): too many
/// to be directly skipped, too few to be directly committed. The leader is
/// left **undecided**, which stalls every leader sequenced behind it. A single
/// leader per round hits one undecided leader ~30% of rounds; with five
/// leaders per round ~70% of rounds carry an undecided leader, so more leaders
/// is *worse* under degradation — the regime that makes the adaptive scheduler
/// pay off.
fn three_phase_config(consensus: ConsensusProtocol, k: usize) -> SimulationConfig {
    let healthy = |duration_secs| SimulationPhase {
        name: Some("healthy".to_string()),
        latency_min_ms: 50,
        latency_max_ms: 100,
        duration_secs,
        sample_interval_secs: 3,
        ..Default::default()
    };
    let degraded = SimulationPhase {
        name: Some("high-latency".to_string()),
        latency_min_ms: 50,
        latency_max_ms: 100,
        slow_validators: rotate(&SLOW_VALIDATORS, k),
        slow_latency_min_ms: 5200,
        slow_latency_max_ms: 7500,
        slow_blind: rotate(&BLIND_VALIDATORS, k),
        duration_secs: 500,
        sample_interval_secs: 10,
        ..Default::default()
    };
    let _degraded_uniform = SimulationPhase {
        name: Some("high-latency".to_string()),
        latency_min_ms: 50,
        latency_max_ms: 1600,
        duration_secs: 300,
        sample_interval_secs: 10,
        ..Default::default()
    };

    SimulationConfig {
        committee_size: COMMITTEE_SIZE,
        rng_seed: k as u64,
        replica_parameters: ReplicaParameters {
            consensus,
            ..Default::default()
        },
        phases: vec![healthy(50), degraded, healthy(50)],
        ..Default::default()
    }
}

/// Next free `results/<protocol>/<variant>/<n>.csv` path, so repeated runs
/// accumulate rather than overwrite (the plot script pools every CSV in a
/// directory).
fn next_csv_path(protocol: &str, variant: &str) -> PathBuf {
    let dir = results_dir(protocol, variant);
    fs::create_dir_all(&dir).expect("failed to create CSV directory");
    let existing = fs::read_dir(&dir)
        .map(|entries| entries.filter_map(|e| e.ok()).count())
        .unwrap_or(0);
    dir.join(format!("{}.csv", existing + 1))
}

fn write_csv(path: &Path, samples: &[LatencySample]) {
    let mut file = fs::File::create(path).expect("failed to create latency CSV");
    writeln!(file, "time_s,validator,p90_ms,avg_ms,phase").unwrap();
    for s in samples {
        writeln!(
            file,
            "{},{},{},{},{}",
            s.time_s, s.validator, s.p90_ms, s.avg_ms, s.phase
        )
        .unwrap();
    }
}

/// Healthy → uniformly-high-latency → healthy, all full-mesh.
///
/// Unlike `three_phase_config`, there are no selective slow validators or
/// blind validators. Every link degrades to 50–1600 ms in the degraded
/// phase, stressing the protocol under general slowness rather than the
/// undecided-leader mechanism.
fn high_latency_config(consensus: ConsensusProtocol, k: usize) -> SimulationConfig {
    let healthy = |duration_secs| SimulationPhase {
        name: Some("healthy".to_string()),
        latency_min_ms: 50,
        latency_max_ms: 100,
        duration_secs,
        sample_interval_secs: 3,
        ..Default::default()
    };
    let degraded = SimulationPhase {
        name: Some("high-latency".to_string()),
        latency_min_ms: 50,
        latency_max_ms: 1600,
        duration_secs: 300,
        sample_interval_secs: 10,
        ..Default::default()
    };
    SimulationConfig {
        committee_size: COMMITTEE_SIZE,
        rng_seed: k as u64,
        replica_parameters: ReplicaParameters {
            consensus,
            ..Default::default()
        },
        phases: vec![healthy(50), degraded, healthy(50)],
        ..Default::default()
    }
}

/// Run one experiment and append its latency CSV under
/// `results/<protocol>/<variant>/`.
fn run_variant_config(
    protocol: &str,
    variant: &str,
    consensus: ConsensusProtocol,
    config_fn: fn(ConsensusProtocol, usize) -> SimulationConfig,
) {
    let k = run_index(protocol, variant);
    let config = config_fn(consensus, k);
    let results = SimulationRunner::new(config)
        .run()
        .expect("simulation run failed");

    assert_ne!(
        results.outcome,
        Outcome::Diverged,
        "{protocol}/{variant}: replicas diverged — safety violation"
    );

    let path = next_csv_path(protocol, variant);
    write_csv(&path, &results.latency_series);
    println!(
        "{protocol}/{variant}: run_index={k}, outcome={:?}, wrote {} latency samples to {}",
        results.outcome,
        results.latency_series.len(),
        path.display()
    );
}

fn run_variant(protocol: &str, variant: &str, consensus: ConsensusProtocol) {
    run_variant_config(protocol, variant, consensus, three_phase_config);
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mysticeti_adaptive() {
    run_variant("mysticeti", "adaptive", mysticeti(5, true));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mysticeti_fixed_1() {
    run_variant("mysticeti", "fixed_1", mysticeti(1, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mysticeti_fixed_5() {
    run_variant("mysticeti", "fixed_5", mysticeti(5, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn blue_bottle_adaptive() {
    run_variant("blue-bottle", "adaptive", blue_bottle(5, true));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn blue_bottle_fixed_1() {
    run_variant("blue-bottle", "fixed_1", blue_bottle(1, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn blue_bottle_fixed_5() {
    run_variant("blue-bottle", "fixed_5", blue_bottle(5, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn nemo_nemo_adaptive() {
    run_variant("nemo-nemo", "adaptive", nemo_nemo(5, true));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn nemo_nemo_fixed_1() {
    run_variant("nemo-nemo", "fixed_1", nemo_nemo(1, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn nemo_nemo_fixed_5() {
    run_variant("nemo-nemo", "fixed_5", nemo_nemo(5, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_4_adaptive() {
    run_variant("mahi-mahi-4", "adaptive", mahi_mahi(5, 4, true));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_4_fixed_1() {
    run_variant("mahi-mahi-4", "fixed_1", mahi_mahi(1, 4, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_4_fixed_5() {
    run_variant("mahi-mahi-4", "fixed_5", mahi_mahi(5, 4, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_5_adaptive() {
    run_variant("mahi-mahi-5", "adaptive", mahi_mahi(5, 5, true));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_5_fixed_1() {
    run_variant("mahi-mahi-5", "fixed_1", mahi_mahi(1, 5, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_5_fixed_5() {
    run_variant("mahi-mahi-5", "fixed_5", mahi_mahi(5, 5, false));
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_5_hl_adaptive() {
    run_variant_config("mahi-mahi-5-hl", "adaptive", mahi_mahi(5, 5, true), high_latency_config);
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_5_hl_fixed_1() {
    run_variant_config("mahi-mahi-5-hl", "fixed_1", mahi_mahi(1, 5, false), high_latency_config);
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn mahi_mahi_5_hl_fixed_5() {
    run_variant_config("mahi-mahi-5-hl", "fixed_5", mahi_mahi(5, 5, false), high_latency_config);
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn cordial_miners_ps() {
    run_variant(
        "cordial-miners-ps",
        "default",
        ConsensusProtocol::CordialMinersPartiallySynchronous,
    );
}

#[test]
#[ignore = "long-running experiment; run via scripts/run_protocol.sh"]
fn cordial_miners_async() {
    run_variant(
        "cordial-miners-async",
        "default",
        ConsensusProtocol::CordialMinersAsynchronous,
    );
}
