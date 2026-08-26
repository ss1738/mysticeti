// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{
    io,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use consensus::committer::Committer;
use dag::{
    authority::Authority,
    config::{ConfigError, ImportExport},
    consensus::CommittedSubDag,
    context::Ctx,
    core::syncer::Syncer,
    metrics::Metrics,
    storage::Storage,
};
use rand::{SeedableRng, rngs::StdRng};
use replica::{
    builder::{ReplicaBuilder, StorageKind},
    config::{LoadGeneratorConfig, PrivateReplicaConfig, PublicReplicaConfig},
    replica::ReplicaHandle,
    result::{LatencySample, RunKind, RunResult},
};
use tokio::sync::mpsc;

use crate::{
    config::{NetworkTopology, SimulationConfig},
    context::SimulatorContext,
    executor::{JoinHandle, SimulatorExecutor},
    network::SimulatedNetwork,
    tracing::SimulatorTracing,
};

pub struct SimulationRunner {
    config: SimulationConfig,
}

impl SimulationRunner {
    pub fn new(config: SimulationConfig) -> Self {
        Self { config }
    }

    pub fn from_yaml(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let config = SimulationConfig::load(path)?;
        Ok(Self::new(config))
    }

    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    /// Run the simulation to completion and return the result.
    ///
    /// Executes inside a deterministic discrete-event simulator:
    /// all time is simulated, no real wall-clock time elapses.
    ///
    /// With no `phases` configured the runner applies a single static
    /// topology for `duration_secs`. With `phases` set it instead walks each
    /// phase — reconfiguring the network and sampling per-replica latency at
    /// the phase's interval — and the result carries the latency time-series.
    pub fn run(self) -> io::Result<RunResult<SimulationConfig>> {
        let _guard = SimulatorTracing::new().setup().ok();
        let rng = StdRng::seed_from_u64(self.config.rng_seed);
        let Self { config } = self;
        SimulatorExecutor::run(rng, async move {
            let mut state = SimulationState::setup(config).await;
            let latency_series = state.run_scenario().await;
            state.collect_result(latency_series).await
        })
    }
}

struct SimulationState {
    config: SimulationConfig,
    network: SimulatedNetwork,
    replicas: Vec<ReplicaHandle<SimulatorContext>>,
    /// Per-replica metrics handles, retained so the phased runner can sample
    /// latency mid-run. These are the same `Arc<Metrics>` injected into each
    /// replica at build time.
    metrics: Vec<Arc<Metrics>>,
    /// JoinHandles for any load generators we started, so they stay alive for the duration
    /// of the simulation.
    _load_generators: Vec<JoinHandle<()>>,
}

impl SimulatedNetwork {
    /// A fully connected committee with default parameters. The returned network must be
    /// kept alive for the duration of the run.
    #[cfg(any(test, feature = "test-utils"))]
    pub async fn new_for_test(
        committee_size: usize,
    ) -> (Self, Vec<ReplicaHandle<SimulatorContext>>) {
        Self::new_for_test_with_commit_consumers(vec![None; committee_size]).await
    }

    /// Like [`SimulatedNetwork::new_for_test`], with one optional commit consumer per
    /// replica.
    #[cfg(any(test, feature = "test-utils"))]
    pub async fn new_for_test_with_commit_consumers(
        commit_consumers: Vec<Option<mpsc::Sender<CommittedSubDag>>>,
    ) -> (Self, Vec<ReplicaHandle<SimulatorContext>>) {
        let public_config = PublicReplicaConfig::new_for_tests(commit_consumers.len());
        let latency_range = Duration::from_millis(50)..Duration::from_millis(100);
        let (mut network, replicas, _, _) =
            SimulationState::build_replicas(public_config, None, commit_consumers).await;
        network
            .connect_all(move |_, _| latency_range.clone())
            .await;
        (network, replicas)
    }
}

impl SimulationState {
    /// Build a committee of simulated replicas wired through a [`SimulatedNetwork`]. The
    /// committee is derived from `public_config`; `commit_consumers` holds one optional
    /// consumer per replica. Links are created later by the caller (see
    /// [`SimulatedNetwork::connect_all`]), which is where each link's latency is set.
    /// Also returns every replica's metrics handle so the phased runner can sample
    /// latency mid-run.
    async fn build_replicas(
        public_config: PublicReplicaConfig,
        load_generator: Option<LoadGeneratorConfig>,
        commit_consumers: Vec<Option<mpsc::Sender<CommittedSubDag>>>,
    ) -> (
        SimulatedNetwork,
        Vec<ReplicaHandle<SimulatorContext>>,
        Vec<Arc<Metrics>>,
        Vec<JoinHandle<()>>,
    ) {
        let committee = public_config.committee();
        let committee_size = committee.len();
        assert_eq!(commit_consumers.len(), committee_size);
        let (network, networks) = SimulatedNetwork::new(&committee);

        // The simulator doesn't touch disk; the WAL path in the private
        // configs is unused once we override storage with `InMemory`.
        let private_configs =
            PrivateReplicaConfig::new_for_benchmarks(&PathBuf::from("simulator"), committee_size);

        let mut replicas = Vec::with_capacity(committee_size);
        let mut metrics = Vec::with_capacity(committee_size);
        let mut load_generators = Vec::new();
        for (i, ((node_network, private_config), commit_consumer)) in networks
            .into_iter()
            .zip(private_configs)
            .zip(commit_consumers)
            .enumerate()
        {
            let authority = Authority::from(i);
            let node_metrics = Metrics::new_for_test(committee_size);
            metrics.push(node_metrics.clone());
            let mut builder = ReplicaBuilder::new(authority, public_config.clone(), private_config)
                .with_storage(StorageKind::Ephemeral)
                .with_crypto_disabled()
                .with_metrics(node_metrics)
                .with_network(node_network);
            if let Some(commit_consumer) = commit_consumer {
                builder = builder.with_commit_consumer(commit_consumer);
            }
            let mut handle = builder
                .build()
                .run::<SimulatorContext>()
                .await
                .expect("simulator replica build must not fail");
            if let Some(load_generator) = load_generator.clone() {
                load_generators.push(handle.start_load_generator(load_generator));
            }
            replicas.push(handle);
        }
        (network, replicas, metrics, load_generators)
    }

    async fn setup(config: SimulationConfig) -> Self {
        let public_config = PublicReplicaConfig::new_for_tests(config.committee_size)
            .with_parameters(config.replica_parameters.clone());
        let commit_consumers = vec![None; config.committee_size];
        let (network, replicas, metrics, load_generators) =
            Self::build_replicas(public_config, config.load_generator.clone(), commit_consumers)
                .await;

        Self {
            config,
            network,
            replicas,
            metrics,
            _load_generators: load_generators,
        }
    }

    /// Drive the network through the configured scenario. Returns the latency
    /// time-series (empty for a static, non-phased run).
    async fn run_scenario(&mut self) -> Vec<LatencySample> {
        if self.config.phases.is_empty() {
            let topology = self.config.topology.clone();
            let latency = self.config.latency_range();
            self.apply_topology(&topology, move |_, _| latency.clone())
                .await;
            SimulatorContext::sleep(self.config.duration()).await;
            Vec::new()
        } else {
            self.run_phased().await
        }
    }

    /// Walk the configured phases: reconfigure the network for each, then
    /// sample every replica's windowed latency at the phase's interval.
    async fn run_phased(&mut self) -> Vec<LatencySample> {
        let phases = self.config.phases.clone();
        let mut samples = Vec::new();
        // Per-replica (cumulative_sum_ms, cumulative_count) at the previous
        // sample, used to derive a windowed mean by differencing.
        let mut previous = vec![(0.0_f64, 0_u64); self.metrics.len()];
        let mut elapsed_total = 0u64;

        for (index, phase) in phases.iter().enumerate() {
            let phase_number = (index + 1) as u8;
            // Tear every link down before installing the phase's topology so
            // links the new topology omits do not linger at the old latency.
            self.network.disconnect_all();
            self.apply_topology(&phase.topology, |a, b| phase.link_latency_range(a, b))
                .await;

            let interval = phase.sample_interval_secs.max(1);
            let mut elapsed_phase = 0u64;
            while elapsed_phase < phase.duration_secs {
                let step = interval.min(phase.duration_secs - elapsed_phase);
                SimulatorContext::sleep(Duration::from_secs(step)).await;
                elapsed_phase += step;
                elapsed_total += step;

                for (validator, metrics) in self.metrics.iter().enumerate() {
                    let snapshot = metrics.collect();
                    let latency = snapshot.precise_committed_latency();
                    let leaders = snapshot.leaders_per_round();
                    let (previous_sum, previous_count) = previous[validator];
                    let delta_count = latency.cumulative_count.saturating_sub(previous_count);
                    let avg_ms = if delta_count > 0 {
                        (latency.cumulative_sum_ms - previous_sum) / delta_count as f64
                    } else {
                        0.0
                    };
                    previous[validator] = (latency.cumulative_sum_ms, latency.cumulative_count);
                    samples.push(LatencySample {
                        time_s: elapsed_total,
                        validator,
                        p90_ms: latency.p90_ms.round() as u64,
                        avg_ms: avg_ms.round() as u64,
                        phase: phase_number,
                        commits: delta_count,
                        leaders,
                    });
                }
            }
        }
        samples
    }

    async fn apply_topology<L>(&mut self, topology: &NetworkTopology, link_latency: L)
    where
        L: Fn(usize, usize) -> Range<Duration>,
    {
        match topology {
            NetworkTopology::FullMesh => {
                self.network.connect_all(link_latency).await;
            }
            NetworkTopology::OneDown(node) => {
                let excluded = *node;
                self.network
                    .connect_some(|a, _b| a != excluded, link_latency)
                    .await;
            }
            NetworkTopology::Partition(groups) => {
                self.network
                    .connect_some(
                        |a, b| {
                            groups
                                .iter()
                                .any(|group| group.contains(&a) && group.contains(&b))
                        },
                        link_latency,
                    )
                    .await;
            }
            NetworkTopology::Star(center) => {
                let center = *center;
                self.network
                    .connect_some(|a, b| a == center || b == center, link_latency)
                    .await;
            }
        }
    }

    async fn collect_result(
        self,
        latency_series: Vec<LatencySample>,
    ) -> io::Result<RunResult<SimulationConfig>> {
        let Self {
            config, replicas, ..
        } = self;
        let duration = if config.phases.is_empty() {
            config.duration()
        } else {
            Duration::from_secs(config.phases.iter().map(|p| p.duration_secs).sum())
        };
        let syncers: Vec<Syncer<SimulatorContext, Committer>> =
            futures::future::join_all(replicas.into_iter().map(ReplicaHandle::shutdown)).await;
        let metrics: Vec<_> = syncers
            .iter()
            .map(|syncer| syncer.core().metrics.collect())
            .collect();
        let storages: Vec<Storage> = syncers.into_iter().map(Syncer::into_storage).collect();

        Ok(
            RunResult::new(metrics, storages, config, duration, RunKind::Simulation)
                .with_latency_series(latency_series),
        )
    }
}
