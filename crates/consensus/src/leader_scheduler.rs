// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use dag::{
    block::{Block, BlockReference, RoundNumber},
    committee::{Committee, Stake, StakeAggregator},
    data::Data,
    storage::BlockReader,
};

use crate::{leader::LeaderElector, protocol::Protocol};

/// Adaptive controller for the number of leaders per round.
///
/// Every `interval` rounds it measures the direct-commit rate over the most
/// recently committed sub-dag and adjusts the leader count: additive increase
/// when the network is healthy, multiplicative (exponential) backoff when it
/// is not.
pub(crate) struct LeaderScheduler {
    committee: Arc<Committee>,
    leader_elector: LeaderElector,
    /// Wave length of the active protocol.
    wave_length: RoundNumber,
    /// Stake threshold for a direct commit under the active protocol.
    direct_commit_quorum: Stake,
    /// Whether the active protocol pipelines (a leader every round).
    pipeline: bool,
    /// Upper bound on the leader count.
    max_leaders_per_round: usize,
    /// Rounds between reconfigurations.
    interval: RoundNumber,
    /// Round of the last reconfiguration.
    last_round: RoundNumber,
    /// Current number of leaders per round.
    number_of_leaders: usize,
    /// Consecutive backoff steps; drives the exponential decrease.
    consecutive_decreases: u32,
}

impl LeaderScheduler {
    pub(crate) fn new(protocol: &Protocol, committee: Arc<Committee>) -> Self {
        let leader_elector = LeaderElector::new(committee.len());
        Self {
            committee,
            leader_elector,
            wave_length: protocol.wave_length,
            direct_commit_quorum: protocol.direct_commit_quorum,
            pipeline: protocol.pipeline,
            max_leaders_per_round: protocol.max_leaders_per_round.get(),
            interval: protocol.scheduler_interval,
            last_round: 0,
            number_of_leaders: protocol.leader_count.get(),
            consecutive_decreases: 0,
        }
    }

    /// Whether `round` is far enough past the last reconfiguration to trigger
    /// a new one.
    pub(crate) fn is_schedule_round(&self, round: RoundNumber) -> bool {
        if round == 0 || round <= self.last_round {
            return false;
        }
        (round - self.last_round) > self.interval
    }

    /// Measure the commit rate over the latest sub-dag and return the new
    /// leader count. `sequence_leader` is the committed leader at the schedule
    /// round.
    pub(crate) fn update_leader(
        &mut self,
        block_reader: &BlockReader,
        sequence_leader: Data<Block>,
    ) -> usize {
        let anchor_round = sequence_leader.round();
        let subdag = self.collect_subdag(block_reader, sequence_leader);

        let expected_commits = self.calculate_expected_commits(self.number_of_leaders);
        // Avoid `count / 0` (NaN/inf): with nothing expected we cannot measure,
        // so keep the leader count unchanged.
        if expected_commits == 0 {
            self.last_round = anchor_round;
            return self.number_of_leaders;
        }

        let count = self.count_direct_commits(&subdag, block_reader, self.number_of_leaders);
        let rate = count as f64 / expected_commits as f64;

        let new_value = if rate >= 0.96 {
            // Additive increase.
            self.consecutive_decreases = 0;
            (self.number_of_leaders + 1).min(self.max_leaders_per_round)
        } else {
            // Exponential backoff. `2^n` is computed with `checked_shl` so a
            // long-running degraded network cannot overflow the shift.
            let decrement = 1usize
                .checked_shl(self.consecutive_decreases)
                .unwrap_or(usize::MAX);
            self.consecutive_decreases = self.consecutive_decreases.saturating_add(1);
            self.number_of_leaders.saturating_sub(decrement).max(1)
        };

        tracing::info!(
            "Leader scheduler: round {anchor_round}, commit rate {rate:.3} \
             ({count}/{expected_commits}), leaders {} -> {new_value}",
            self.number_of_leaders,
        );

        self.last_round = anchor_round;
        self.number_of_leaders = new_value;
        new_value
    }

    /// Collect the sub-dag rooted at `anchor`, walking back at most `interval`
    /// rounds.
    fn collect_subdag(&self, block_reader: &BlockReader, anchor: Data<Block>) -> Vec<Data<Block>> {
        // `saturating_sub` guards against underflow when the anchor round is
        // below the interval.
        let until = anchor.round().saturating_sub(self.interval);
        let mut visited = HashSet::new();
        let mut to_visit = vec![anchor];
        let mut subdag = Vec::new();

        while let Some(block) = to_visit.pop() {
            if visited.insert(*block.reference()) {
                for include_ref in block.includes() {
                    if include_ref.round() >= until && !visited.contains(include_ref) {
                        if let Some(included) = block_reader.get_block(*include_ref) {
                            to_visit.push(included);
                        }
                    }
                }
                subdag.push(block);
            }
        }
        subdag
    }

    /// Expected number of direct commits in the analysis window.
    fn calculate_expected_commits(&self, leaders_per_round: usize) -> u64 {
        // The two most recent rounds cannot have produced a decision yet.
        let round_count = self.interval.saturating_sub(self.wave_length - 1);
        let wave_length = if self.pipeline { 1 } else { self.wave_length };
        (round_count * leaders_per_round as u64) / wave_length
    }

    /// Count leaders directly committed within `blocks`.
    fn count_direct_commits(
        &self,
        blocks: &[Data<Block>],
        block_reader: &BlockReader,
        leaders_per_round: usize,
    ) -> usize {
        let mut count = 0;

        let mut rounds: Vec<RoundNumber> = blocks.iter().map(|b| b.round()).collect();
        rounds.sort_unstable();
        rounds.dedup();

        for round in rounds {
            let wave = round / self.wave_length;
            // Without pipelining, only the first round of a wave has leaders.
            if !self.pipeline && wave * self.wave_length != round {
                continue;
            }

            for leader_offset in 0..leaders_per_round as u64 {
                let leader = self.leader_elector.elect_leader(round + leader_offset);
                for leader_block in block_reader.get_blocks_at_authority_round(leader, round) {
                    if self.is_directly_committed(&leader_block, blocks) {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Whether `leader` was directly committed within `blocks`, applying the
    /// active protocol's wave length and direct-commit quorum.
    ///
    /// Support is propagated transitively: starting from the leader, each round
    /// up to `decision_round - 1` extends the supporter set to blocks that include
    /// any block from the previous supporter set. At `decision_round` (the last
    /// round of the wave) a block is a certificate if it includes a quorum of
    /// supporters from the preceding round; a quorum of certificates is a direct
    /// commit.
    ///
    /// Wave length 2 is special-cased: the decision round sits one round above
    /// the leader, so votes and certificates coincide (mirroring
    /// `BaseCommitter::is_certificate`) — a decision-round block that includes
    /// the leader is itself a certificate. Works for any wave length ≥ 2.
    fn is_directly_committed(&self, leader: &Data<Block>, blocks: &[Data<Block>]) -> bool {
        let leader_round = leader.round();
        let decision_round = leader_round + self.wave_length - 1;

        // Wave length 2: votes and certificates coincide on the decision round
        // — an `R+1` block that includes the leader is itself a certificate, so
        // the transitive support loop below (empty for wave 2) does not apply.
        if self.wave_length == 2 {
            let mut certificate_stake = StakeAggregator::new(self.direct_commit_quorum);
            for block in blocks.iter().filter(|b| b.round() == decision_round) {
                if block.includes().contains(leader.reference())
                    && certificate_stake.add(block.reference().authority, &self.committee)
                {
                    return true;
                }
            }
            return false;
        }

        // Seed with the leader itself, then propagate support round by round.
        // A block at round r supports the leader if it includes any supporter
        // from round r-1.
        let mut supporters: HashSet<BlockReference> = [*leader.reference()].into();
        for round in (leader_round + 1)..decision_round {
            supporters = blocks
                .iter()
                .filter(|b| b.round() == round)
                .filter(|b| b.includes().iter().any(|r| supporters.contains(r)))
                .map(|b| *b.reference())
                .collect();
            if supporters.is_empty() {
                return false;
            }
        }

        // A decision-round block is a certificate when it includes supporters
        // from a quorum of stake; a quorum of certificates is a direct commit.
        let mut certificate_stake = StakeAggregator::new(self.direct_commit_quorum);
        for block in blocks.iter().filter(|b| b.round() == decision_round) {
            let mut vote_stake = StakeAggregator::new(self.direct_commit_quorum);
            let has_quorum = supporters
                .iter()
                .filter(|s| block.includes().contains(s))
                .any(|s| vote_stake.add(s.authority, &self.committee));
            if has_quorum && certificate_stake.add(block.reference().authority, &self.committee) {
                return true;
            }
        }
        false
    }
}
