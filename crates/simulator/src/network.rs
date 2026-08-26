// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, fmt::Debug, ops::Range, time::Duration};

use rand::Rng;
use tokio::sync::mpsc;

use super::context::SimulatorContext;
use dag::committee::Committee;
use dag::context::Ctx;
use dag::sync::network::{Connection, Network};

pub struct SimulatedNetwork {
    senders: Vec<mpsc::Sender<Connection>>,
    /// Kill handles for the latency tasks backing each active link, keyed by
    /// the ordered `(a, b)` pair. Dropping a pair's senders makes its latency
    /// tasks exit, tearing the old link down so it can be re-established at a
    /// different latency. This is what lets a simulation oscillate between
    /// network conditions mid-run.
    active_kills: HashMap<(usize, usize), Vec<mpsc::Sender<()>>>,
}

impl SimulatedNetwork {
    pub fn new(committee: &Committee) -> (SimulatedNetwork, Vec<Network>) {
        let (networks, senders): (Vec<_>, Vec<_>) = committee
            .authorities()
            .map(|_| {
                let (sender, receiver) = mpsc::channel(16);
                (Network::new_from_raw(receiver), sender)
            })
            .unzip();
        (
            Self {
                senders,
                active_kills: HashMap::new(),
            },
            networks,
        )
    }

    /// Connect every pair of peers. `link_latency(from, to)` gives the one-way
    /// latency range for messages from `from` to `to`, so heterogeneous and
    /// asymmetric networks (e.g. a few slow validators) are expressed by the
    /// caller.
    pub async fn connect_all<L: Fn(usize, usize) -> Range<Duration>>(&mut self, link_latency: L) {
        self.connect_some(|_, _| true, link_latency).await
    }

    /// Connect the peer pairs for which `should_connect` returns true. The
    /// per-direction latency comes from `link_latency(from, to)`, evaluated
    /// once for each direction so a link may be degraded only one way.
    pub async fn connect_some<F, L>(&mut self, should_connect: F, link_latency: L)
    where
        F: Fn(usize, usize) -> bool,
        L: Fn(usize, usize) -> Range<Duration>,
    {
        for a in 0..self.senders.len() {
            for b in a + 1..self.senders.len() {
                if should_connect(a, b) {
                    self.connect(a, b, link_latency(a, b), link_latency(b, a))
                        .await
                }
            }
        }
    }

    /// Tear down every active link. Pairs that the next topology does not
    /// reconnect would otherwise linger with stale latency tasks.
    pub fn disconnect_all(&mut self) {
        self.active_kills.clear();
    }

    /// Establish the link between peers `a` and `b`. The two directions are
    /// independent: `latency_a_to_b` delays messages `a` sends to `b`, and
    /// `latency_b_to_a` delays messages `b` sends to `a`, so a link can be
    /// slow only one way.
    async fn connect(
        &mut self,
        a: usize,
        b: usize,
        latency_a_to_b: Range<Duration>,
        latency_b_to_a: Range<Duration>,
    ) {
        // Drop any latency tasks left over from a previous `connect` of this
        // pair before installing the new link.
        self.active_kills.remove(&(a, b));

        // Channel A carries b -> a traffic; channel B carries a -> b traffic.
        let (a_sender, a_receiver, kill_a) = Self::latency_channel(latency_b_to_a);
        let (b_sender, b_receiver, kill_b) = Self::latency_channel(latency_a_to_b);
        let a_connection = Connection {
            peer_id: b,
            sender: b_sender,
            receiver: a_receiver,
        };
        let b_connection = Connection {
            peer_id: a,
            sender: a_sender,
            receiver: b_receiver,
        };
        self.senders[a].send(a_connection).await.ok();
        self.senders[b].send(b_connection).await.ok();
        self.active_kills.insert((a, b), vec![kill_a, kill_b]);
    }

    /// Spawn a latency task that delays each message by a random draw from
    /// `latency` before forwarding it.
    ///
    /// The link models *propagation delay*, not a serial pipe: every message
    /// is scheduled for delivery at `now + delay` (never earlier than the
    /// message before it, so the link stays FIFO) and many messages can be in
    /// flight at once. Sleeping per message instead would cap the link at one
    /// message per `delay` — a 1-s link could then carry at most ~1 block/s
    /// and would throttle the whole DAG rather than delay it.
    ///
    /// The returned kill sender, when dropped, makes the task stop accepting
    /// messages; messages already in flight are still delivered.
    fn latency_channel<T: Send + 'static + Debug>(
        latency: Range<Duration>,
    ) -> (mpsc::Sender<T>, mpsc::Receiver<T>, mpsc::Sender<()>) {
        let (buf_sender, mut buf_receiver) = mpsc::channel(16);
        let (sender, receiver) = mpsc::channel(16);
        let (kill_sender, mut kill_receiver) = mpsc::channel::<()>(1);
        SimulatorContext::spawn(async move {
            let mut last_delivery = Duration::ZERO;
            loop {
                tokio::select! {
                    message = buf_receiver.recv() => {
                        let Some(message) = message else { return };
                        let delay =
                            SimulatorContext::with_rng(|rng| rng.gen_range(latency.clone()));
                        let now = SimulatorContext::time();
                        let deliver_at =
                            (now + delay).max(last_delivery + Duration::from_nanos(1));
                        last_delivery = deliver_at;
                        let sender = sender.clone();
                        SimulatorContext::spawn(async move {
                            SimulatorContext::sleep(deliver_at - now).await;
                            sender.send(message).await.ok();
                        });
                    }
                    _ = kill_receiver.recv() => return,
                }
            }
        });
        (buf_sender, receiver, kill_sender)
    }
}
