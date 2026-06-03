//! Production [`TriggerSource`]: the Event Bus consumer adapter.
//!
//! Wraps `os_sdk::UnixEventConsumer` (subscribe, framing, auto-reconnect)
//! and decodes each protobuf `Event` envelope into the [`AgentEvent`] the
//! router/engine work on: the event type, the payload fields the filters
//! read, and the external-content origin flag.
//!
//! The decode is pure and unit-tested; the subscription itself is thin I/O
//! exercised only against a live bus.

use std::collections::{BTreeMap, BTreeSet};

use os_sdk::proto::{Event, FileOpenedPayload, WindowFocusedPayload};
use os_sdk::{EventConsumer, SubscribeError, UnixEventConsumer};
use prost::Message as _;
use tokio::sync::mpsc;

use crate::behaviour::TriggerKind;
use crate::loader::LoadedBehaviour;
use crate::seams::{AgentEvent, TriggerSource};

/// The Event Bus type filters to subscribe to: the event pattern of every
/// enabled, event-triggered behaviour. Schedule and manual triggers are not
/// bus subscriptions (a scheduler / explicit invocation drives those), so
/// they contribute nothing here.
pub fn subscription_types(behaviours: &[LoadedBehaviour]) -> Vec<String> {
    let mut types: BTreeSet<String> = BTreeSet::new();
    for lb in behaviours {
        if lb.status.is_enabled() && lb.behaviour.manifest.trigger.kind == TriggerKind::Event {
            if let Some(event) = &lb.behaviour.manifest.trigger.event {
                types.insert(event.clone());
            }
        }
    }
    types.into_iter().collect()
}

/// The Event Bus consumer socket. The daemon resolves the real path from
/// `LUNARIS_CONSUMER_SOCKET` (with this as the fallback).
pub const DEFAULT_CONSUMER_SOCKET: &str = "/run/lunaris/event-bus-consumer.sock";

/// A [`TriggerSource`] backed by an Event Bus subscription.
pub struct EventBusSource {
    rx: mpsc::Receiver<Event>,
}

impl EventBusSource {
    /// Subscribe to the given event-type filters on the consumer socket and
    /// return a source that yields decoded [`AgentEvent`]s. Fails if the bus
    /// is unreachable after the SDK's eager-retry budget.
    pub async fn subscribe(
        socket_path: impl Into<String>,
        types: Vec<String>,
    ) -> Result<Self, SubscribeError> {
        let consumer = UnixEventConsumer::new(socket_path);
        let rx = consumer.subscribe(types).await?;
        Ok(Self { rx })
    }
}

impl TriggerSource for EventBusSource {
    async fn recv(&mut self) -> Option<AgentEvent> {
        self.rx.recv().await.map(decode_event)
    }
}

/// Decode a bus `Event` envelope into an [`AgentEvent`]: the type, the
/// payload fields the router/filters read, and the external-content flag.
/// A payload that fails to decode yields an event with no fields (rather
/// than dropping it); filters then fail closed on the missing fields.
pub fn decode_event(ev: Event) -> AgentEvent {
    let mut fields = BTreeMap::new();
    match ev.r#type.as_str() {
        "file.opened" => {
            if let Ok(p) = FileOpenedPayload::decode(ev.payload.as_slice()) {
                fields.insert("path".to_string(), p.path);
                fields.insert("app_id".to_string(), p.app_id);
            }
        }
        "window.focused" => {
            if let Ok(p) = WindowFocusedPayload::decode(ev.payload.as_slice()) {
                fields.insert("app_id".to_string(), p.app_id);
                fields.insert("window_title".to_string(), p.window_title);
            }
        }
        // Other event types carry no router-readable fields yet; their
        // payload decoders are added as behaviours need them.
        _ => {}
    }
    AgentEvent {
        // Fail-safe. `Event.source` is producer-supplied and spoofable (the
        // bus only requires it to be non-empty; it authenticates `uid` via
        // SO_PEERCRED, but not the origin), so it must NOT be trusted for
        // the external-content gate. Until the bus stamps an authenticated
        // origin class (privileged-producer registration / peer creds) and
        // S18-A content tagging lands, every bus event is treated as
        // external, so any action it triggers requires confirmation.
        external_content: true,
        id: ev.id,
        event_type: ev.r#type,
        fields,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded_file_opened(path: &str) -> Vec<u8> {
        FileOpenedPayload {
            path: path.to_string(),
            app_id: "org.lunaris.editor".to_string(),
            flags: 0,
        }
        .encode_to_vec()
    }

    #[test]
    fn decodes_file_opened_fields() {
        let ev = Event {
            id: "e1".to_string(),
            r#type: "file.opened".to_string(),
            source: "ebpf".to_string(),
            payload: encoded_file_opened("~/Repositories/foo.rs"),
            ..Default::default()
        };
        let agent_event = decode_event(ev);
        assert_eq!(agent_event.event_type, "file.opened");
        assert_eq!(agent_event.fields.get("path").unwrap(), "~/Repositories/foo.rs");
        assert_eq!(agent_event.fields.get("app_id").unwrap(), "org.lunaris.editor");
    }

    #[test]
    fn source_is_not_trusted_for_external_content() {
        // A producer can spoof `source`, so even a claimed local "ebpf"
        // origin must still be treated as external (fail-safe) until the
        // bus authenticates the origin.
        for source in ["ebpf", "wayland", "app:com.example.thing", ""] {
            let ev = Event {
                id: "e".to_string(),
                r#type: "file.opened".to_string(),
                source: source.to_string(),
                payload: encoded_file_opened("~/foo.rs"),
                ..Default::default()
            };
            assert!(
                decode_event(ev).external_content,
                "source {source:?} must not downgrade external_content"
            );
        }
    }

    #[test]
    fn unknown_event_type_decodes_with_no_fields() {
        let ev = Event {
            id: "e3".to_string(),
            r#type: "app.action".to_string(),
            source: "ebpf".to_string(),
            payload: vec![1, 2, 3], // not decoded for this type
            ..Default::default()
        };
        let agent_event = decode_event(ev);
        assert_eq!(agent_event.event_type, "app.action");
        assert!(agent_event.fields.is_empty());
    }

    #[test]
    fn a_corrupt_payload_yields_no_fields_not_a_drop() {
        let ev = Event {
            id: "e4".to_string(),
            r#type: "file.opened".to_string(),
            source: "ebpf".to_string(),
            payload: vec![0xff, 0xff, 0xff], // invalid protobuf
            ..Default::default()
        };
        let agent_event = decode_event(ev);
        assert_eq!(agent_event.event_type, "file.opened");
        assert!(agent_event.fields.is_empty()); // filters then fail closed
    }

    #[test]
    fn subscription_types_collects_enabled_event_triggers() {
        use crate::behaviour::parse;
        use crate::loader::{LoadedBehaviour, Provenance, Status};
        use std::path::PathBuf;

        let load = |skill: &str, status: Status| LoadedBehaviour {
            behaviour: parse(skill).expect("valid"),
            provenance: Provenance::BuiltIn,
            dir: PathBuf::from("/test"),
            status,
        };
        let event_wf = |name: &str, event: &str| {
            format!("---\nname: {name}\ndescription: d\nkind: workflow\nhandler: h\ntrigger:\n  type: event\n  event: {event}\n---\n")
        };
        let behaviours = [
            load(&event_wf("a", "file.opened"), Status::Enabled),
            load(&event_wf("b", "window.focused"), Status::Enabled),
            // Disabled behaviours and schedule/manual triggers contribute nothing.
            load(&event_wf("c", "clipboard.copy"), Status::Disabled(crate::loader::DisableReason::NotEnabledInSettings)),
            load("---\nname: d\ndescription: d\nkind: workflow\nhandler: h\ntrigger:\n  type: manual\n---\n", Status::Enabled),
        ];
        assert_eq!(
            subscription_types(&behaviours),
            vec!["file.opened".to_string(), "window.focused".to_string()]
        );
    }
}
