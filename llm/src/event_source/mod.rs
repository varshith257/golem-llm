// Based on https://github.com/jpopesculian/eventsource-stream and https://github.com/jpopesculian/reqwest-eventsource
// modified to use the wasi-http based reqwest, and wasi pollables

mod error;
mod event_stream;
mod message_event;
mod parser;
mod utf8_stream;

use crate::event_source::error::Error;
use crate::event_source::event_stream::EventStream;
pub use message_event::MessageEvent;
use reqwest::header::HeaderValue;
use reqwest::{Response, StatusCode};
use std::task::Poll;
use golem_rust::wasm_rpc::Pollable;

/// The ready state of an [`EventSource`]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum ReadyState {
    /// The EventSource is waiting on a response from the endpoint
    Connecting = 0,
    /// The EventSource is connected
    Open = 1,
    /// The EventSource is closed and no longer emitting Events
    Closed = 2,
}

pub struct EventSource {
    response: Option<Response>,
    cur_stream: Option<EventStream>,
    last_event_id: String,
    is_closed: bool,
    has_checked: bool,
}

impl EventSource {
    pub fn new(response: Response) -> Self {
        Self {
            response: Some(response),
            cur_stream: None,
            last_event_id: String::new(),
            is_closed: false,
            has_checked: false,
        }
    }

    /// Close the EventSource stream and stop trying to reconnect
    pub fn close(&mut self) {
        self.is_closed = true;
    }

    /// Get the current ready state
    pub fn ready_state(&self) -> ReadyState {
        if self.is_closed {
            ReadyState::Closed
        } else {
            ReadyState::Open
        }
    }

    pub fn subscribe(&self) -> Pollable {
        self.cur_stream.as_ref().unwrap().subscribe()
    }

    pub fn poll_next(&mut self) -> Poll<Option<Result<Event, Error>>> {
        if self.is_closed {
            return Poll::Ready(None);
        }

        if !self.has_checked {
            self.clear_fetch();
            match check_response(self.response.take().unwrap()) {
                Ok(res) => {
                    self.handle_response(res);
                    return Poll::Ready(Some(Ok(Event::Open)));
                }
                Err(err) => {
                    self.is_closed = true;
                    return Poll::Ready(Some(Err(err)));
                }
            }
        }

        match self.cur_stream.as_mut().unwrap().poll_next() {
            Poll::Ready(Some(Err(err))) => {
                let err = err.into();
                self.handle_error(&err);
                Poll::Ready(Some(Err(err)))
            }
            Poll::Ready(Some(Ok(event))) => {
                self.handle_event(&event);
                Poll::Ready(Some(Ok(event.into())))
            }
            Poll::Ready(None) => {
                let err = Error::StreamEnded;
                self.handle_error(&err);
                Poll::Ready(Some(Err(err)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

fn check_response(response: Response) -> Result<Response, Error> {
    match response.status() {
        StatusCode::OK => {}
        status => {
            return Err(Error::InvalidStatusCode(status, response));
        }
    }
    let content_type =
        if let Some(content_type) = response.headers().get(&reqwest::header::CONTENT_TYPE) {
            content_type
        } else {
            return Err(Error::InvalidContentType(
                HeaderValue::from_static(""),
                response,
            ));
        };
    if content_type
        .to_str()
        .map_err(|_| ())
        .and_then(|s| s.parse::<mime::Mime>().map_err(|_| ()))
        .map(|mime_type| {
            matches!(
                (mime_type.type_(), mime_type.subtype()),
                (mime::TEXT, mime::EVENT_STREAM)
            )
        })
        .unwrap_or(false)
    {
        Ok(response)
    } else {
        Err(Error::InvalidContentType(content_type.clone(), response))
    }
}

impl EventSource {
    fn clear_fetch(&mut self) {
        self.cur_stream.take();
    }

    fn handle_response(&mut self, res: Response) {
        let handle = unsafe { std::mem::transmute(res.into_raw_input_stream()) };
        let mut stream = EventStream::new(handle);
        stream.set_last_event_id(self.last_event_id.clone());
        self.has_checked = true;
        self.cur_stream.replace(stream);
    }

    fn handle_event(&mut self, event: &MessageEvent) {
        self.last_event_id = event.id.clone();
    }

    fn handle_error(&mut self, _error: &Error) {
        self.clear_fetch();
        self.is_closed = true;
    }
}

/// Events created by the [`EventSource`]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Event {
    /// The event fired when the connection is opened
    Open,
    /// The event fired when a [`MessageEvent`] is received
    Message(MessageEvent),
}

impl From<MessageEvent> for Event {
    fn from(event: MessageEvent) -> Self {
        Event::Message(event)
    }
}
