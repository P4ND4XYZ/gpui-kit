//! Strict, one-shot source-line navigation. Measurement never paints or scrolls.
use super::{
    RenderedText, TextViewState, TextViewStyle,
    document::{NodeRenderOptions, flow_position},
    node::{BlockNode, NodeContext},
    state::TextViewFormat,
    stream_fade::TextLeafKey,
};
use gpui::{
    AnyElement, App, AvailableSpace, Bounds, Context, Element, ElementId, EntityId,
    GlobalElementId, InspectorElementId, IntoElement, LayoutId, ListOffset, ParentElement as _,
    Pixels, Point, ScrollHandle, Task, TextStyle, WeakEntity, Window, div, point, size,
};
#[cfg(not(target_family = "wasm"))]
use std::time::Instant;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fmt,
    future::Future,
    ops::Range,
    pin::Pin,
    rc::{Rc, Weak},
    sync::Arc,
    task::{Context as TaskContext, Poll, Waker},
};
#[cfg(target_family = "wasm")]
use web_time::Instant;

// BEGIN REVEAL_BUDGET
mod reveal_budget {
    use std::time::Duration;
    pub(super) const DEADLINE: Duration = Duration::from_secs(1);
    const LIMIT: u8 = 8;
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum Exhausted {
        Attempts,
        Deadline,
    }
    #[derive(Default)]
    pub(super) struct Budget {
        attempts: u8,
    }
    impl Budget {
        pub(super) fn check_time(elapsed: Duration) -> Result<(), Exhausted> {
            if elapsed >= DEADLINE {
                Err(Exhausted::Deadline)
            } else {
                Ok(())
            }
        }
        pub(super) fn begin(&mut self, elapsed: Duration) -> Result<(), Exhausted> {
            Self::check_time(elapsed)?;
            if self.attempts == LIMIT {
                return Err(Exhausted::Attempts);
            }
            self.attempts += 1;
            Ok(())
        }
        pub(super) fn retry(&self, elapsed: Duration) -> Result<(), Exhausted> {
            Self::check_time(elapsed)?;
            if self.attempts == LIMIT {
                Err(Exhausted::Attempts)
            } else {
                Ok(())
            }
        }
        pub(super) fn commit<R>(
            &self,
            elapsed: Duration,
            apply: impl FnOnce() -> R,
        ) -> Result<R, Exhausted> {
            Self::check_time(elapsed)?;
            Ok(apply())
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn eighth_attempt_can_commit_but_ninth_cannot_start() {
            let mut budget = Budget::default();
            for expected in 1..=8 {
                assert_eq!(budget.begin(Duration::ZERO), Ok(()));
                assert_eq!(budget.attempts, expected);
            }
            let mut writes = 0;
            assert_eq!(budget.commit(Duration::ZERO, || writes += 1), Ok(()));
            assert_eq!(writes, 1);
            assert_eq!(budget.begin(Duration::ZERO), Err(Exhausted::Attempts));
            assert_eq!(budget.attempts, 8);
        }
        #[test]
        fn unresolved_eighth_attempt_is_terminal_including_initial() {
            let mut budget = Budget::default();
            for _ in 0..7 {
                budget.begin(Duration::ZERO).unwrap();
                assert_eq!(budget.retry(Duration::ZERO), Ok(()));
            }
            budget.begin(Duration::ZERO).unwrap();
            assert_eq!(budget.retry(Duration::ZERO), Err(Exhausted::Attempts));
            assert_eq!(budget.attempts, 8);
        }
        #[test]
        fn deadline_wins_even_when_attempt_budget_is_exhausted() {
            let mut budget = Budget::default();
            for _ in 0..8 {
                budget.begin(Duration::ZERO).unwrap();
            }
            assert_eq!(budget.begin(DEADLINE), Err(Exhausted::Deadline));
            assert_eq!(budget.retry(DEADLINE), Err(Exhausted::Deadline));
            let mut initial = Budget::default();
            assert_eq!(initial.begin(DEADLINE), Err(Exhausted::Deadline));
            assert_eq!(initial.attempts, 0);
        }
        #[test]
        fn measurement_and_precommit_deadline_guards_never_write_scroll() {
            let mut budget = Budget::default();
            budget.begin(Duration::ZERO).unwrap();
            assert_eq!(Budget::check_time(DEADLINE), Err(Exhausted::Deadline));
            let mut writes = 0;
            assert_eq!(
                budget.commit(DEADLINE, || writes += 1),
                Err(Exhausted::Deadline)
            );
            assert_eq!(writes, 0);
            assert_eq!(
                budget.commit(DEADLINE + Duration::from_nanos(1), || writes += 1),
                Err(Exhausted::Deadline)
            );
            assert_eq!(writes, 0);
            assert_eq!(
                budget.commit(DEADLINE - Duration::from_nanos(1), || writes += 1),
                Ok(())
            );
            assert_eq!(writes, 1);
        }
    }
}
// END REVEAL_BUDGET
use reveal_budget::{Budget, DEADLINE, Exhausted};
fn budget_failure(reason: Exhausted) -> TextRevealFailure {
    match reason {
        Exhausted::Attempts => TextRevealFailure::AttemptsExhausted,
        Exhausted::Deadline => TextRevealFailure::TimedOut,
    }
}

/// A position in one immutable rendered snapshot, or an explicitly empty code frame.
#[derive(Clone, Debug)]
pub struct TextRevealTarget(Target);
#[derive(Clone, Debug)]
enum Target {
    Rendered(usize),
    Code(Range<usize>, usize),
    Empty(Range<usize>),
    Noop,
}
impl TextRevealTarget {
    /// The UTF-8 byte position of a real text character. Separators and atoms
    /// are rejected rather than moved to nearby text.
    pub fn rendered_position(offset: usize) -> Self {
        Self(Target::Rendered(offset))
    }
    /// An exact source-span match and a UTF-8 byte offset in `CodeBlock::code`.
    /// A newline addresses the row it ends; `len` requires a final empty row.
    pub fn code_position(source_span: Range<usize>, offset: usize) -> Self {
        Self(Target::Code(source_span, offset))
    }
    /// The actual border box of an empty code block, excluding outer margins.
    pub fn empty_code(source_span: Range<usize>) -> Self {
        Self(Target::Empty(source_span))
    }
    /// Succeeds without measurement only when the document has no display target.
    pub fn noop() -> Self {
        Self(Target::Noop)
    }
}
/// Opaque view identity and monotonically increasing request serial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextRevealId {
    owner: EntityId,
    serial: u64,
}
/// A one-way request state. Terminal values never change or revive after reflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextRevealStatus {
    Pending,
    Shown,
    Failed(TextRevealFailure),
    Cancelled(TextRevealCancellation),
}
/// Why a request could not establish full visibility of its exact target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextRevealFailure {
    UnsupportedFormat,
    InvalidTarget,
    StaleSnapshot,
    UnsupportedGeometry,
    HorizontallyClipped,
    TargetTooLarge,
    AttemptsExhausted,
    TimedOut,
    VerificationFailed,
}
/// The operation or layout change that retired a pending request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextRevealCancellation {
    Explicit,
    Superseded,
    ContentChanged,
    ViewDropped,
    UserScroll,
    LayoutChangedAfterScroll,
}
struct ResultRecord {
    status: TextRevealStatus,
    waiters: Vec<Weak<RefCell<Option<Waker>>>>,
    timer: Option<Task<()>>,
}
/// A retained one-shot result. It never strongly owns a view or window.
#[derive(Clone)]
pub struct TextRevealHandle {
    id: TextRevealId,
    record: Rc<RefCell<ResultRecord>>,
}
impl fmt::Debug for TextRevealHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextRevealHandle")
            .field("id", &self.id)
            .field("status", &self.status())
            .finish()
    }
}
impl TextRevealHandle {
    /// The identity shared by all clones of this handle.
    pub fn id(&self) -> TextRevealId {
        self.id
    }
    /// An owned snapshot; even synchronous rejection is retained here.
    pub fn status(&self) -> TextRevealStatus {
        self.record.borrow().status
    }
    /// Returns true only for the first Pending-to-Cancelled transition.
    /// No frame or application context is required.
    pub fn cancel(&self) -> bool {
        self.finish(TextRevealStatus::Cancelled(
            TextRevealCancellation::Explicit,
        ))
    }
    /// Waits for a terminal result. Multiple independent waiters are supported;
    /// dropping a waiter or a handle clone does not cancel the request.
    pub async fn wait(&self) -> TextRevealStatus {
        Wait {
            record: self.record.clone(),
            slot: Rc::new(RefCell::new(None)),
            registered: false,
        }
        .await
    }
    fn finish(&self, status: TextRevealStatus) -> bool {
        finish(&self.record, status)
    }
}
fn finish(record: &Rc<RefCell<ResultRecord>>, status: TextRevealStatus) -> bool {
    let (waiters, timer) = {
        let mut result = record.borrow_mut();
        if result.status != TextRevealStatus::Pending {
            return false;
        }
        result.status = status;
        (std::mem::take(&mut result.waiters), result.timer.take())
    };
    drop(timer);
    for waiter in waiters {
        if let Some(slot) = waiter.upgrade() {
            if let Some(waker) = slot.borrow_mut().take() {
                waker.wake();
            }
        }
    }
    true
}
struct Wait {
    record: Rc<RefCell<ResultRecord>>,
    slot: Rc<RefCell<Option<Waker>>>,
    registered: bool,
}
impl Future for Wait {
    type Output = TextRevealStatus;
    fn poll(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        let status = self.record.borrow().status;
        if status != TextRevealStatus::Pending {
            return Poll::Ready(status);
        }
        *self.slot.borrow_mut() = Some(cx.waker().clone());
        if !self.registered {
            {
                let mut record = self.record.borrow_mut();
                record.waiters.retain(|slot| slot.strong_count() > 0);
                record.waiters.push(Rc::downgrade(&self.slot));
            }
            self.registered = true;
        }
        Poll::Pending
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Resolved {
    pub block_ix: usize,
    pub key: TextLeafKey,
    pub offset: usize,
    pub empty: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ScrollKey {
    pub source: usize,
    pub code: bool,
}
#[derive(Clone)]
pub(super) struct StrictRender {
    pub owner: WeakEntity<TextViewState>,
    pub target: Option<Resolved>,
    pub offsets: Option<Arc<BTreeMap<ScrollKey, Point<Pixels>>>>,
}
impl StrictRender {
    pub fn offset(&self, key: Option<TextLeafKey>) -> Option<usize> {
        let target = self.target?;
        (key == Some(target.key) && !target.empty).then_some(target.offset)
    }
    pub fn empty(&self, key: Option<TextLeafKey>) -> bool {
        self.target.is_some_and(|t| t.empty && key == Some(t.key))
    }
    pub fn handle(&self, key: ScrollKey, live: Option<ScrollHandle>, cx: &App) -> ScrollHandle {
        if let Some(offsets) = &self.offsets {
            let handle = ScrollHandle::default();
            handle.set_offset(offsets.get(&key).copied().unwrap_or_default());
            return handle;
        }
        let handle = live.expect("live scroll handle");
        // A list can construct the same item in a height-only namespace before
        // its visible subtree. All those renders must share this view-owned
        // handle; otherwise a plan could update only the measurement instance.
        self.owner
            .read_with(cx, |state, _| {
                state
                    .line_reveal
                    .inner
                    .borrow_mut()
                    .entry(key)
                    .or_insert(handle.clone())
                    .clone()
            })
            .unwrap_or(handle)
    }
}
#[derive(Clone, Debug)]
struct InnerGeometry {
    key: ScrollKey,
    bounds: Bounds<Pixels>,
    frame_bounds: Bounds<Pixels>,
    offset: Point<Pixels>,
    max: Point<Pixels>,
}
#[derive(Clone)]
pub(super) struct Observation {
    rect: Bounds<Pixels>,
    mask: Bounds<Pixels>,
    inner: Vec<InnerGeometry>,
}
#[derive(Default)]
struct Collector {
    observation: Option<Observation>,
    serial: u64,
}
thread_local! {
    static COLLECTORS: RefCell<Vec<Collector>> = const { RefCell::new(Vec::new()) };
    static MEASURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(super) fn is_measuring() -> bool {
    MEASURING.with(|flag| flag.get())
}
fn measuring<R>(f: impl FnOnce() -> R) -> R {
    let previous = MEASURING.with(|flag| flag.replace(true));
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            MEASURING.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(previous);
    f()
}
fn collect<R>(f: impl FnOnce() -> R) -> (R, Option<Observation>) {
    COLLECTORS.with(|stack| stack.borrow_mut().push(Collector::default()));
    struct Pop;
    impl Drop for Pop {
        fn drop(&mut self) {
            COLLECTORS.with(|stack| {
                stack.borrow_mut().pop();
            });
        }
    }
    let guard = Pop;
    let result = f();
    let observation =
        COLLECTORS.with(|stack| stack.borrow_mut().last_mut().unwrap().observation.take());
    drop(guard);
    (result, observation)
}
pub(super) fn observe(rect: Bounds<Pixels>, window: &Window) {
    COLLECTORS.with(|stack| {
        if let Some(sink) = stack.borrow_mut().last_mut() {
            sink.serial += 1;
            sink.observation = Some(Observation {
                rect,
                mask: window.content_mask().bounds,
                inner: Vec::new(),
            });
        }
    });
}
fn serial() -> u64 {
    COLLECTORS.with(|stack| stack.borrow().last().map_or(0, |sink| sink.serial))
}

/// Observes the renderer's actual border box, before any outer paragraph gap.
pub(super) struct Frame {
    child: AnyElement,
    inner: Option<(ScrollKey, ScrollHandle, gpui::Style)>,
    empty: bool,
}
impl Frame {
    pub fn new(
        child: AnyElement,
        inner: Option<(ScrollKey, ScrollHandle, gpui::Style)>,
        empty: bool,
    ) -> AnyElement {
        Self {
            child,
            inner,
            empty,
        }
        .into_any_element()
    }
}
impl IntoElement for Frame {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}
impl Element for Frame {
    type RequestLayoutState = ();
    type PrepaintState = ();
    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }
    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.request_layout(window, cx), ())
    }
    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        let before = serial();
        self.child.prepaint(window, cx);
        if self.empty {
            observe(bounds, window);
        }
        if !self.empty && before != serial() {
            if let Some((key, handle, style)) = &self.inner {
                COLLECTORS.with(|stack| {
                    if let Some(observation) = stack
                        .borrow_mut()
                        .last_mut()
                        .and_then(|sink| sink.observation.as_mut())
                    {
                        observation.inner.push(InnerGeometry {
                            key: *key,
                            bounds: style
                                .overflow_mask(handle.bounds(), window.rem_size())
                                .map_or(handle.bounds(), |mask| mask.bounds),
                            frame_bounds: handle.bounds(),
                            offset: handle.offset(),
                            max: handle.max_offset(),
                        });
                    }
                });
            }
        }
    }
    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.paint(window, cx);
    }
}

struct Pending {
    handle: TextRevealHandle,
    target: Resolved,
    started: Instant,
    budget: Budget,
    committed: Option<Epoch>,
    requested_outer: Point<Pixels>,
    requested_inner: BTreeMap<ScrollKey, Point<Pixels>>,
}
#[derive(Clone)]
struct Epoch {
    bounds: Bounds<Pixels>,
    mask: Bounds<Pixels>,
    style: Arc<TextViewStyle>,
    typography: Option<(TextStyle, Pixels)>,
    revision: (usize, u64),
    outer: Point<Pixels>,
    inner: Vec<InnerGeometry>,
}
#[derive(Default)]
struct ScrollbarGesture {
    active: std::cell::Cell<bool>,
    pending: RefCell<Weak<RefCell<ResultRecord>>>,
}
#[derive(Clone)]
pub(super) struct RevealScrollbarHandle {
    list: gpui::ListState,
    gesture: Rc<ScrollbarGesture>,
}
impl crate::ScrollbarHandle for RevealScrollbarHandle {
    fn viewport_bounds(&self) -> Bounds<Pixels> {
        crate::ScrollbarHandle::viewport_bounds(&self.list)
    }
    fn offset(&self) -> Point<Pixels> {
        crate::ScrollbarHandle::offset(&self.list)
    }
    fn set_offset(&self, offset: Point<Pixels>) {
        crate::ScrollbarHandle::set_offset(&self.list, offset);
    }
    fn content_size(&self) -> gpui::Size<Pixels> {
        crate::ScrollbarHandle::content_size(&self.list)
    }
    fn start_drag(&self) {
        self.gesture.active.set(true);
        if let Some(record) = self.gesture.pending.borrow().upgrade() {
            finish(
                &record,
                TextRevealStatus::Cancelled(TextRevealCancellation::UserScroll),
            );
        }
        crate::ScrollbarHandle::start_drag(&self.list);
    }
    fn end_drag(&self) {
        self.gesture.active.set(false);
        crate::ScrollbarHandle::end_drag(&self.list);
    }
}

#[derive(Default)]
pub(super) struct LineRevealState {
    serial: u64,
    pending: Option<Pending>,
    pub inner: RefCell<BTreeMap<ScrollKey, ScrollHandle>>,
    ready: Option<(TextRevealId, u64, Observation)>,
    cookie: u64,
    layout: Option<Epoch>,
    parent_mask: Bounds<Pixels>,
    layout_revision: std::cell::Cell<u64>,
    gesture: Rc<ScrollbarGesture>,
}
impl Drop for LineRevealState {
    fn drop(&mut self) {
        self.cancel(TextRevealCancellation::ViewDropped);
    }
}
impl LineRevealState {
    pub fn scrollbar_handle(&self, list: gpui::ListState) -> RevealScrollbarHandle {
        RevealScrollbarHandle {
            list,
            gesture: self.gesture.clone(),
        }
    }
    pub fn invalidate_layout(&self) {
        self.layout_revision.set(
            self.layout_revision
                .get()
                .checked_add(1)
                .expect("text reveal layout revision exhausted"),
        );
    }
    pub fn cancel(&mut self, reason: TextRevealCancellation) {
        if let Some(pending) = self.pending.take() {
            pending.handle.finish(TextRevealStatus::Cancelled(reason));
        }
        self.ready = None;
    }
    pub fn render(&self, owner: WeakEntity<TextViewState>) -> StrictRender {
        StrictRender {
            owner,
            target: self
                .pending
                .as_ref()
                .filter(|p| p.handle.status() == TextRevealStatus::Pending)
                .map(|p| p.target),
            offsets: None,
        }
    }
    fn fail(&mut self, reason: TextRevealFailure) {
        if let Some(pending) = self.pending.take() {
            pending.handle.finish(TextRevealStatus::Failed(reason));
        }
        self.ready = None;
    }
}
fn find_code(
    block: &BlockNode,
    span: &Range<usize>,
    top: usize,
    matches: &mut Vec<(Resolved, String)>,
) {
    match block {
        BlockNode::Root { children, .. }
        | BlockNode::Blockquote { children, .. }
        | BlockNode::List { children, .. }
        | BlockNode::ListItem { children, .. } => {
            for child in children {
                find_code(child, span, top, matches);
            }
        }
        BlockNode::CodeBlock(code)
            if code
                .span
                .is_some_and(|s| s.start == span.start && s.end == span.end) =>
        {
            matches.push((
                Resolved {
                    block_ix: top,
                    key: TextLeafKey::block(span.start),
                    offset: 0,
                    empty: false,
                },
                code.code().to_string(),
            ));
        }
        _ => {}
    }
}
fn invisible(block: &BlockNode) -> bool {
    match block {
        BlockNode::Root { children, .. }
        | BlockNode::Blockquote { children, .. }
        | BlockNode::List { children, .. }
        | BlockNode::ListItem { children, .. } => children.iter().all(invisible),
        BlockNode::Paragraph(p) => p.children().iter().all(|c| {
            c.text().trim().is_empty() && c.as_image().is_none() && c.as_custom().is_none()
        }),
        BlockNode::Break { .. } | BlockNode::Definition { .. } | BlockNode::Unknown => true,
        _ => false,
    }
}
pub(super) fn request(
    state: &mut TextViewState,
    snapshot: &RenderedText,
    target: TextRevealTarget,
    cx: &mut Context<TextViewState>,
) -> TextRevealHandle {
    state.line_reveal.cancel(TextRevealCancellation::Superseded);
    state.pending_reveal = None;
    state.line_reveal.serial = state
        .line_reveal
        .serial
        .checked_add(1)
        .expect("text reveal request serial exhausted");
    let handle = TextRevealHandle {
        id: TextRevealId {
            owner: cx.entity_id(),
            serial: state.line_reveal.serial,
        },
        record: Rc::new(RefCell::new(ResultRecord {
            status: TextRevealStatus::Pending,
            waiters: Vec::new(),
            timer: None,
        })),
    };
    *state.line_reveal.gesture.pending.borrow_mut() = Rc::downgrade(&handle.record);
    if state.line_reveal.gesture.active.get() {
        handle.finish(TextRevealStatus::Cancelled(
            TextRevealCancellation::UserScroll,
        ));
        return handle;
    }
    let (input, empty) = match target.0 {
        Target::Empty(span) => (Target::Code(span, 0), true),
        other => (other, false),
    };
    let resolved = (|| {
        if state.format != TextViewFormat::Markdown {
            return Err(TextRevealFailure::UnsupportedFormat);
        }
        if *snapshot != state.rendered_text() || state.revision != state.committed_revision {
            return Err(TextRevealFailure::StaleSnapshot);
        }
        if !state.scrollable || state.max_lines.is_some() {
            return Err(TextRevealFailure::UnsupportedGeometry);
        }
        match input {
            Target::Noop => {
                if state.parsed_content.document.blocks.iter().all(invisible) {
                    Ok(None)
                } else {
                    Err(TextRevealFailure::InvalidTarget)
                }
            }
            Target::Rendered(offset) => snapshot
                .index()
                .strict_position(offset)
                .map(|(block_ix, key, offset)| {
                    Some(Resolved {
                        block_ix,
                        key,
                        offset,
                        empty: false,
                    })
                })
                .ok_or(TextRevealFailure::InvalidTarget),
            Target::Empty(_) => unreachable!(),
            Target::Code(span, offset) => {
                let mut matches = Vec::new();
                for (ix, block) in state.parsed_content.document.blocks.iter().enumerate() {
                    find_code(block, &span, ix, &mut matches);
                }
                if matches.len() != 1 {
                    return Err(TextRevealFailure::InvalidTarget);
                }
                let (mut resolved, code) = matches.pop().unwrap();
                if empty {
                    if !code.is_empty() {
                        return Err(TextRevealFailure::InvalidTarget);
                    }
                } else if offset > code.len()
                    || !code.is_char_boundary(offset)
                    || (offset == code.len() && !code.ends_with('\n'))
                {
                    return Err(TextRevealFailure::InvalidTarget);
                }
                resolved.offset = if empty { 0 } else { offset };
                resolved.empty = empty;
                Ok(Some(resolved))
            }
        }
    })();
    let resolved = match resolved {
        Err(reason) => {
            handle.finish(TextRevealStatus::Failed(reason));
            return handle;
        }
        Ok(None) => {
            handle.finish(TextRevealStatus::Shown);
            return handle;
        }
        Ok(Some(target)) => target,
    };
    let weak = Rc::downgrade(&handle.record);
    let timer = cx.spawn(async move |_, cx| {
        cx.background_executor().timer(DEADLINE).await;
        if let Some(record) = weak.upgrade() {
            finish(
                &record,
                TextRevealStatus::Failed(TextRevealFailure::TimedOut),
            );
        }
    });
    handle.record.borrow_mut().timer = Some(timer);
    state.line_reveal.pending = Some(Pending {
        handle: handle.clone(),
        target: resolved,
        started: cx.background_executor().now(),
        budget: Budget::default(),
        committed: None,
        requested_outer: state.list_state.scroll_px_offset_for_scrollbar(),
        requested_inner: state
            .line_reveal
            .inner
            .borrow()
            .iter()
            .map(|(key, handle)| (*key, handle.offset()))
            .collect(),
    });
    cx.notify();
    handle
}

fn correction(rect: Bounds<Pixels>, viewport: Bounds<Pixels>) -> Pixels {
    if rect.top() < viewport.top() {
        rect.top() - viewport.top()
    } else if rect.bottom() > viewport.bottom() {
        rect.bottom() - viewport.bottom()
    } else {
        Pixels::ZERO
    }
}
fn contains(rect: Bounds<Pixels>, viewport: Bounds<Pixels>) -> bool {
    rect.left() >= viewport.left()
        && rect.right() <= viewport.right()
        && rect.top() >= viewport.top()
        && rect.bottom() <= viewport.bottom()
}
fn offset_rect(rect: Bounds<Pixels>, delta: Point<Pixels>) -> Bounds<Pixels> {
    Bounds::new(rect.origin + delta, rect.size)
}

/// Runs before the real list prepaint, so all inner and outer offsets are installed
/// together and the list's own normalization happens in this same frame.
pub(super) struct RevealElement {
    child: AnyElement,
    owner: WeakEntity<TextViewState>,
}
impl RevealElement {
    pub fn new(child: AnyElement, owner: WeakEntity<TextViewState>) -> Self {
        Self { child, owner }
    }
}
impl IntoElement for RevealElement {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}
impl Element for RevealElement {
    type RequestLayoutState = ();
    type PrepaintState = ();
    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }
    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.request_layout(window, cx), ())
    }
    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        let _ = self.owner.update(cx, |state, cx| {
            state.line_reveal.parent_mask = window.content_mask().bounds;
            prepare(state, bounds, window, cx);
        });
        let (_, observation) = collect(|| self.child.prepaint(window, cx));
        let _ = self.owner.update(cx, |state, _| {
            let stamp = Epoch {
                bounds,
                mask: state.line_reveal.parent_mask,
                style: state.text_view_style.clone(),
                typography: state.layout_text_style.clone(),
                revision: (
                    state.selection_revision,
                    state.line_reveal.layout_revision.get(),
                ),
                outer: state.list_state.scroll_px_offset_for_scrollbar(),
                inner: Vec::new(),
            };
            let layout_changed = state
                .line_reveal
                .layout
                .as_ref()
                .is_none_or(|previous| !epoch_matches(state, previous, bounds));
            if layout_changed {
                if let Some(pending) = &mut state.line_reveal.pending {
                    if pending.committed.is_none() {
                        pending.requested_outer = stamp.outer;
                        pending.requested_inner = state
                            .line_reveal
                            .inner
                            .borrow()
                            .iter()
                            .map(|(key, handle)| (*key, handle.offset()))
                            .collect();
                    }
                }
            }
            state.line_reveal.layout = Some(stamp);
            state.line_reveal.cookie += 1;
            let runtime = &mut state.line_reveal;
            runtime.ready = None;
            if let Some(pending) = &mut runtime.pending {
                if let Some(epoch) = &mut pending.committed {
                    epoch.outer = state.list_state.scroll_px_offset_for_scrollbar();
                    if let Some(observation) = observation {
                        epoch.inner = observation.inner.clone();
                        runtime.ready = Some((pending.handle.id, runtime.cookie, observation));
                    }
                }
            }
        });
    }
    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.paint(window, cx);
    }
}
fn measure(
    state: &TextViewState,
    target: Resolved,
    bounds: Bounds<Pixels>,
    window: &mut Window,
    cx: &mut App,
) -> Result<Observation, TextRevealFailure> {
    let block = state.parsed_content.document.blocks[target.block_ix]
        .measurement_clone()
        .ok_or(TextRevealFailure::UnsupportedGeometry)?;
    let offsets = state
        .line_reveal
        .inner
        .borrow()
        .iter()
        .map(|(key, handle)| (*key, handle.offset()))
        .collect();
    let mut node = NodeContext {
        style: state.text_view_style.clone(),
        link_refs: state.parsed_content.node_cx.link_refs.clone(),
        offset: state.parsed_content.node_cx.offset,
        code_block_highlighter: state.code_block_highlighter.clone(),
        code_block_actions: state.code_block_actions.clone(),
        table_actions: state.table_actions.clone(),
        markdown_extensions: state.markdown_extensions.clone(),
        body_font_size: window.text_style().font_size.to_pixels(window.rem_size()),
        ..Default::default()
    };
    node.strict = Some(StrictRender {
        owner: state.line_reveal_owner.clone().expect("strict owner"),
        target: Some(target),
        offsets: Some(Arc::new(offsets)),
    });
    let (is_first, prev) = flow_position(&state.parsed_content.document.blocks, target.block_ix);
    let previous = window.take_autoscroll();
    let result = measuring(|| {
        window.with_element_namespace("strict-line-measurement", |window| {
            let mut element = div()
                .child(block.render_block(
                    NodeRenderOptions {
                        ix: target.block_ix,
                        is_first,
                        prev,
                        is_last: target.block_ix + 1 == state.parsed_content.document.blocks.len(),
                        ..Default::default()
                    },
                    &node,
                    window,
                    cx,
                ))
                .into_any_element();
            element.layout_as_root(
                size(
                    AvailableSpace::Definite(bounds.size.width),
                    AvailableSpace::MinContent,
                ),
                window,
                cx,
            );
            let result: Result<(), _> = window.transact(|window| {
                let (_, observation) = collect(|| element.prepaint_at(bounds.origin, window, cx));
                Err(observation)
            });
            result.unwrap_err()
        })
    });
    let unexpected = window.take_autoscroll();
    if let Some(previous) = previous {
        window.request_autoscroll(previous);
    }
    if unexpected.is_some() {
        return Err(TextRevealFailure::UnsupportedGeometry);
    }
    result.ok_or(TextRevealFailure::UnsupportedGeometry)
}
fn prepare(
    state: &mut TextViewState,
    bounds: Bounds<Pixels>,
    window: &mut Window,
    cx: &mut Context<TextViewState>,
) {
    let Some(pending) = state.line_reveal.pending.as_ref() else {
        return;
    };
    if pending.handle.status() != TextRevealStatus::Pending {
        state.line_reveal.pending = None;
        return;
    }
    if cx
        .background_executor()
        .now()
        .duration_since(pending.started)
        >= DEADLINE
    {
        state.line_reveal.fail(TextRevealFailure::TimedOut);
        return;
    }
    if let Some(epoch) = &pending.committed {
        if !epoch_matches(state, epoch, bounds) {
            state
                .line_reveal
                .cancel(TextRevealCancellation::LayoutChangedAfterScroll);
        } else if state.list_state.scroll_px_offset_for_scrollbar() != epoch.outer
            || epoch.inner.iter().any(|inner| {
                state
                    .line_reveal
                    .inner
                    .borrow()
                    .get(&inner.key)
                    .is_none_or(|handle| handle.offset() != inner.offset)
            })
        {
            state.line_reveal.cancel(TextRevealCancellation::UserScroll);
        }
        return;
    }
    // First layout supplies exact row extents. No preliminary navigation is used.
    if state.list_state.viewport_bounds() != bounds
        || bounds.size.width <= Pixels::ZERO
        || bounds.size.height <= Pixels::ZERO
    {
        window.request_animation_frame();
        return;
    }
    if state
        .line_reveal
        .layout
        .as_ref()
        .is_none_or(|previous| !epoch_matches(state, previous, bounds))
    {
        window.request_animation_frame();
        return;
    }
    if state.list_state.scroll_px_offset_for_scrollbar() != pending.requested_outer
        || pending.requested_inner.iter().any(|(key, offset)| {
            state
                .line_reveal
                .inner
                .borrow()
                .get(key)
                .is_some_and(|handle| handle.offset() != *offset)
        })
    {
        state.line_reveal.cancel(TextRevealCancellation::UserScroll);
        return;
    }
    let target = pending.target;
    let elapsed = cx
        .background_executor()
        .now()
        .duration_since(pending.started);
    if let Err(reason) = state
        .line_reveal
        .pending
        .as_mut()
        .unwrap()
        .budget
        .begin(elapsed)
    {
        state.line_reveal.fail(budget_failure(reason));
        return;
    }
    let observation = match measure(state, target, bounds, window, cx) {
        Ok(o) => o,
        Err(reason) => {
            state.line_reveal.fail(reason);
            return;
        }
    };
    let pending = state.line_reveal.pending.as_ref().unwrap();
    if let Err(reason) = Budget::check_time(
        cx.background_executor()
            .now()
            .duration_since(pending.started),
    ) {
        state.line_reveal.fail(budget_failure(reason));
        return;
    }
    let viewport = bounds.intersect(&state.line_reveal.parent_mask);
    if observation.rect.size.height > viewport.size.height
        || observation.rect.size.width > viewport.size.width
    {
        state.line_reveal.fail(TextRevealFailure::TargetTooLarge);
        return;
    }
    let mut rect = observation.rect;
    let mut changes = Vec::new();
    for inner in &observation.inner {
        if rect.size.height > inner.bounds.size.height || rect.size.width > inner.bounds.size.width
        {
            state.line_reveal.fail(TextRevealFailure::TargetTooLarge);
            return;
        }
        if rect.left() < inner.bounds.left() || rect.right() > inner.bounds.right() {
            state
                .line_reveal
                .fail(TextRevealFailure::HorizontallyClipped);
            return;
        }
        let handle = state.line_reveal.inner.borrow().get(&inner.key).cloned();
        let Some(handle) = handle else {
            if let Err(reason) = pending.budget.retry(
                cx.background_executor()
                    .now()
                    .duration_since(pending.started),
            ) {
                state.line_reveal.fail(budget_failure(reason));
            } else {
                window.request_animation_frame();
            }
            return;
        };
        if handle.offset() != inner.offset {
            state.line_reveal.cancel(TextRevealCancellation::UserScroll);
            return;
        }
        let delta = correction(rect, inner.bounds);
        let offset = point(inner.offset.x, inner.offset.y - delta);
        if -offset.y > inner.max.y || offset.y > Pixels::ZERO {
            state
                .line_reveal
                .fail(TextRevealFailure::UnsupportedGeometry);
            return;
        }
        changes.push((handle, offset));
        rect = offset_rect(rect, point(Pixels::ZERO, -delta));
    }
    if rect.left() < viewport.left() || rect.right() > viewport.right() {
        state
            .line_reveal
            .fail(TextRevealFailure::HorizontallyClipped);
        return;
    }
    let item = state.list_state.bounds_for_item(target.block_ix);
    let outer = if let Some(item) = item {
        let positioned = offset_rect(rect, point(Pixels::ZERO, item.top() - bounds.top()));
        Some(correction(positioned, viewport))
    } else if target.block_ix < state.list_state.logical_scroll_top().item_ix {
        None
    } else {
        if let Err(reason) = pending.budget.retry(
            cx.background_executor()
                .now()
                .duration_since(pending.started),
        ) {
            state.line_reveal.fail(budget_failure(reason));
        } else {
            window.request_animation_frame();
        }
        return;
    };
    // The tested production guard owns the entire scroll-write boundary.
    let result = pending.budget.commit(
        cx.background_executor()
            .now()
            .duration_since(pending.started),
        || {
            for (handle, offset) in changes {
                handle.set_offset(offset);
            }
            match outer {
                Some(delta) => {
                    if delta != Pixels::ZERO {
                        state.list_state.scroll_by(delta);
                    }
                }
                None => state.list_state.scroll_to(ListOffset {
                    item_ix: target.block_ix,
                    offset_in_item: rect.top() - bounds.top() - (viewport.top() - bounds.top()),
                }),
            }
        },
    );
    if let Err(reason) = result {
        state.line_reveal.fail(budget_failure(reason));
        return;
    }
    let epoch = Epoch {
        bounds,
        mask: state.line_reveal.parent_mask,
        style: state.text_view_style.clone(),
        typography: state.layout_text_style.clone(),
        revision: (
            state.selection_revision,
            state.line_reveal.layout_revision.get(),
        ),
        outer: state.list_state.scroll_px_offset_for_scrollbar(),
        inner: Vec::new(),
    };
    state.line_reveal.pending.as_mut().unwrap().committed = Some(epoch);
}
fn epoch_matches(state: &TextViewState, epoch: &Epoch, bounds: Bounds<Pixels>) -> bool {
    epoch.bounds == bounds
        && epoch.mask == state.line_reveal.parent_mask
        && *epoch.style == *state.text_view_style
        && epoch.typography == state.layout_text_style
        && epoch.revision
            == (
                state.selection_revision,
                state.line_reveal.layout_revision.get(),
            )
}
/// Called only by the actual TextView paint, never the discarded measurement.
pub(super) fn painted(state: &mut TextViewState, window: &mut Window) {
    let Some((id, cookie, observation)) = state.line_reveal.ready.take() else {
        return;
    };
    let Some(pending) = state.line_reveal.pending.as_ref() else {
        return;
    };
    if pending.handle.id != id || pending.handle.status() != TextRevealStatus::Pending {
        return;
    }
    let Some(epoch) = pending.committed.clone() else {
        return;
    };
    let owner = state.line_reveal_owner.clone().unwrap();
    window.on_next_frame(move |_, cx| {
        let _ = owner.update(cx, |state, cx| {
            let Some(pending) = state.line_reveal.pending.as_ref() else {
                return;
            };
            if pending.handle.id != id
                || state.line_reveal.cookie != cookie
                || pending.handle.status() != TextRevealStatus::Pending
            {
                return;
            }
            if cx
                .background_executor()
                .now()
                .duration_since(pending.started)
                >= DEADLINE
            {
                state.line_reveal.fail(TextRevealFailure::TimedOut);
                return;
            }
            if !epoch_matches(state, &epoch, state.list_state.viewport_bounds()) {
                state
                    .line_reveal
                    .cancel(TextRevealCancellation::LayoutChangedAfterScroll);
                return;
            }
            if state.list_state.scroll_px_offset_for_scrollbar() != epoch.outer
                || epoch.inner.iter().any(|inner| {
                    state
                        .line_reveal
                        .inner
                        .borrow()
                        .get(&inner.key)
                        .is_none_or(|handle| {
                            handle.offset() != inner.offset
                                || handle.bounds() != inner.frame_bounds
                                || handle.max_offset() != inner.max
                        })
                })
            {
                state.line_reveal.cancel(TextRevealCancellation::UserScroll);
                return;
            }
            let visible = contains(observation.rect, observation.mask)
                && contains(observation.rect, epoch.bounds.intersect(&epoch.mask))
                && observation
                    .inner
                    .iter()
                    .all(|inner| contains(observation.rect, inner.bounds));
            if visible {
                pending.handle.finish(TextRevealStatus::Shown);
                state.line_reveal.pending = None;
            } else {
                state
                    .line_reveal
                    .fail(TextRevealFailure::VerificationFailed);
            }
        });
    });
}
