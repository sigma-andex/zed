use super::{
    ContainerStyle, Drawable, Element, Flex, KeystrokeLabel, MouseEventHandler, Overlay,
    OverlayFitMode, ParentElement, Text,
};
use crate::{
    fonts::TextStyle,
    geometry::{rect::RectF, vector::Vector2F},
    json::json,
    Action, Axis, ElementStateHandle, SceneBuilder, SizeConstraint, Task, View, ViewContext,
};
use serde::Deserialize;
use std::{
    cell::{Cell, RefCell},
    ops::Range,
    rc::Rc,
    time::Duration,
};
use util::ResultExt;

const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(500);

pub struct Tooltip<V: View> {
    child: Element<V>,
    tooltip: Option<Element<V>>,
    _state: ElementStateHandle<Rc<TooltipState>>,
}

#[derive(Default)]
struct TooltipState {
    visible: Cell<bool>,
    position: Cell<Vector2F>,
    debounce: RefCell<Option<Task<()>>>,
}

#[derive(Clone, Deserialize, Default)]
pub struct TooltipStyle {
    #[serde(flatten)]
    pub container: ContainerStyle,
    pub text: TextStyle,
    keystroke: KeystrokeStyle,
    pub max_text_width: Option<f32>,
}

#[derive(Clone, Deserialize, Default)]
pub struct KeystrokeStyle {
    #[serde(flatten)]
    container: ContainerStyle,
    #[serde(flatten)]
    text: TextStyle,
}

impl<V: View> Tooltip<V> {
    pub fn new<Tag: 'static, T: View>(
        id: usize,
        text: String,
        action: Option<Box<dyn Action>>,
        style: TooltipStyle,
        child: Element<V>,
        cx: &mut ViewContext<V>,
    ) -> Self {
        struct ElementState<Tag>(Tag);
        struct MouseEventHandlerState<Tag>(Tag);
        let focused_view_id = cx.focused_view_id();

        let state_handle = cx.default_element_state::<ElementState<Tag>, Rc<TooltipState>>(id);
        let state = state_handle.read(cx).clone();
        let tooltip = if state.visible.get() {
            let mut collapsed_tooltip = Self::render_tooltip(
                focused_view_id,
                text.clone(),
                style.clone(),
                action.as_ref().map(|a| a.boxed_clone()),
                true,
            )
            .boxed();
            Some(
                Overlay::new(
                    Self::render_tooltip(focused_view_id, text, style, action, false)
                        .constrained()
                        .dynamically(move |constraint, view, cx| {
                            SizeConstraint::strict_along(
                                Axis::Vertical,
                                collapsed_tooltip.layout(constraint, view, cx).y(),
                            )
                        })
                        .boxed(),
                )
                .with_fit_mode(OverlayFitMode::SwitchAnchor)
                .with_anchor_position(state.position.get())
                .boxed(),
            )
        } else {
            None
        };
        let child = MouseEventHandler::<MouseEventHandlerState<Tag>, _>::new(id, cx, |_, _| child)
            .on_hover(move |e, _, cx| {
                let position = e.position;
                if e.started {
                    if !state.visible.get() {
                        state.position.set(position);

                        let mut debounce = state.debounce.borrow_mut();
                        if debounce.is_none() {
                            *debounce = Some(cx.spawn_weak({
                                let state = state.clone();
                                |view, mut cx| async move {
                                    cx.background().timer(DEBOUNCE_TIMEOUT).await;
                                    state.visible.set(true);
                                    if let Some(view) = view.upgrade(&cx) {
                                        view.update(&mut cx, |_, cx| cx.notify()).log_err();
                                    }
                                }
                            }));
                        }
                    }
                } else {
                    state.visible.set(false);
                    state.debounce.take();
                    cx.notify();
                }
            })
            .boxed();
        Self {
            child,
            tooltip,
            _state: state_handle,
        }
    }

    pub fn render_tooltip(
        focused_view_id: Option<usize>,
        text: String,
        style: TooltipStyle,
        action: Option<Box<dyn Action>>,
        measure: bool,
    ) -> impl Drawable<V> {
        Flex::row()
            .with_child({
                let text = if let Some(max_text_width) = style.max_text_width {
                    Text::new(text, style.text)
                        .constrained()
                        .with_max_width(max_text_width)
                } else {
                    Text::new(text, style.text).constrained()
                };

                if measure {
                    text.flex(1., false).boxed()
                } else {
                    text.flex(1., false).aligned().boxed()
                }
            })
            .with_children(action.and_then(|action| {
                let keystroke_label = KeystrokeLabel::new(
                    focused_view_id?,
                    action,
                    style.keystroke.container,
                    style.keystroke.text,
                );
                if measure {
                    Some(keystroke_label.boxed())
                } else {
                    Some(keystroke_label.aligned().boxed())
                }
            }))
            .contained()
            .with_style(style.container)
    }
}

impl<V: View> Drawable<V> for Tooltip<V> {
    type LayoutState = ();
    type PaintState = ();

    fn layout(
        &mut self,
        constraint: SizeConstraint,
        view: &mut V,
        cx: &mut ViewContext<V>,
    ) -> (Vector2F, Self::LayoutState) {
        let size = self.child.layout(constraint, view, cx);
        if let Some(tooltip) = self.tooltip.as_mut() {
            tooltip.layout(
                SizeConstraint::new(Vector2F::zero(), cx.window_size()),
                view,
                cx,
            );
        }
        (size, ())
    }

    fn paint(
        &mut self,
        scene: &mut SceneBuilder,
        bounds: RectF,
        visible_bounds: RectF,
        _: &mut Self::LayoutState,
        view: &mut V,
        cx: &mut ViewContext<V>,
    ) {
        self.child
            .paint(scene, bounds.origin(), visible_bounds, view, cx);
        if let Some(tooltip) = self.tooltip.as_mut() {
            tooltip.paint(scene, bounds.origin(), visible_bounds, view, cx);
        }
    }

    fn rect_for_text_range(
        &self,
        range: Range<usize>,
        _: RectF,
        _: RectF,
        _: &Self::LayoutState,
        _: &Self::PaintState,
        view: &V,
        cx: &ViewContext<V>,
    ) -> Option<RectF> {
        self.child.rect_for_text_range(range, view, cx)
    }

    fn debug(
        &self,
        _: RectF,
        _: &Self::LayoutState,
        _: &Self::PaintState,
        view: &V,
        cx: &ViewContext<V>,
    ) -> serde_json::Value {
        json!({
            "child": self.child.debug(view, cx),
            "tooltip": self.tooltip.as_ref().map(|t| t.debug(view, cx)),
        })
    }
}
