---
layout: post
title: "FRP on One Click Counter: Missed First Event, and What map Does"
description: "Blank counter UI: listeners only see future clicks, so you must apply the current 0 at setup. hold does the same job as a Behavior sample. Same counter, then map."
keywords: "FRP, functional reactive programming, Event, Behavior, hold, map, snapshot, missed first event, Reactive Banana, click counter"
date: 2026-11-16
comments: true
tags:
  - FRP
  - automata
  - event-driven
  - Haskell
glossary:
  - FRP
  - Haskell
  - Transducer
  - React
---

[Part VI of the automata series](/2026/11/12/automata-basics-for-coders-part-6/) put [FRP](/cs-glossary/frp/) next to Observer and event logs. That section is short on purpose: Event vs Behavior, a click counter sketch, then Rx and React. If you only know `addEventListener`, the sketch is thin. This note slows down on **one** program and traces it twice: once with ordinary DOM callbacks, once with classic FRP combinators (`map`, `snapshot`, `hold`).

The program:

> One button. Each click adds 1. The number is shown on screen. Initial value is `0`.

No new framework. The FRP side is [Haskell](/cs-glossary/haskell/) pseudocode in the Fran / Reactive Banana / Sodium model (same as Part VI).

## Event and Behavior

Classic FRP splits time-varying data into two types.

An **Event** is a discrete occurrence. A button click is an Event. Between clicks there is no click value to read. After the handler returns, that occurrence is gone unless something else stored it.

A **Behavior** (Fran wrote *behaviour*; libraries usually say `Behavior`) is a value defined at every moment. The counter total is a Behavior: right now it is some integer, and you can sample it without waiting for the next click.

`hold` builds a Behavior from an initial value and an Event stream of updates:

```text
hold initialValue updateEvents  →  Behavior
```

`map` and `snapshot` compose Events and Behaviors into a pipeline instead of mutating a variable inside a listener.

## Callback baseline: the blank display

This is the usual DOM counter:

```js
let counter = 0;
const display = document.querySelector("#display");
const button = document.querySelector("#btn");

button.addEventListener("click", () => {
  counter = counter + 1;
  display.textContent = counter;
});
```

Timeline:

1. Page loads. `counter` is `0`. The `#display` node still shows whatever the HTML left there (often empty).
2. The click listener is registered.
3. `addEventListener` does **not** run the handler for the current value of `counter`. It only runs for clicks that arrive after registration.
4. Until the first click, the screen never receives `0` from this code path.
5. First click: `counter` becomes `1`, then the display shows `1`.

That gap is the **missed first event** bug in this setting. The state already has a valid value (`0`) before any click. The listener only reacts to future click Events. There is no Event that carries the starting value. If the UI update lives only inside the listener, the initial state never reaches the screen.

The usual fix duplicates the write:

```js
let counter = 0;
const display = document.querySelector("#display");
const button = document.querySelector("#btn");

function render() {
  display.textContent = counter;
}

button.addEventListener("click", () => {
  counter = counter + 1;
  render();
});

render(); // required, or the display stays blank until the first click
```

That fix is not exotic. You already have `counter === 0`. Calling `render()` applies the **current value** to the display without waiting for a click. The click handler later applies the same `render` after each update. Two call sites, one function. Forget the setup call and the screen stays blank.

The same pattern shows up with a range input that starts at `50`: an `input` listener does not fire for the starting value, so you read `.value` once after attach and run the same handler.

Part VI also mentions a related failure: an Event already fired before you subscribed (config finished loading; a late widget never hears `"loaded"`). Same gap: if you only listen for future Events, you never see a value that already exists. Holding that value somewhere you can read now (a variable you `render()`, or a Behavior you sample) closes it.

## FRP version of the same counter

No separate mutable `counter` plus a listener that mutates it. Build:

- `s_click`: Event stream of clicks (payload unused)
- `s_value`: Event stream of increment amounts (`1` per click)
- `s_update`: Event stream of new totals
- `counterBehaviour`: Behavior of the current total, from `hold 0 s_update`

The block below is classic FRP pseudocode in [Haskell](/cs-glossary/haskell/). You do not need to write Haskell; you need the four lines that define the pipeline. Syntax used here:

| Syntax | Meaning |
|-------------|---------|
| `name :: Type` | type annotation (`s_click` has type `Event ()`) |
| `Event a` | discrete stream; each occurrence carries a value of type `a` |
| `Behavior a` | continuous value of type `a`, always sampleable |
| `()` | empty payload (“unit”); a click with no useful data |
| `Int` | integer |
| `const 1` | function that ignores its argument and returns `1` (Python: `lambda _: 1`) |
| `map f events` | apply `f` to each event payload; same timing, new payloads |
| `(+)` | the ordinary add function, written as a value you can pass around |
| `snapshot f ev beh` | when `ev` fires: `f(eventPayload, currentSampleOf(beh))`, emit that |
| `hold init ev` | Behavior that starts at `init`, then updates whenever `ev` fires |
| `f x y` | call `f` with arguments `x` and `y` (no commas, few parentheses) |

```haskell
s_click :: Event ()
-- clicks; payload () means "fired, nothing else"

s_value :: Event Int
s_value = map (const 1) s_click
-- each click → an Int event carrying 1
-- same as Python: s_value = map(lambda _: 1, s_click)

s_update :: Event Int
s_update = snapshot (+) s_value counterBehaviour
-- each s_value Int + current counterBehaviour → new total event
-- first click: snapshot sees 1 and 0, emits 1

counterBehaviour :: Behavior Int
counterBehaviour = hold 0 s_update
-- always has a current Int; starts at 0; follows s_update
```

When an Event arrives, `snapshot` samples the Behavior at that instant, applies `(+)` to the event payload and that sample, and emits the sum. Part VI uses the same idea under the name `tag`.

Pipeline:

```text
Button → s_click → map (const 1) → s_value → snapshot (+) → s_update → hold
                                              ↑ current value
                                         counterBehaviour
```

`counterBehaviour` has value `0` at the moment `hold` creates it. No click is required for that sample to exist. UI code that keeps the label equal to the current Behavior reads `0` at construction: same effect as the callback `render()` at setup.

What FRP changes is structure, not the fix itself. Callbacks keep state in `let counter` and Events in the listener; you must remember to push the current value once by hand. `hold 0 s_update` puts that current value in the Behavior from the start, so anything that samples the Behavior gets `0` without a second code path. The initial `0` is still current state you apply now. It is just not a separate call after the listener.

Timeline:

1. Construct `counterBehaviour = hold 0 s_update`. Current value is `0`.
2. Bind the display to that Behavior. First sample draws `0` (same job as manual `render()`).
3. Click: `s_click` fires → `map` emits `1` on `s_value` → `snapshot (+)` computes `1 + 0` → `s_update` emits `1` → `hold` sets the Behavior to `1`.
4. Display follows the Behavior to `1`. Next click: `1 + 1 = 2`, then `3`, and so on.

Listeners run only for Events after attach. The initial counter is not an Event, so an Event-only UI path skips it unless you also apply the current value at setup (callback `render()`, or sample a Behavior from `hold`). FRP does not remove that requirement; `hold` makes the current value part of the type you bind the UI to, so the setup sample is the same operation as later updates.

## `map` on this pipeline

`map` here is an Event combinator:

```text
map f inputEvents  →  outputEvents
```

For each occurrence on the input stream, `map` applies `f` to the payload and emits one occurrence on the output stream at the same logical time. It does not invent extra Events, drop Events, or delay them. It only changes what the Event carries.

In the counter, `map (const 1)`:

- Input: `s_click`. A real DOM click object has coordinates, button id, timestamp, and more. None of that is the increment.
- Output: `s_value`, an `Event Int` whose payload is always `1`.

`snapshot (+)` needs a numeric left argument. Feeding raw `s_click` into `snapshot` would try to add a click object to the current count. `map` turns “a click occurred” into the numeric event “add 1”.

`map` alone is not the counter. Accumulation is `hold` plus sampling the current Behavior at each delta.

### One click at a time

Before any clicks:

| signal | value |
|--------|--------|
| `s_click` | no occurrences |
| `s_value` | no occurrences |
| `counterBehaviour` | `0` |

First click:

1. `s_click` fires (click payload).
2. `map` emits `1` on `s_value` at the same instant.
3. `snapshot (+)`: `1 + 0 = 1` on `s_update`.
4. `hold` sets `counterBehaviour` to `1`.

Second click: `s_value` is `1` again, `1 + 1 = 2`, Behavior becomes `2`. Third: `1 + 2 = 3`.

Same marble table as Part VI:

| signal | over time |
|--------|-----------|
| `s_click` | • • • |
| `s_value` after `map` | 1 1 1 |
| `s_update` | 1 2 3 |
| Behavior after `hold` | 0 \| 1 \| 2 \| 3 |

## What to keep

The blank display happens when the UI only updates on Events and never applies the current value (`0`) at setup. The callback fix is exactly that apply: `render()` once with `counter` already `0`. FRP’s `hold 0` stores the same current value in a Behavior; sampling the Behavior at bind time is the same step, not a different mechanism.

Events stay discrete. Behaviors always have a current sample. `map` on Events only changes payloads at the same times; here it turns each click into `1` so `snapshot (+)` can add that delta to the value in `hold`.

For Observer, Rx vs classic FRP, and event logs, stay with [Part VI](/2026/11/12/automata-basics-for-coders-part-6/). This page is only the counter, the blank initial display, and `map`.

## References

- Elliott, C. and Hudak, P. Functional Reactive Animation. In *Proceedings of ICFP 1997*. (Fran: behaviours and events in Haskell.)  
  Read in [BetterReads](https://www.betterreads.dev/reader?bookId=elliott_hudak_functional_reactive_animation&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}  
  Author PDF: [conal.net](https://conal.net/papers/icfp97/icfp97.pdf){:target="_blank" rel="noopener noreferrer"}
- Teaching libraries in the same model: Reactive Banana, Sodium, Yampa.
- [Automata Basics for Coders, Part VI](/2026/11/12/automata-basics-for-coders-part-6/).
