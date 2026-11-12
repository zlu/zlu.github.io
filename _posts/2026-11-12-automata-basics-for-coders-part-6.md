---
layout: post
title: "Automata Basics for Coders, Part VI: Delivering Events (Observer, FRP, Logs)"
date: 2026-11-12
comments: true
tags:
  - automata
  - event-driven
  - Observer pattern
  - FRP
  - event log
  - software architecture
description: "How software delivers the next input symbol: Observer notify, classic FRP Event and Behavior (Fran / Haskell), Rx as a cousin, React mappings, and durable event logs."
---

Parts [I](/2026/11/06/automata-basics-for-coders-part-1/)–[III](/2026/11/09/automata-basics-for-coders-part-3/) treated **strings over an alphabet** and machines that accept or reject them. [Part IV](/2026/11/10/automata-basics-for-coders-part-4/) made the machine a **[transducer](/cs-glossary/transducer/)**: each input symbol updates state and can emit output. [Part V](/2026/11/11/automata-basics-for-coders-part-5/) covered several machines that share resources (Petri nets).

Those parts assume an input sequence is available. This part is how a program **delivers** symbols and how consumers keep state after a prefix.

**Question.** Given an event sequence $e_1 e_2 e_3 \ldots$ over $\Sigma$ (button presses, `UserRegistered`, socket frames): how do consumers see those symbols, and what state do they keep?

| Mechanism | Delivery | Relation to automata |
|-----------|----------|----------------------|
| **Observer** | in-process `notify` on each occurrence | broadcast the next input symbol |
| **[FRP](/cs-glossary/frp/)** | compose `Event` and `Behavior` | [transducer](/cs-glossary/transducer/) over the event string; Behavior ≈ state / Moore output after the prefix |
| **Event log** | append symbols to a durable ordered log | store the string; many machines replay prefixes |

Same $\Sigma$ and the same “state after a prefix.” The difference is whether the string is discarded after `notify`, folded into a Behavior, or stored for replay.

Content:

1. Delivery after [transducers](/cs-glossary/transducer/) and Petri nets  
2. Observer: in-process notify (GoF / [Smalltalk](/cs-glossary/smalltalk/))  
3. [FRP](/cs-glossary/frp/): event-driven pain → `Event` + `Behavior` → combinators ([Haskell](/cs-glossary/haskell/))  
4. Transactions; Rx as cousin; [React](/cs-glossary/react/) mapping  
5. Unified event log: durable string, many readers  
6. Cheat sheet and self-check  

---

## Why this belongs in the series

In Part I a language is a set of strings over $\Sigma$. A click counter can use $\Sigma = \{\mathsf{click}\}$. A registration pipeline can use $\Sigma = \{\mathsf{UserRegistered}, \mathsf{PaymentFailed}, \ldots\}$. The event history is one concrete string.

A DFA, Moore machine, or Petri-net marking answers: after the prefix seen so far, what is the **current configuration**? Running software still needs:

1. emit the next symbol,  
2. let zero or more consumers react,  
3. optionally store the string so a late or restarted consumer can catch up.

Observer, FRP, and event logs are three ways to do that.

---

## Observer pattern

Simplest delivery: do not store the string. On each new symbol, call the registered listeners.

The name most people meet first is from Gamma, Helm, Johnson, and Vlissides (*Design Patterns: Elements of Reusable Object-Oriented Software*, 1994), often called the **Gang of Four** (GoF). Their intent line is: define a **one-to-many** dependency so that when one object changes state, all dependents are notified and updated automatically.

The idea is older than that book. [Smalltalk](/cs-glossary/smalltalk/)-80’s **dependents** mechanism (model broadcasts `changed` / `changed:`; views and controllers respond with `update` / `update:`) is the usual source the GoF catalog points at. Krasner and Pope documented that MVC / dependents loop in 1988. The same four authors also listed Observer in their earlier ECOOP’93 paper before the book’s full write-up.

### Where you already see it

| Setting | Subject (source) | Observers (dependents) |
|---------|------------------|------------------------|
| Spreadsheet | a cell’s value | formulas that read it; charts bound to the range |
| IDE / editor | document buffer | syntax highlighter, outline, status bar, dirty-flag |
| Stock / metrics dashboard | latest quote or gauge reading | table row, sparkline, alert rule |
| GUI toolkit | button / toggle / model object | labels, loggers, analytics sinks |

**Spreadsheet → chart.** Edit a table cell. Dependent formulas recompute. A chart subscribed to that range redraws. The cell store does not import the chart library; it notifies whoever registered.

### Components

| Component | Role |
|-----------|------|
| **Subject** (observable) | holds state or is the event source |
| **Observer** interface | e.g. `update(...)` / `onEvent(...)` |
| **Concrete observers** | views, loggers, analytics sinks |
| **subscribe / attach** | register an observer with the subject |
| **unsubscribe / detach** | remove it from the list |
| **notify** | subject calls each registered observer |

The subject stores a list of observers. On a change, it calls `notify`. It does not hard-code each concrete observer class.

### Toy GUI

- Subject: `Toggle`  
- Event: switched on/off  
- Observers: (1) update a status label, (2) write a log line, (3) refresh a chart  

Flow: user flips the toggle → `Toggle.notify()` → each subscribed object runs its handler. The `Toggle` class does not import the chart implementation.

### Why this structure is used

1. **Loose coupling:** the subject depends on an observer interface, not on each listener’s class.  
2. **Many listeners:** one event, $N$ reactions.  
3. **Dynamic registration:** subscribe when a panel opens; unsubscribe when it closes.  
4. **Separation of concerns:** reaction logic lives in observers.

### Automata reading

Treat the subject’s event type as a symbol from $\Sigma$. Each `notify` feeds that symbol to every attached consumer. Each concrete observer updates its own state (and may emit UI). The subject does not keep the full input string. After `notify` returns, that occurrence is gone unless an observer stored it. FRP Behaviors and event logs address that gap.

---

## FRP: Event and Behavior

Observer answers who to call when the next symbol arrives. It does not treat time-varying values as data you can compose.

**[FRP](/cs-glossary/frp/)** (functional reactive programming) = programming with values that vary over time, as first-class entities. The early formulation is Elliott and Hudak, *Functional Reactive Animation* (Fran, ICFP 1997), in [Haskell](/cs-glossary/haskell/). Later teaching libraries in the same model: Yampa, Reactive Banana, Sodium.

Two primitives (Fran said *behaviour*; most Haskell libraries say `Behavior`):

| Type | Meaning |
|------|---------|
| `Event a` | discrete **occurrences** over time; each carries a value of type `a`; no value between occurrences |
| `Behavior a` | a value of type `a` defined at **every** time $t$; always has a current sample |

A mouse-move burst is an `Event`. The current mouse $x$ is a `Behavior Double`. A button press is an `Event ()`. Spreadsheet cell `A1` is a Behavior; finishing an edit is closer to an Event that updates it.

**Litmus test.** If the library has a first-class value that exists at every moment (not only when something emits), it is closer to classic FRP. If everything is a stream of discrete emissions, it is Rx-style reactive programming (cousin, not the same model). Starting from RxJS / RxDart and calling that “FRP” skips the Behavior half.

In series terms: an `Event` sequence is a string over $\Sigma$; a `Behavior` is state (or Moore output) after the prefix so far. Combinators are how you write the [transducer](/cs-glossary/transducer/) without a hand-rolled observer list.

### Baseline: plain event-driven (the pain)

Callbacks + mutable locals. Mouse $x$ and click:

```text
var mouseX = 0
onMouseMove(x) { mouseX = x }
onClick()      { drawAt(mouseX) }
```

Questions this raises:

- Who owns `mouseX`?  
- What if move and click are ordered differently than you assumed?  
- How do you combine mouse position with a held key without more shared mutables?  
- How do you unsubscribe without leaking handlers?

Event-driven code is **actions that run when something happens**. State is outside the handlers and coordinated by hand. That is Observer / DOM `addEventListener` territory.

### The FRP shift

Instead of “on event, mutate state,” you **define** time-varying values as functions of other time-varying values.

```haskell
-- Classic FRP types (Fran / Reactive Banana / Sodium style)
-- Event a      -- discrete occurrences
-- Behavior a   -- continuous current value

mouseMove :: Event Double    -- x on each move
click     :: Event ()

mouseX :: Behavior Double
mouseX = hold 0 mouseMove    -- Event → Behavior (initial 0)

-- sample Behavior at the instant Event fires (Sodium: snapshot; often called tag)
clickPos :: Event Double
clickPos = tag mouseX click
```

`tag` (also `snapshot`) is the operation Rx does not make first-class: read the current Behavior when an Event occurs. Between clicks, `clickPos` has no value; `mouseX` always does.

### Core combinators (minimal set)

**On `Event`:** `map` (transform payload), `filter`, `merge`, `tag` / `snapshot` (sample a Behavior), `accumE` / fold into a new Event of running state.

**On `Behavior`:** `map`, apply / combine several Behaviors, `hold` (Event + initial → Behavior), `accumB` (fold Events into a Behavior).

Running example (click positions, then a list of them):

```haskell
mouseMove :: Event Double
click     :: Event ()

mouseX :: Behavior Double
mouseX = hold 0 mouseMove

clickPos :: Event Double
clickPos = tag mouseX click

-- Behavior of all click x positions so far (newest at the front)
points :: Behavior [Double]
points = accumB [] ((:) <$> clickPos)
```

The UI **consumes** `points` (draw the list). It is not the coordinator of `onMouseMove` / `onClick` mutables. State is **derived**.

### Toy: click counter (`map`, `hold`, `tag`)

Same idea over $\Sigma = \{\mathsf{click}\}$. Count clicks; Behavior holds the total (Moore-style “output = current state”).

```haskell
clicks :: Event ()
deltas :: Event Int
deltas = map (const 1) clicks          -- payload 1 per click; map does not accumulate

-- sample current count at each delta (Sodium snapshot / classic tag)
updates :: Event Int
updates = (\c d -> c + d) <$> snapshot count deltas

count :: Behavior Int
count = hold 0 updates

label :: Behavior String
label = map show count
```

Pipeline:

```text
clicks  --map(const 1)-->  deltas  --snapshot(+)-->  updates
                                   ↑ current
                                [ hold ]  <--------- updates
```

| signal | over time |
|--------|-----------|
| clicks | • • • |
| deltas after `map` | 1 1 1 |
| `count` after `hold` | 0 \| 1 \| 2 \| 3 |
| updates | 1 2 3 |

| primitive | job |
|-----------|-----|
| `map` | click → delta $1$ (no accumulation) |
| `hold` | current count (Behavior) |
| `tag` / `snapshot` | newCount = oldCount + 1 at event time |

`map` alone is not a counter. Accumulation needs `hold` plus sampling the current Behavior.

### Missed first event vs Behavior

1. An Event fires before a listener subscribes.  
2. The listener attaches later.  
3. That past occurrence is not delivered again.

Example: config finished loading at startup; a widget that subscribes afterwards never hears `"loaded"`.

A **Behavior** (UI people often say **cell**) always has a current value. A late reader samples it now. Correctness does not require having been subscribed at the first change. “FRP is asynchronous” does not state this distinction; the Event / Behavior split does.

### Spreadsheet as Behaviors

| Spreadsheet | FRP |
|-------------|-----|
| `A1` holds `10` | `Behavior` with current `10` |
| user commits a new number | `Event` that updates the cell |
| `B1 = A1 * 2` | `map` / arithmetic on a Behavior |
| chart on `B1` | another reader of that Behavior |

### Transactions (why purists separate FRP from Rx)

Classic FRP updates are **transactional**: occurrences at the same logical instant are processed together; you do not observe a half-updated intermediate Behavior. Sodium is built around that. Rx `Observable` chains fire sequentially by default; simultaneous logical events are not one transaction unless you build that yourself.

### Rx after the model (cousin, not the core)

Only after `Event` + `Behavior` is clear:

| Classic FRP | Rx (RxJS / RxDart / …) |
|-------------|-------------------------|
| `Event a` | `Observable<a>` (roughly) |
| `Behavior a` | no true continuous Behavior; `BehaviorSubject` remembers last emission |
| `accumB` / fold | `scan` |
| `tag` / `snapshot` | not primitive; sample by hand |
| focus | values over time + sampling | async IO, debounce, `switchMap`, cancellation |

Rx is a popular **discrete reactive stream** library inspired by FRP ideas. It is not classic FRP. Same warning for “FRP” marketing around UI state libraries that only expose streams.

### Boundaries (short)

| Model | Core objects |
|-------|----------------|
| Event-driven / Observer | callbacks on discrete occurrences |
| Classic FRP | `Event` + `Behavior`, sampling, often transactional updates |
| Rx-style | `Observable` streams only |
| Signal / fine-grained reactivity (MobX, Solid, Vue) | dependency-tracked cells; not classic FRP, not Rx |

[React](/cs-glossary/react/) is a render layer that can **consume** state. It is not an FRP runtime.

### In React (mapping, not identity)

I use React because the mapping is short; [Next.js](/cs-glossary/nextjs/) uses the same client hooks.

| Classic FRP | React-ish counterpart |
|-------------|------------------------|
| `Behavior` (current value) | `useState` / props present on first render |
| `Event` | `onClick`, `onChange`, socket push |
| `hold` + `tag` / fold | `setCount((c) => c + 1)` |
| subscribe / unsubscribe | `useEffect` + cleanup |
| late joiner reads Behavior | child mounts and reads current `count` |

```jsx
function ClickCounter() {
  const [count, setCount] = useState(0); // Behavior-like current total

  function handleClick() {
    // Event: one click; fold into the Behavior
    setCount((c) => c + 1);
  }

  return <button onClick={handleClick}>{count}</button>;
}
```

External pushes still look like Observer:

```jsx
useEffect(() => {
  function onMessage(e) {
    setLatest(JSON.parse(e.data));
  }
  socket.addEventListener("message", onMessage);
  return () => socket.removeEventListener("message", onMessage);
}, [socket]);
```

If a message arrived before this effect ran, this listener missed that **Event**. Keep the latest payload in state (a Behavior / cell) if late UI must show current data.

React does not provide `hold` / `tag` as named primitives. The split still holds: Events are occurrences; state is the current Behavior you can read after you arrive.

Observer delivered each click by callback. FRP named the `Event` and the derived `Behavior`. Neither stores the full click string. An event log does: append the symbols, then let consumers read them.

---

## Unified event log vs tight observers

### Tight observers (recap)

Producers call consumers in-process (`notify`). Low latency. The subject holds references to listeners. The input string is usually not retained. After notify, history is gone unless something else stored it. Replay needs an explicit log.

### Unified event log

Facts are **appended** to an ordered log. The log is the string $e_1 e_2 e_3 \ldots$ over the domain alphabet. Independent consumers **read** it on their own schedule (Kafka-style systems are the usual example). Each consumer is a [transducer](/cs-glossary/transducer/) / state machine: offset = how much of the string it has processed; local tables = state after that prefix.

Part I: a language is a set of strings; here the system emits one concrete string (the log). Part V: several machines share a concurrent world; here they share an append-only history.

**Toy.** Service writes `UserRegistered` records to a log ($\Sigma$ includes that symbol).

1. **Email** service reads the log and sends a welcome message.  
2. **Billing** reads the same log and creates an account record.  
3. **Analytics** reads the log for funnels.  

The registration service does not import those three codebases. A reader can restart from an earlier offset and re-run on a prefix of the stored string.

### Benefits

| Benefit | Meaning | Automata reading |
|---------|---------|------------------|
| Decoupling | writers need not know readers | producer only appends to the string |
| Replay | rebuild state from history | re-run the machine on a prefix |
| Audit | durable record of what occurred | the stored string is the record |
| Multiple consumers | add a reader without changing the writer | many transducers, one input string |
| Temporal decoupling | a consumer can catch up after downtime | late joiner processes missed symbols |

### One cost

Operational complexity, eventual consistency between readers, or retention / storage growth of the log.

### Comparison

| | Where the string lives | What a consumer keeps | Late joiner |
|--|------------------------|----------------------|-------------|
| Observer | not retained (push only) | whatever the handler stored | misses past symbols unless state was added |
| FRP Behavior | usually not retained; Behavior holds current state after the prefix | the Behavior | sees current state; not every past payload unless logged |
| Event log | durable ordered log | offset + derived state | replays symbols from the log |

FRP makes **state after the prefix** explicit (the Behavior). It does not require storing every symbol for every consumer.

---

## Cheat sheet

```text
Core           event string over Σ → deliver symbols → keep / derive state
Observer       Subject, subscribe/unsubscribe, notify, concrete observers
               in-process broadcast of the next symbol; string not retained
GoF            one-to-many notify on state change (1994 book; ECOOP’93 catalog)
Source idea    Smalltalk dependents / MVC (Krasner & Pope 1988)
FRP            time-varying values as first-class data (Fran / Haskell)
Event a        discrete occurrences; no value between them
Behavior a     value at every time t (Fran: behaviour); always sampleable
hold           Event + initial → Behavior
tag/snapshot   sample Behavior at Event time
map            transform Event payload or Behavior value (no accumulation)
accumB         fold Events into a Behavior
vs Rx          Observable ≈ Event only; not classic Behavior + tag
vs Observer    Observer = who to call; FRP = how Event/Behavior are derived
React          useState ≈ Behavior; onClick ≈ Event; not an FRP runtime
Event log      durable ordered string; offset = how much prefix consumed
log vs notify  durable log + many readers vs in-process callbacks
```

---

## Self-check

1. In one line: what shared question ties Observer, FRP, and event logs to Parts I–V?  
2. Name the six Observer pieces from memory.  
3. Give one production example (subject + at least one observer).  
4. State the FRP axiom and name the two classic primitives.  
5. What does `tag` / `snapshot` do that plain `map` on Events does not?  
6. Give a one-line “missed first event” failure and the Behavior-based fix.  
7. Why is “RxJS is FRP” the wrong teaching start?  
8. In React terms: what is closer to a Behavior, and what is closer to an Event?  
9. Does `map (const 1)` alone implement a counter?  
10. Map each mechanism to “where the event string lives.”  
11. State two benefits of a unified log over `notify`, and one downside.

### Answers

**1.** How software delivers symbols of an event sequence (a string over $\Sigma$) to consumers and keeps state after a prefix.

**2.** Subject, Observer interface, concrete observers, subscribe, unsubscribe, notify.

**3.** e.g. spreadsheet cell → dependent formulas and chart; or document buffer → highlighter and outline.

**4.** FRP = program with time-varying values as first-class data. Primitives: `Event a`, `Behavior a`.

**5.** Samples a Behavior at the instant an Event fires (payload from continuous state, not only from the Event).

**6.** Failure: subscribe after “config loaded”; never hear it. Fix: hold config in a Behavior; read current value on join.

**7.** Rx exposes stream/`Observable` ≈ Event only; classic FRP also has continuous Behavior and first-class sampling (`tag` / `snapshot`).

**8.** Behavior: `useState` / current props. Event: click / `message` (or any push you listen for).

**9.** No. It only produces deltas. Accumulation needs `hold` plus `snapshot` / `tag` (or `accumB`).

**10.** Observer: not retained (push only). FRP Behavior: current state after the prefix. Event log: durable ordered log.

**11.** Benefits: any two of decoupling, replay, audit, independent consumers, temporal decoupling. Downside: e.g. ops complexity, eventual consistency, or log retention cost.

---

## Series map

| Part | Topic |
|------|-------|
| [I](/2026/11/06/automata-basics-for-coders-part-1/) | languages, grammars, accepters (Linz Ch 1) |
| [II](/2026/11/07/automata-basics-for-coders-part-2/) | DFA, NFA, subset construction (Linz Ch 2) |
| [III](/2026/11/09/automata-basics-for-coders-part-3/) | regex, ε-NFA, lazy remove ε (Linz Ch 3) |
| [IV](/2026/11/10/automata-basics-for-coders-part-4/) | Moore / Mealy transducers (Linz App A) |
| [V](/2026/11/11/automata-basics-for-coders-part-5/) | Petri nets, reachability, safeness |
| **VI** (this post) | delivering events: Observer, classic FRP, logs |

---

## References

- Gamma, E., Helm, R., Johnson, R., and Vlissides, J. *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley, 1994. (Observer / “Gang of Four.” No free full PDF hosted here.)  
- Gamma, E., Helm, R., Johnson, R., and Vlissides, J. Design Patterns: Abstraction and Reuse of Object-Oriented Design. In *ECOOP’93*, LNCS 707, Springer, 1993, pp. 406–431. (Earlier catalog entry that includes Observer.)  
  Read in [BetterReads](https://www.betterreads.dev/reader?bookId=gamma_design_patterns_ecoop93&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}
- Krasner, G. E. and Pope, S. T. A Description of the Model-View-Controller User Interface Paradigm in the Smalltalk-80 System. *Journal of Object-Oriented Programming*, 1(3):26–49, 1988. (Dependents / MVC change notification.)  
  Read in [BetterReads](https://www.betterreads.dev/reader?bookId=krasner_pope_mvc_smalltalk80&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}
- Elliott, C. and Hudak, P. Functional Reactive Animation. In *Proceedings of ICFP 1997*. (Fran: behaviours and events in Haskell.)  
  Read in [BetterReads](https://www.betterreads.dev/reader?bookId=elliott_hudak_functional_reactive_animation&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}  
  Author PDF: [conal.net](https://conal.net/papers/icfp97/icfp97.pdf){:target="_blank" rel="noopener noreferrer"}
- Teaching libraries in the same classic model: Reactive Banana (Haskell), Sodium (several hosts), Yampa (Haskell arrowized FRP). RxJS / RxDart are discrete-stream cousins, not drop-in FRP.
- [Part I](/2026/11/06/automata-basics-for-coders-part-1/)–[Part V](/2026/11/11/automata-basics-for-coders-part-5/).
