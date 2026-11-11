---
layout: post
title: "Automata Basics for Coders, Part IV: Moore and Mealy Machines"
date: 2026-11-10
comments: true
tags:
  - automata
  - formal languages
  - theory of computation
  - Moore machine
  - Mealy machine
description: "Linz Appendix A in plain terms: transducers with output, Moore and Mealy definitions, a side-by-side timing demo, a design checklist for controller briefs, and worked Moore toys (traffic light, desk lamp, microwave)."
---

Parts [I](/2026/11/06/automata-basics-for-coders-part-1/)–[III](/2026/11/09/automata-basics-for-coders-part-3/) treated **accepters**: machines that read a string and answer yes/no. Linz also defines **transducers**: machines that produce an **output** while they run. The reference is Linz and Rodger, [*An Introduction to Formal Languages and Automata*](https://www.betterreads.dev/discover/ia_an-introduction-to-formal-languages-and-automata-6th-edition?q=An%20Introduction%20to%20Formal%20Languages%20and%20Automata){:target="_blank" rel="noopener noreferrer"}, **Appendix A** (Mealy A.2, Moore A.3, equivalence A.4).

Content:

1. Accepter vs transducer  
2. Moore machine (definition)  
3. Mealy machine (definition)  
4. Same job both ways: timing demo  
5. Design method for a controller brief  
6. Toy: traffic light (Moore)  
7. Toy: two-button lamp (Moore with assumptions)  
8. Toy: microwave (Moore controller)  
9. Cheat sheet and self-check  

---

## Accepter vs transducer

| | Accepter (Parts I–III) | Transducer |
|--|------------------------|------------|
| Input | string over $\Sigma$ | string / event sequence over $\Sigma$ |
| Output | accept or reject at the end | a symbol (or vector) **at each step** |
| Typical use | “is $w$ in $L$?” | control a motor, lamp, UI mode |

A finite transducer still has a finite state set $Q$, an input alphabet $\Sigma$, a start state, and a transition function. It also has an **output alphabet** $\Gamma$ and a rule that assigns outputs. Where that rule attaches is the Moore / Mealy split.

For controllers, think of $\Sigma$ as **events** (sensor edges, button presses, timer ticks) and $\Gamma$ as **commands** to the outside world (motor direction, lamp level, UI mode). The machine is no longer answering a language question; it is driving hardware or software while time and sensors unfold.

---

## Moore machine

### Definition

A **Moore machine** is a six-tuple

$$
M = (Q,\ \Sigma,\ \Gamma,\ \delta,\ \lambda,\ q_0)
$$

where:

| Component | Meaning |
|-----------|---------|
| $Q$ | finite set of **states** |
| $\Sigma$ | finite **input** alphabet |
| $\Gamma$ | finite **output** alphabet |
| $\delta: Q \times \Sigma \to Q$ | **transition** function (total, deterministic) |
| $\lambda: Q \to \Gamma$ | **output** function: each state has a fixed output |
| $q_0 \in Q$ | **start** state |

There is no accept set $F$. The job is not yes/no; it is the output stream.

**Rule.** While the machine sits in state $q$, the output is $\lambda(q)$. The output changes only when a transition moves the machine into a state with a different $\lambda$.

**Entry-action reading (useful for code).** Entering $q$ “turns on” $\lambda(q)$ and keeps it until you leave. Arcs only decide *which* state comes next; they do not carry the command. That is why Moore drawings put the command **inside** the bubble.

**Diagram.** Write the output **inside or beside the state**. Label arcs with the **input** only.

**Run.** On input $a_1 a_2 \ldots a_n$, the machine visits states $q_0, q_1, \ldots, q_n$ with $q_{i} = \delta(q_{i-1}, a_i)$. The Moore output sequence is

$$
\lambda(q_0),\ \lambda(q_1),\ \ldots,\ \lambda(q_n)
$$

That is **$n+1$** symbols: one for the start state before any input, then one after each transition.

**Why Moore fits “modes.”** If a mode has a steady command—“while Heating, magnetron = ON”—the output is a property of the mode. Knowing the current state is enough to know the command. That is the Moore contract.

---

## Mealy machine

### Definition

A **Mealy machine** is a six-tuple

$$
M = (Q,\ \Sigma,\ \Gamma,\ \delta,\ \lambda,\ q_0)
$$

Same components as Moore, except the output function is

$$
\lambda: Q \times \Sigma \to \Gamma
$$

So the output depends on the **current state and the input symbol** that is being read: the pair $(q,a)$ that labels the transition about to be taken.

**Rule.** The machine emits $\lambda(q,a)$ when it takes the $a$-edge out of $q$. There is no output attached to “just sitting in a state.”

**Diagram.** Label arcs `input / output`. States carry names only (no $\lambda$ on the node).

**Run.** On input $a_1 a_2 \ldots a_n$, visiting $q_0 \xrightarrow{a_1} q_1 \xrightarrow{a_2} \cdots \xrightarrow{a_n} q_n$, the Mealy output sequence is

$$
\lambda(q_0,a_1),\ \lambda(q_1,a_2),\ \ldots,\ \lambda(q_{n-1},a_n)
$$

That is **$n$** symbols: one per input, none for the idle start state.

| | Moore | Mealy |
|--|-------|-------|
| Output attached to | **state** ($\lambda: Q \to \Gamma$) | **transition** ($\lambda: Q \times \Sigma \to \Gamma$) |
| Diagram | output in / beside the state | `input / output` on the arc |
| Output length on $\|w\|=n$ | $n+1$ (includes start) | $n$ |
| When the output changes | on **entering** a state | on **taking** an arc |
| Good fit | steady mode / hold a command | pulse / react on this edge |

Linz Appendix A.4: Moore and Mealy are equivalent in expressive power. Each can simulate the other. The usual catch is a **one-step shift** in when the output appears (exactly the $n+1$ vs $n$ difference above). Moore often needs more states for the same job, because the output must be “remembered” as which state you are in; Mealy can put that difference on the arcs.

---

## Same job both ways: timing demo

**Job.** Input alphabet $\Sigma = \{0,1\}$. Output alphabet $\Gamma = \{0,1\}$. Emit the **current input bit** as the output bit (echo).

### Mealy version

One state is enough. The output lives on the arc, so it can equal the input without remembering anything.

| From | Input | To | Output |
|------|-------|----|--------|
| $s$ | $0$ | $s$ | $0$ |
| $s$ | $1$ | $s$ | $1$ |

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> s
    s --> s: 0 / 0
    s --> s: 1 / 1
</div>

### Moore version

The output is a function of the state only, so the machine must **be in a state whose $\lambda$ matches the last bit read**. Two states:

| State | $\lambda$ (output) |
|-------|--------------------|
| $Z$ | $0$ |
| $O$ | $1$ |

Start in $Z$ (before any input, treat the “last bit” as $0$).

| From | Input | To |
|------|-------|----|
| $Z$ | $0$ | $Z$ |
| $Z$ | $1$ | $O$ |
| $O$ | $0$ | $Z$ |
| $O$ | $1$ | $O$ |

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> Z
    Z --> Z: 0
    Z --> O: 1
    O --> Z: 0
    O --> O: 1
</div>

On paper, mark $Z / 0$ and $O / 1$ on the nodes. Arcs carry only $0$ or $1$.

### Trace on the same string $1101$

| Step | Input | Mealy: arc taken | Mealy out | Moore: state after step | Moore out |
|------|-------|------------------|-----------|-------------------------|-----------|
| (start) | | | | $Z$ | $0$ |
| 1 | $1$ | $s \xrightarrow{1/1} s$ | $1$ | $O$ | $1$ |
| 2 | $1$ | $s \xrightarrow{1/1} s$ | $1$ | $O$ | $1$ |
| 3 | $0$ | $s \xrightarrow{0/0} s$ | $0$ | $Z$ | $0$ |
| 4 | $1$ | $s \xrightarrow{1/1} s$ | $1$ | $O$ | $1$ |

- Mealy output string: $1101$ (length $4$).  
- Moore output string: $01101$ (length $5$).  

Drop the Moore start symbol and the remaining $1101$ matches Mealy. That is the one-step shift: Moore reports $\lambda$ of the state **after** the move (and also reports the start), Mealy reports the output **on** the move.

If a homework asks for “the output for $w$,” check whether the course counts the initial Moore symbol. Linz does; some slides drop it.

---

## How to design from a brief

Controller briefs are incomplete on purpose. A clean answer is a diagram **plus** the assumptions that make the diagram determinate.

1. List **inputs** (sensors, buttons, timers)—these become arc labels.  
2. List **outputs** (actuators, lamps, motor commands)—these become $\Gamma$.  
3. Decide **Moore or Mealy**: fixed mode displays / hold a command → Moore; pulse / “react to this edge” → Mealy is often shorter.  
4. List **modes** the system can be in → candidate states. For Moore, every distinct steady command usually needs its own state (or a state with that $\lambda$).  
5. Write **assumptions** explicitly: power-on / start state, how you combine sensors, what you do with illegal combinations, reopen / priority rules.  
6. Draw: Moore puts $\lambda$ on states and inputs on arcs; Mealy puts `input / output` on arcs.  
7. **Completeness:** for every state and every input that can occur, decide stay / move / ignore. Missing self-loops are a common bug.  
8. Check every required behaviour against a **path** and an **output column** in the diagram.

The toys below are **Moore**: each mode has a steady output.

---

## Toy 1: traffic light (Moore)

**Inputs:** `tick` (timer elapsed).  
**Outputs:** lamp colour in $\Gamma = \{\mathrm{RED},\mathrm{GREEN},\mathrm{YELLOW}\}$.

| State | $\lambda$ |
|-------|-----------|
| Red | RED |
| Green | GREEN |
| Yellow | YELLOW |

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> Red
    Red --> Green: tick
    Green --> Yellow: tick
    Yellow --> Red: tick
</div>

Arcs carry only `tick`. The lamp colour is $\lambda$ of the current state, not of the arc. A Mealy drawing of the same cycle would put `tick / GREEN` on `Red → Green`, `tick / YELLOW` on `Green → Yellow`, and so on; the colour would be named on the edge that *enters* the mode rather than on the mode itself.

---

## Toy 2: desk lamp with two buttons

**Brief.**

1. On power-up, the lamp is **off**.  
2. Pressing **On** turns the lamp on and keeps it on until **Off** is pressed.  
3. Pressing **Off** while on turns the lamp off.  
4. Pressing **On** while already on leaves the lamp on.  
5. Pressing **Off** while already off leaves the lamp off.

### Assumptions (write them down)

1. Power-up starts in `Off`.  
2. Inputs are instantaneous press events `on`, `off`.  
3. Simultaneous on+off is ignored (or treated as no change).  
4. No dimming levels: output is only `LAMP_ON` or `LAMP_OFF`.

### Inputs and outputs

| Inputs (arcs) | Outputs (states) |
|---------------|------------------|
| `on`, `off` | `LAMP_OFF`, `LAMP_ON` |

### Moore diagram

| State | $\lambda$ |
|-------|-----------|
| Off | LAMP_OFF |
| On | LAMP_ON |

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> Off
    Off --> On: on
    Off --> Off: off
    On --> Off: off
    On --> On: on
</div>

### Trace against the brief

| Situation | Path | Output while in state |
|-----------|------|------------------------|
| power-up | enter `Off` | LAMP_OFF |
| press On | `Off --on--> On` | LAMP_ON |
| press Off | `On --off--> Off` | LAMP_OFF |
| On then On again | `On --on--> On` | stays LAMP_ON |

### Same lamp as Mealy (labelling only)

Keep the same two states and the same $\delta$. Move the lamp command onto the arcs:

| Arc | Mealy label |
|-----|-------------|
| `Off --on--> On` | `on / LAMP_ON` |
| `On --off--> Off` | `off / LAMP_OFF` |
| `Off --off--> Off` | `off / LAMP_OFF` |
| `On --on--> On` | `on / LAMP_ON` |

Same control behaviour class; different place to write the output. The echo demo above is where the **timing** difference shows up cleanly; the lamp is where the **drawing** difference shows up.

---

## Toy 3: microwave oven (Moore)

A longer controller brief. Same design habit: inputs, outputs, modes, assumptions, then a Moore diagram with commands **on states**.

### Brief (plain English)

- The oven has a magnetron (cook power) and a lamp.  
- Events: door opens / closes, user presses Start or Cancel, cook timer expires (`tickDone`).  
- Power-up: idle, door closed, nothing cooking.  
- With the door closed, Start begins cooking until the timer finishes or the user cancels.  
- Opening the door while cooking **pauses** (magnetron off, lamp on). Closing the door alone does not resume—Start must be pressed again.  
- When the timer finishes with the door closed, enter a “done” mode (magnetron off, beep/lamp cue) until Cancel or the door opens.

### Assumptions

1. Power-up starts in `Idle` (door closed).  
2. Start is ignored if the door is open.  
3. Cancel from any non-idle mode returns to `Idle` (door still closed) or to a door-open idle if the door is open—here we simplify: Cancel always goes to `Idle`, and we require the door to be closed in `Idle`.  
4. No power levels: magnetron is only ON or OFF. Lamp is ON whenever the door is open or we are in `Done`.  
5. `tickDone` only matters while cooking; ignore it elsewhere (self-loop / stay).  
6. We do not model setting the cook time—assume time was already set before Start.

### Inputs and outputs

| Inputs (arcs) | Outputs $\Gamma$ (pairs on states) |
|---------------|-------------------------------------|
| `doorOpen`, `doorClose`, `start`, `cancel`, `tickDone` | magnetron ∈ {ON, OFF}, lamp ∈ {ON, OFF} |

Write $\lambda$ as a pair `(magnetron, lamp)`.

### States and $\lambda$

| State | $\lambda$ | Role |
|-------|-----------|------|
| Idle | (OFF, OFF) | door closed, waiting |
| DoorOpen | (OFF, ON) | door open, not cooking |
| Cooking | (ON, OFF) | door closed, magnetron on |
| Paused | (OFF, ON) | door opened mid-cook; waiting to resume |
| Done | (OFF, ON) | timer finished; wait for Cancel or door |

### Moore diagram

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> Idle
    Idle --> DoorOpen: doorOpen
    Idle --> Cooking: start
    DoorOpen --> Idle: doorClose
    Cooking --> Done: tickDone
    Cooking --> Paused: doorOpen
    Cooking --> Idle: cancel
    Paused --> Cooking: start
    Paused --> DoorOpen: cancel
    Done --> Idle: cancel
    Done --> DoorOpen: doorOpen
</div>

On paper, write the `(magnetron, lamp)` pair **inside** each bubble. Arcs carry only events.

### Trace against the brief

| Behaviour | Path | Output while in mode |
|-----------|------|----------------------|
| Power-up idle | enter `Idle` | (OFF, OFF) |
| Start cook, run to end | `Idle --start→ Cooking --tickDone→ Done` | (ON, OFF) then (OFF, ON) |
| Open door mid-cook, resume | `Cooking --doorOpen→ Paused --start→ Cooking` | pause (OFF, ON), then cook again |
| Cancel while cooking | `Cooking --cancel→ Idle` | back to (OFF, OFF) |

### Completeness notes

- From `Idle`, `cancel` / `tickDone` stay in `Idle` (draw self-loops if your course demands every event).  
- From `DoorOpen`, `start` is ignored (assumption 2)—self-loop.  
- From `Paused`, `doorClose` alone does **not** resume; only `start` returns to `Cooking`.  
- A one-state “OvenOn” drawing fails Moore: Idle, Cooking, and Paused need **different** $\lambda$ pairs.

### Mealy sketch (optional contrast)

Same five modes, same $\delta$, but put e.g. `start / (ON, OFF)` on `Idle → Cooking` and `doorOpen / (OFF, ON)` on `Cooking → Paused`. Commands move onto arcs; bubbles lose $\lambda$. Same informal behaviour class; different attachment point—and stream comparison still needs the $n$ vs $n+1$ alignment from the echo demo.

---

## Common mistakes

| Mistake | Correction |
|---------|------------|
| Outputs only on arcs for a “Moore” answer | Move outputs onto states ($\lambda: Q \to \Gamma$) |
| Outputs only on states for a “Mealy” answer | Move outputs onto arcs ($\lambda: Q \times \Sigma \to \Gamma$) |
| Comparing Moore and Mealy strings without aligning length | Moore has $n+1$ symbols; drop or keep the start per course convention |
| No assumptions | State power-on, sensors, ignored cases, reopen / priority |
| One state for everything | Split modes that need different Moore outputs |
| Missing self-loops | If an input is possible and should leave the output unchanged, draw the self-loop |
| “(ON, OFF) while Cooking” written only on an arc | That sentence is Moore: the command belongs to the state |
| Forgetting power-on behaviour | Start state (and its $\lambda$) is part of the answer |
| Resume cook on doorClose alone | Brief says Start must be pressed again—model that as `Paused --start→ Cooking` |

---

## Cheat sheet

```text
accepter     yes/no at end of input
transducer   output while running
Moore        λ: Q → Γ ; output on state (entry) ; |out| = n+1
Mealy        λ: Q×Σ → Γ ; output on transition (edge) ; |out| = n
equiv.       same power; watch the one-step shift
design       inputs → outputs → Moore/Mealy → states → assumptions
             → arcs → completeness → check paths vs brief
```

---

## Self-check

1. State the domain of $\lambda$ for Moore and for Mealy.  
2. For the echo machines above, what is the Moore output string on input $01$? What is the Mealy output string?  
3. Add a third state `Fault / LAMP_OFF` entered by a `fault` event from either `Off` or `On`, with no exit. Is the machine still Moore?  
4. Convert the desk-lamp Moore machine to a Mealy labelling: what appears on the arc `Off --on--> On`?  
5. Why is “magnetron ON while Cooking” a Moore statement?  
6. In the microwave Moore machine, what is $\lambda(\mathrm{Paused})$? Which arc resumes cooking?

### Answers

**1.** Moore: $\lambda: Q \to \Gamma$. Mealy: $\lambda: Q \times \Sigma \to \Gamma$.

**2.** Moore (start in $Z$): $0\,0\,1$. Mealy: $0\,1$.

**3.** Yes. `Fault` still has a fixed output on the state; `fault` is an input on arcs into it.

**4.** `on / LAMP_ON` (input / output).

**5.** The output is tied to being in the state `Cooking`, not to a particular input edge.

**6.** $\lambda(\mathrm{Paused}) = (\mathrm{OFF},\ \mathrm{ON})$. Resume is `Paused --start→ Cooking` (doorClose alone does not resume).

---

## References

- Linz and Rodger, 7th ed., Appendix A.2–A.4.  
- Moore, E. F. Gedanken-Experiments on Sequential Machines. In Shannon and McCarthy (eds.), *Automata Studies*, Princeton University Press, 1956.  
  Read in [BetterReads](https://www.betterreads.dev/reader?bookId=moore_gedanken_experiments_sequential_machines&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}
- Mealy, G. H. A Method for Synthesizing Sequential Circuits. *Bell System Technical Journal*, 1955.  
  Read in [BetterReads](https://www.betterreads.dev/reader?bookId=mealy_synthesizing_sequential_circuits&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}
- [Part I](/2026/11/06/automata-basics-for-coders-part-1/)–[Part III](/2026/11/09/automata-basics-for-coders-part-3/).  
- Next in this series: [Part V](/2026/11/08/automata-basics-for-coders-part-5/) (Petri nets and reachability).
