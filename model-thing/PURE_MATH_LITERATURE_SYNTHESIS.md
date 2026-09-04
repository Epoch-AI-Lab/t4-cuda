# Pure mathematics literature synthesis: Final structure versus messy exploration

## 1. Executive summary

Mathematical publications conceal their own history. A finished research paper presents definitions, lemmas, theorems, and corollaries in a clean deductive line. Axioms appear first, followed by technical lemmas whose motivations remain obscure until they snap together in the final theorem. This presentation format, codified in the twentieth century by the Nicolas Bourbaki collective, creates a retrospective linear illusion: it suggests that mathematical truths unfold by steady, inevitable deduction from first principles.

Real-time mathematical discovery operates in the opposite direction. Human mathematicians explore chaotic scratchpads. They test small numbers, draw rough diagrams, formulate false conjectures, discover counterexamples, bar pathological cases, and revise definitions retroactively to make tentative proofs hold. The lemmas in a final paper were not discovered before the theorem; they were isolated backward from the wreckage of failed proof attempts.

Current autoregressive foundation models struggle with mathematical reasoning precisely because machine learning systems train on finished, Bourbaki-style outputs. When a model must generate reasoning tokens and a final proof in a single forward pass without structural separation, it attempts to produce both the discovery path and the cleaned presentation simultaneously. This causes early commitments to unverified hypotheses, hallucinated lemmas, and unrecoverable drift.

This synthesis surveys pure mathematics literature outside computer science. It examines how masters of mathematical cognition analyzed invention, proof construction, and communication: Henri Poincaré, Jacques Hadamard, George Pólya, Imre Lakatos, Alexander Grothendieck, William Thurston, Terence Tao, Felix Klein, David Hilbert, and Michael Atiyah. From their insights, we derive concrete architectural specifications for mathematical foundation models:
1. Dual-phase reasoning architectures that isolate exploratory scratchpads from formal proof synthesizers.
2. Tagged reasoning semantics that explicitly separate exploratory search, conjecture formulation, adversarial counterexample hunting, lemma isolation, and formal synthesis.
3. Post-rigorous proof compilation pipelines that translate high-level structural intuition into verified formal proofs.
4. Reinforcement learning objectives that reward both heuristic refutation in the scratchpad and deductive compression in the final output.

---

## 2. Foundational pure mathematics literature: Cognitive mechanics of discovery

The literature on mathematical cognition and epistemology provides an extensive record of how mathematical discovery differs from mathematical presentation. The key thinkers examined below rejected the view that mathematics is purely mechanical deduction.

```
+-----------------------------------------------------------------------------------------------+
|                             EPISTEMOLOGICAL SPECTRUM OF MATHEMATICS                           |
|                                                                                               |
|  Heuristic Discovery & Incubation                  Rigorous Formal Presentation              |
|  (Non-linear, intuitive, visual)                   (Linear, axiomatic, deductive)             |
|                                                                                               |
|  Poincare / Hadamard / Polya / Lakatos    <----->  Hilbert / Bourbaki / Lean Formalization   |
|  - Small numerical trials                          - Definition-Lemma-Theorem chains          |
|  - Monster-barring & lemma incorporation           - Sanitized notation                       |
|  - Unconscious incubation & aesthetic sieves       - Proof checking without human motivation  |
+-----------------------------------------------------------------------------------------------+
```

### 2.1. Henri poincaré: Unconscious incubation and the aesthetic sieve

Henri Poincaré delivered his famous lecture on mathematical invention at the Institut Général Psychologique in Paris in 1908, later published in *Science et Méthode* (1908) and *Science and Hypothesis* (1902). Poincaré attacked the naive mechanical view of mathematics. If mathematics were simply syllogistic logic, anyone with a sound memory and logical processing capacity could invent new mathematics. Yet most people cannot, and even accomplished mathematicians make blunders.

Poincaré asked: what is mathematical invention? It does not consist in making new combinations with mathematical entities already known. Anyone can make an infinite number of combinations, most of which are sterile. To invent is to discern, to select. Invention is the art of choosing the useful combinations among an infinity of useless ones.

Poincaré identified four distinct stages in mathematical discovery:
1. **Conscious preparation**: The mathematician attacks a problem with all available tools, testing ideas, calculating examples, writing scratch work, and reaching an impasse. This phase feels unsuccessful because it produces no clean solution, but it mobilizes the unconscious mind.
2. **Unconscious incubation**: The mathematician steps away from the problem. The conscious mind turns to unrelated matters or rests during sleep. Subconscious thought continues recombining ideas.
3. **Illumination (sudden insight)**: A complete, fully formed conviction of truth flashes into consciousness. Poincaré gave the famous autobiographical account of his discovery of Fuchsian functions. After days of fruitless labor at his desk working on automorphic functions, he drank black coffee and could not sleep; ideas swarmed through his mind. Later, during a geological excursion organized by the École des Mines at Coutances, at the exact instant he put his foot on the step of the omnibus, the idea arrived without any conscious preparation: the transformations he had used to define Fuchsian functions were identical with those of non-Euclidean geometry. He continued his conversation without pause and verified the calculation at his leisure upon returning home.
4. **Conscious verification**: The mathematician subjects the flash of illumination to strict logical deduction. Calculations are written out, lemmas verified, and the argument formatted for publication.

Poincaré emphasized that the selection mechanism of the unconscious mind is not logical; it is aesthetic. The unconscious mind possesses what he called mathematical sensibility. The ideas that pass into consciousness are those that produce harmonious, symmetrical, and elegant relations, because these aesthetic forms correspond to deep structural economy.

```
Poincare Discovery Pipeline:
Conscious Preparation  -->  Unconscious Incubation  -->  Illumination (Aesthetic Sieve)  -->  Conscious Verification
(Chaotic search, brute      (Background stochastic       (Harmonious structural match          (Deductive proof,
 scratch work, dead ends)    recombination of ideas)      crosses into conscious awareness)     symbolic checking)
```

For foundation models, Poincaré provides two insights:
- Brute combinatorial search in token space fails because the search space explodes exponentially.
- An effective model requires an internal scoring function, an analog of Poincaré's aesthetic sieve, that filters candidate proof directions long before writing complete formal derivations.

### 2.2. Jacques Hadamard: Mental imagery and non-verbal contemplation

Jacques Hadamard formalized and extended Poincaré's framework in *The Psychology of Invention in the Mathematical Field* (1945). Hadamard surveyed prominent mathematicians and physicists, including Albert Einstein, George Birkhoff, and Norbert Wiener, to determine what kind of mental representations mathematicians use when thinking.

The most striking finding of Hadamard's inquiry was that creative mathematical thought is almost entirely non-verbal and non-symbolic. Words and algebraic symbols do not appear in the discovery phase; they appear only in the reporting phase.

Hadamard quoted Albert Einstein's response to his questionnaire:
"Words or the language, as they are written or spoken, do not seem to play any role in my mechanism of thought. The psychical entities which seem to serve as elements in thought are certain signs and more or less clear images, which can be 'voluntarily' reproduced and combined... Conventional words or other signs have to be sought for laboriously only in a secondary stage, when the mentioned associative play is sufficiently established and can be reproduced at will."

Hadamard documented his own mental state during research: he thought in vague geometric spots, kinetic movements, and spatial arrangements of colors and weights. He stated plainly:
"I insist that words are totally absent from my mind when I really think... not only words, but also algebraic signs are absent, which would be for me an even heavier impediment."

Hadamard concluded that formal algebraic notation and rigorous language are linear constraints that slow down the mind's ability to view multiple relations at once. The human brain uses a high-dimensional, compressed, non-verbal representation during the preparation and incubation phases. It translates these thoughts into symbols only during the verification and reporting phases.

```
Hadamard Cognition Model:
High-Dimensional Spatial/Kinetic Imagery  ==================>  Linear Symbolic Language
(Parallel, non-verbal, holistic intuition)   Translation Gap    (Sequential, algebraic, Bourbaki text)
```

This discovery has direct implications for language models. Transformer models are forced to think in the exact same 1D token stream in which they output their final results. This forces the model to perform high-dimensional structural search using linear text tokens, which creates a cognitive bottleneck that human mathematicians actively avoid.

### 2.3. George pólya: Heuristics and plausible reasoning

George Pólya provided the most comprehensive catalog of explicit mathematical heuristics in twentieth-century mathematics across three masterworks: *How to Solve It* (1945), and the two volumes of *Mathematics and Plausible Reasoning* (*Induction and Analogy in Mathematics*, 1954, and *Patterns of Plausible Inference*, 1954).

Pólya drew a sharp boundary between two types of reasoning:
- **Demonstrative reasoning**: Rigid, deductive, safe, final. Governed by logic: if $A \implies B$ and $A$ is true, then $B$ is true. This is the logic of mathematical proof.
- **Plausible reasoning**: Flexible, inductive, hazardous, exploratory. Governed by heuristics: if $A \implies B$ and $B$ is observed to be true, $A$ becomes more credible. This is the logic of mathematical discovery.

```
+-----------------------------------------------------------------------------------------------+
|                               POLYA'S HEURISTIC TAXONOMY                                      |
+--------------------------+--------------------------------------------------------------------+
| Heuristic Operation      | Concrete Mathematical Action                                       |
+--------------------------+--------------------------------------------------------------------+
| Specialization           | Test the conjecture on extreme cases: n = 0, n = 1, infinity,       |
|                          | degenerate triangles, empty sets, identity matrices.               |
+--------------------------+--------------------------------------------------------------------+
| Generalization           | Replace a specific constant with a variable; embed the problem in  |
|                          | a broader category where the core invariant becomes visible.       |
+--------------------------+--------------------------------------------------------------------+
| Analogy                  | Find a simpler problem in another field that shares the same       |
|                          | structural geometry (e.g. plane geometry vs. solid geometry).      |
+--------------------------+--------------------------------------------------------------------+
| Auxiliary Problems       | If you cannot solve problem P, find an auxiliary problem P' whose  |
|                          | solution can be imported as a stepping stone.                      |
+--------------------------+--------------------------------------------------------------------+
| Decomposing & Recombining| Break the problem into atomic sub-goals, solve each in isolation,  |
|                          | and rearrange the elements to find new structural relations.       |
+--------------------------+--------------------------------------------------------------------+
| Working Backwards        | Assume the desired conclusion holds; trace backward along the      |
| (Analysis vs. Synthesis) | implications to discover what premises are sufficient to reach it. |
+--------------------------+--------------------------------------------------------------------+
```

Pólya analyzed the ancient Greek distinction between Analysis and Synthesis:
- **Analysis (working backwards)**: Start from the desired target $T$. Ask: from what premise $S_1$ could $T$ be derived? Then from what $S_2$ could $S_1$ be derived? Continue until reaching an established axiom or known theorem $K$.
- **Synthesis (working forwards)**: Reverse the chain. Start from $K$, deduce $S_2$, deduce $S_1$, and finally deduce $T$.

Pólya pointed out that textbook proofs show only the Synthesis. The reader sees a miraculous sequence of steps starting from an obscure trick and terminating in the answer. The real work was the Analysis, which explored the problem in reverse. When textbooks hide the Analysis, they prevent students from understanding how the proof was found.

Pólya demonstrated inductive verification with numerical examples. Before attempting to prove that the sum of the first $n$ cubes equals the square of the sum of the first $n$ integers:
$$1^3 + 2^3 + \dots + n^3 = (1 + 2 + \dots + n)^2$$
the mathematician calculates small values:
- For $n=1$: $1 = 1^2$.
- For $n=2$: $1 + 8 = 9 = 3^2 = (1+2)^2$.
- For $n=3$: $1 + 8 + 27 = 36 = 6^2 = (1+2+3)^2$.
- For $n=4$: $36 + 64 = 100 = 10^2 = (1+2+3+4)^2$.

Only after accumulating overwhelming empirical evidence from small cases does the mathematician commit energy to constructing an inductive or geometric proof.

### 2.4. Imre Lakatos: Proofs, refutations, and proof-generated concepts

Imre Lakatos produced what remains the deepest philosophical analysis of mathematical dialectic in *Proofs and Refutations: The Logic of Mathematical Discovery* (1976). Setting his book as a dialogue in a classroom between a teacher and students, Lakatos analyzed the historical development of Euler's formula for polyhedra:
$$V - E + F = 2$$
where $V$ is vertices, $E$ is edges, and $F$ is faces.

Lakatos demonstrated that mathematics does not grow through the cumulative addition of indubitable theorems proved from fixed axioms. Instead, mathematics develops through a dialectical process of proof-proposals, counterexamples, and concept revisions.

```
Lakatos Dialectical Cycle:
Primitive Conjecture  -->  Proposed Proof  -->  Global/Local Counterexample
         ^                                                |
         |                                                v
Refined Conjecture    <--  Lemma Incorporation  <--  Monster-Barring or Monster-Adjusting
```

Lakatos categorized the responses of mathematicians when confronted with counterexamples:

1. **Primitive Conjecture**: The initial guess based on simple examples (cube: $8 - 12 + 6 = 2$; tetrahedron: $4 - 6 + 4 = 2$).
2. **Local Counterexample**: An example that refutes a specific lemma inside the proposed proof without refuting the main conjecture.
3. **Global Counterexample**: An example that refutes the main conjecture itself:
   - *The Kepler small stellated dodecahedron*: Faces intersect, formula fails ($12 - 30 + 12 = -6$).
   - *The picture-frame (toroidal polyhedron)*: A cube with a square hole through the middle ($16 - 32 + 16 = 0$).
   - *The cylinder or nested hollow cubes*: Polyhedron with cavities inside ($16 - 24 + 12 = 4$).
4. **Monster-Barring**: The defensive reaction. Rather than abandoning the conjecture or re-examining the proof, the mathematician redefines "polyhedron" to exclude the awkward case. A hollow cube is dismissed as not a real polyhedron, but a monster.
5. **Monster-Adjusting**: Reinterpreting the boundaries or vertices of the monster so that it artificially satisfies the formula.
6. **Lemma-Incorporation**: The true engine of mathematical progress. When a counterexample refutes a proof, the mathematician examines where the proof broke down. In Cauchy's proposed 1813 proof of Euler's formula, Cauchy removed one face and flattened the remaining network of polygons into the plane. For a hollow cube or picture-frame, this flattening step is impossible. The counterexample reveals a hidden lemma that had been assumed without being stated: that the polyhedron can be mapped homeomorphically to a sphere (simple connectivity and orientability).
   The mathematician does not discard the conjecture. Instead, they incorporate the hidden lemma into the theorem statement:
   "For every polyhedron that is homeomorphically equivalent to a sphere, $V - E + F = 2$."
7. **Proof-Generated Concepts**: As a result of lemma-incorporation, new mathematical concepts are born. Concepts like simple connectivity, topological genus, and orientability were not invented in the abstract; they were forged as defensive lemmas to rescue Euler's formula from specific monsters.

Lakatos established that mathematical rigor is not a static property of a proof; it is an evolving standard shaped by historical challenges and counterexamples.

### 2.5. Alexander Grothendieck: The rising sea and conceptual dissolution

Alexander Grothendieck transformed algebraic geometry during the 1950s and 1960s through the Éléments de géométrie algébrique (EGA) and Séminaire de géométrie algébrique (SGA). In his autobiographical reflection *Récoltes et Semailles* (1986), Grothendieck articulated a distinct philosophy of mathematical discovery, contrasting two fundamentally different ways of solving mathematical problems:

1. **The Hammer and Chisel (Le marteau et le burin)**: The mathematician attacks a hard problem head-on. The problem is viewed as a nut to crack. The mathematician strikes it repeatedly with sharp tools, technical ingenuity, and heavy calculations until the shell breaks open.
2. **The Rising Sea (La mer qui monte)**: The mathematician refuses to attack the specific problem directly. Instead, they immerse the problem in an expansive, general conceptual ocean. The conceptual level rises slowly, effortlessly, and quietly. Over time, the hard shell of the nut softens, becomes permeable, and dissolves on its own.

Grothendieck described the rising sea in *Récoltes et Semailles*:
"The unknown thing to be known used to send me back to some hard, resistant nut to be cracked open. But the sea advances insensibly and quietly, nothing seems to stir, but it surrounds the resistant substance gradually, and after a while it completely submerges it."

```
Two Paradigms of Problem Solving:
1. Hammer and Chisel (Hard Analysis / Combinatorics):
   Problem [Nut] <--- Strike with force, inequality tricks, specialized bounds

2. Rising Sea (Grothendieckian Conceptual Geometry):
   ~~~~~~~~~~~~ (Concepts rise: schemes, topoi, motives) ~~~~~~~~~~~~
   ~~~~~~~~~~~~ Problem dissolves into a trivial instance of a universal property ~~~~~~~~~~~~
```

Grothendieck’s approach relied on three principles:
- **Universal Properties**: An object is never studied by dissecting its internal coordinates. An object is defined solely by how it interacts with all other objects in its category. A tensor product, fiber product, or scheme is characterized entirely by its universal mapping property.
- **Functorial Perspective**: Rather than studying a single fixed space $X$, Grothendieck studied the functor of points $h_X: Y \mapsto \text{Hom}(Y, X)$. Geometry becomes the study of variations over variable base schemes $S$.
- **The Yoga of Motives**: Grothendieck sought the universal cohomology theory. Recognizing that disparate cohomology theories (Betti, de Rham, étale, crystalline) yield identical numerical invariants on algebraic varieties, he conjectured the existence of motives: the common underlying musical score played by different instruments.

Grothendieck showed that when the right conceptual home is built, proofs shrink. What once required hundreds of pages of ad-hoc calculations becomes a tautological consequence of commutative diagrams.

For machine learning, Grothendieck demonstrates that mathematical power does not come from searching through permutations of algebraic identities. It comes from identifying the right abstraction level where the problem has a trivial proof.

### 2.6. William Thurston: Mathematical understanding versus formal logic strings

William Thurston, Fields Medalist in geometric topology, published *On Proof and Progress in Mathematics* (1994, Bulletin of the AMS) in response to Arthur Jaffe and Frank Quinn’s proposal to formalize standards for speculative mathematics. Thurston mounted a defense of human mathematical cognition against reduction to formal logic.

Thurston posed a central question: What is the true product of mathematics?
The conventional answer is: mathematics produces theorems and formal proofs.
Thurston answered: No. Mathematics produces human understanding. Theorems and proofs are merely vehicles to advance understanding.

Thurston explained why human mathematicians do not think in formal logic strings:
"We have a massive mental capacity for visual and spatial thinking, for kinesthetic intuition, for linguistic association, and for social navigation. Formal logic is a tiny, recent cultural invention that runs on a very narrow cognitive channel."

```
+-----------------------------------------------------------------------------------------------+
|                        THURSTON'S MENTAL CHANNELS IN MATHEMATICS                              |
+--------------------------+--------------------------------------------------------------------+
| Mental Channel           | Cognitive Operation in Pure Mathematics                           |
+--------------------------+--------------------------------------------------------------------+
| Visual & Spatial         | Mental rotation of manifolds, foliation flows, hyperbolic geometry |
| Kinesthetic / Muscle     | Feeling tension in a minimal surface, stretching rubber sheets     |
| Linguistic & Metaphoric  | Transferring terms across domains ("gluing", "surgery", "blow-up") |
| Social & Communicative   | Evaluating peer reliability, shared context, colloquial debate     |
| Logical & Algebraic      | Narrow symbolic verification, formal manipulation (terminal check) |
+--------------------------+--------------------------------------------------------------------+
```

Thurston observed that when two mathematicians converse at a blackboard, they do not recite formal definitions or epsilon-delta bounds. They wave their hands, sketch distorted curves, use physical analogies, and say: "Think of this as a torus where you pinch the neck until it snaps." This transmission transfers the underlying mental model directly. Once the listener internalizes the mental model, they can supply the rigorous formal steps themselves at their leisure.

Thurston warned against confusing the formal artifact with the mathematical truth:
"The formal logic definition of proof does not describe what happens in real life... A formal proof of a deep theorem would be millions of lines long and completely unreadable by any human being. When we read a proof, we verify that the author’s mental model matches our own, and that no fatal obstacles block the path."

Thurston's distinction clarifies why automated theorem provers and pure formal assistants like Lean or Coq feel unnatural to working mathematicians. They operate strictly at the terminal level of formal logic strings, ignoring the rich multimodal representations that guide mathematical thought.

### 2.7. Terence Tao: The three stages of mathematical cognition

Terence Tao, Fields Medalist across harmonic analysis, partial differential equations, combinatorics, and number theory, analyzed mathematical cognitive development in *Solving Mathematical Problems* (2006), his books *Structure and Randomness* (2008) and *Compactness and Contradiction* (2013), and his foundational essay "There's more to mathematics than rigour and proofs" (2008).

Tao divided mathematical learning into three distinct stages:

```
Tao's Three Stages of Mathematical Cognition:
1. Pre-Rigorous  =================>  2. Rigorous  =================>  3. Post-Rigorous
(Intuitive, visual,                 (Formal definitions,              (Re-integrated intuition,
 calculations on small examples,     epsilon-delta mechanics,          structural vision, guided by
 heuristic leaps, error-prone)       fear of intuition, pedantry)      rigor but thinking at scale)
```

1. **The Pre-Rigorous Stage**: The beginner works intuitively. They treat infinitesimals as small numbers, interchange sums and integrals without checking dominated convergence, draw pictures, and manipulate formulas mechanically. The work is fertile and intuitive, but fragile. The student cannot resolve paradoxes (such as $1 - 1 + 1 - 1 + \dots = 1/2$) and cannot prove why theorems hold.
2. **The Rigorous Stage**: The student is introduced to formal definitions, epsilon-delta proofs, and axiomatic foundations. Intuition is temporarily banned as untrustworthy. The student learns to verify every step, check uniform convergence, verify compact support, and adhere strictly to logical rules. This stage is necessary, but it introduces pedantry: the student becomes so consumed by tracking indices and epsilon-deltas that they lose the global picture. They cannot formulate new conjectures because their creative intuition has been suppressed.
3. **The Post-Rigorous Stage**: The mature mathematician re-integrates high-level intuition, but an intuition that has been refined, calibrated, and disciplined by the rigorous stage. The post-rigorous mathematician does not calculate epsilon-deltas in their head during discovery. They think in large structural blocks: "This operator is essentially a Calderón-Zygmund singular integral, so it must be bounded on $L^p$ away from the endpoints." They know that if challenged, they could reconstruct the formal epsilon-delta argument in an afternoon, but they do not need to do so to discover the proof.

Tao noted that non-mathematicians mistake the Rigorous stage for the peak of mathematics. In reality, the Rigorous stage is merely an intermediate apprentice phase. The goal of mathematical training is to reach the Post-Rigorous stage, where one moves freely between broad heuristic intuition and formal verification.

Tao also developed the fundamental dichotomy between **Structure and Randomness**. In problems across number theory, combinatorics, and PDE analysis, mathematical objects decompose into:
- A structured component (low-dimensional, deterministic, periodic, algebraic).
- A pseudorandom component (high-dimensional, uniform, uncorrelated, noise-like).

In the Green-Tao theorem (2004), which proved that the primes contain arbitrarily long arithmetic progressions, this exact dichotomy formed the architecture of the proof. The primes are modeled as a dense subset inside a pseudorandom majorant (a modified Selberg sieve), and Gowers uniformity norms measure pseudorandomness while Furstenberg-style ergodic theory controls the structured component.

For foundation models, Tao's framework shows that an AI reasoning system must not remain stuck at the Rigorous stage, endlessly outputting verbose low-level formalisms. It must operate at the Post-Rigorous level, executing high-level strategic reasoning, while retaining a compiler that can lower post-rigorous plans into rigorous formal proofs.

### 2.8. Additional masters: Klein, Hilbert, and Atiyah

#### Felix Klein (1849-1925): Spatial intuition and invariant groups
In his *Erlangen Program* (1872) and *Elementary Mathematics from an Advanced Standpoint* (1908), Felix Klein united geometry not through axiomatic systems, but through group actions. A geometry is defined by a space and a group of transformations acting on it; geometric properties are those invariants that remain unchanged under the group. Klein emphasized visual and spatial intuition as an active engine of mathematical thought:
"The investigator in mathematics, as in every other field, does not work in an exclusively logical way; with him, especially, imagination is the active element... Intuition must always precede logical analysis."

#### David Hilbert (1862-1943): The problem-driven engine
Hilbert is often associated with the formalist program and the axiomatic foundation of mathematics (*Grundlagen der Geometrie*, 1899). Yet Hilbert’s address to the International Congress of Mathematicians in Paris in 1900 presented twenty-three concrete problems. Hilbert argued that a vital branch of mathematics is sustained not by pure axiomatic derivation, but by difficult, unsolved problems:
"As long as a branch of science offers an abundance of problems, so long is it alive; a lack of problems foreshadows extinction or the cessation of independent development."
Hilbert demonstrated that axiomatics serves to tidy up and secure territory after problems have forced new concepts into existence, not to generate mathematics ex nihilo.

#### Michael Atiyah (1929-2019): Geometry versus algebra
Michael Atiyah, Fields Medalist and Abel Prize laureate, defended geometric intuition against pure algebraic formalism in his essay *Mathematics in the 20th Century* (2001) and *Advice to a Young Mathematician*. Atiyah framed geometry and algebra as two complementary poles:
"Algebra is the offer made by the devil to the mathematician. The devil says: 'I will give you this powerful machine, it will answer any question you like, all you need to do is give me your soul: give up geometry and you will have this marvellous machine.'... But geometry gives you the intuition, the global vision. When you lose geometry, you are operating blindly in algebra."
Atiyah divided mathematicians into two psychological types:
- **Visionaries (Builders)**: Mathematicians who survey the terrain from high above, identify overarching structures, and chart roads across uncharted territories (e.g. Grothendieck, Hermann Weyl).
- **Repairers (Problem Solvers)**: Mathematicians who descend into difficult, rock-strewn passes, solve specific resistant bottlenecks, and bring deep technical calculations to completion (e.g. John Nash, Paul Erdős).

---

## 3. The great divide: Canonical final proof structure versus real-time discovery

### 3.1. The canonical architecture: The Bourbaki sanitization

Modern mathematics presents its results in an architectural style codified by the Nicolas Bourbaki collective. Founded in 1934 by André Weil, Henri Cartan, Claude Chevalley, Jean Delsarte, and Jean Dieudonné, Bourbaki sought to write a completely self-contained, rigorous treatise covering all of mathematics from set theory forward: the *Éléments de mathématique*.

Bourbaki imposed a strict structural discipline on mathematical writing:
- **Strict Deductive Linearity**: Every concept must be defined solely in terms of previously defined concepts. Every theorem must follow strictly from axioms or previously proved theorems.
- **Extreme Generality and Abstraction**: Problems are stated in their most abstract topological, algebraic, or categorical setting. Concrete real-number examples are treated as mere coordinates of abstract spaces.
- **Exclusion of Visual and Geometric Heuristics**: Jean Dieudonné famously declared "À bas Euclide! Mort aux triangles!" (Down with Euclid! Death to triangles!). Bourbaki's volumes contain zero diagrams, zero drawings, and zero geometric sketches. Spatial intuition was regarded as a source of psychological error that must be eradicated from pure mathematics.
- **Sanitization of Discovery History**: Failed proofs, intermediate calculations, arithmetic checks, and historical blunders are excised. A Bourbaki theorem appears fully formed, detached from the human struggle that created it.

```
Bourbaki Presentation Cascade:
[Axioms / Definitions] ---> [Technical Lemma 1] ---> [Technical Lemma 2] ---> [Main Theorem] ---> [Corollaries]
        |                            |                         |                    |
        +----------------------------+-------------------------+--------------------+
          No motivation provided; lemmas exist purely to make the theorem proof short.
```

In a Bourbaki-style proof, the reader encounters technical lemmas whose purpose is completely opaque. Lemma 1 proves an odd algebraic identity involving four indices. Lemma 2 establishes a bound on an obscure functional. Only at the very end of the paper does the author combine Lemma 1 and Lemma 2 in two lines to prove the Main Theorem.

This creates the **retrospective linear illusion**. The presentation implies that the author started at the axioms, deduced Lemma 1, deduced Lemma 2, and arrived smoothly at the Theorem. In reality, the author did the exact opposite: they wanted to prove the Theorem, attempted a direct attack, discovered that it failed because of a gap, constructed a counterexample, isolated the failure into an unstated hypothesis, formulated Lemma 2 backward from the failure, adjusted Lemma 1 to supply the hypothesis for Lemma 2, and then cleaned up all the debris before publication.

### 3.2. Real-time discovery: Chaotic scratchpads and counterexample hunting

The actual working environment of a research mathematician is messy, non-linear, and filled with discarded debris.

```
Real-Time Discovery Reality:
+-------------------------------------------------------------------------------------------------+
|                                 THE MATHEMATICIAN'S SCRATCHPAD                                  |
|                                                                                                 |
|   "Try n=1: trivial.                                    [Rough hand-drawn diagram:             |
|    Try n=2: works by hand (8 - 12 + 6 = 2).              a cube with a twisted puncture.        |
|    Try n=3: 18 - 24 + 8 = 2.                             Crossed out with big 'X']              |
|    Wait, what about the hollow cube?                                                            |
|    Faces = 12, Vertices = 16, Edges = 24.               'NO! This blows up when boundary        |
|    16 - 24 + 12 = 4 != 2!                               is non-orientable. Need orientability!  |
|    DAMN. The formula fails for hollow spaces.            Add as explicit hypothesis.'           |
|    Can we retract the inner cube?                                                               |
|    No, boundary has two components.                      'Check Cauchy's 1813 paper:            |
|    Lemma: assume connected boundary.                     he assumed polygon network can flatten |
|    Wait, Mobius strip has 1 face, 3 edges, 2 vertices.   without self-intersection. False!'     |
|    2 - 3 + 1 = 0 != 2.                                                                          |
|    Now test on Klein bottle..."                          'Re-index: set genus g.                |
|                                                           V - E + F = 2 - 2g. Yes!'             |
+-------------------------------------------------------------------------------------------------+
```

Real-time discovery displays four defining characteristics:
1. **Numerical and Combinatorial Probing**: Mathematicians calculate small cases by hand or with quick computer scripts. Before proving an identity for all polynomials, they plug in $x = 0, 1, -1, 2$. Before analyzing a Lie group, they calculate the result explicitly for $SU(2)$ or $SL(2, \mathbb{R})$.
2. **Adversarial Counterexample Hunting**: A mathematician spends as much time trying to refute their own conjecture as they do trying to prove it. They search for edge cases, extreme geometries, degenerate configurations, and pathological functions (e.g. the Weierstrass nowhere-differentiable function or the Cantor ternary set). If an adversary cannot break the conjecture after hours of intense attack, the mathematician gains confidence that a proof exists.
3. **Wandering Heuristics and Analogies**: Mathematicians use loose analogies between completely unrelated fields: "This heat equation behaves like water flowing down a drain; let's treat the dissipation term as friction."
4. **Discarded Lines and Backward Deduction**: Mathematicians write pages of scratch work that lead to complete dead ends. When an approach hits a wall, the mathematician abandons the scratchpad, steps back, and starts again from a different angle.

### 3.3. The epistemological inversion

The relationship between discovery and presentation is an inversion. Every feature that characterizes discovery is systematically erased in the published text.

```
+-----------------------------------------------------------------------------------------------+
|                       THE EPISTEMOLOGICAL INVERSION MATRIX                                    |
+-----------------------+-------------------------------+---------------------------------------+
| Cognitive Dimension   | Real-Time Discovery Reality   | Canonical Presentation (Bourbaki)     |
+-----------------------+-------------------------------+---------------------------------------+
| Direction of Thought  | Backward (Analysis: from goal | Forward (Synthesis: from axioms to    |
|                       | to required premises)         | technical lemmas to theorem)          |
+-----------------------+-------------------------------+---------------------------------------+
| Representation Medium | Multimodal (diagrams, muscle  | Strictly 1D linear symbolic text and  |
|                       | tension, spatial metaphors)   | formalized algebra                    |
+-----------------------+-------------------------------+---------------------------------------+
| Handling of Errors    | Essential fuel (counterexamples| Completely sanitized; zero record of  |
|                       | force new definitions)        | false starts or failed paths          |
+-----------------------+-------------------------------+---------------------------------------+
| Role of Definitions   | Late stage (definitions are   | Early stage (definitions appear on    |
|                       | proof-generated boundaries)   | page 1 as if born from thin air)      |
+-----------------------+-------------------------------+---------------------------------------+
| Role of Lemmas        | Defensive patches created to  | Polished stepping stones whose real   |
|                       | block specific counterexamples| purpose is hidden until the end       |
+-----------------------+-------------------------------+---------------------------------------+
| Computational Style   | Concrete, small n arithmetic, | Universal, coordinate-free, maximum   |
|                       | degenerate edge cases         | structural abstraction                |
+-----------------------+-------------------------------+---------------------------------------+
| Epistemic Goal        | Understanding: building a     | Verification: establishing deductive  |
|                       | robust mental model           | validity beyond formal doubt          |
+-----------------------+-------------------------------+---------------------------------------+
| Communication Mode    | Conversational, colloquial,   | Sparse, austere, objective, third-    |
|                       | hand-wavy, heuristic          | person passive voice                  |
+-----------------------+-------------------------------+---------------------------------------+
```

When artificial intelligence systems attempt to reason mathematically by imitating only the right-hand column (Bourbaki presentation), they fail. They try to generate finished theorems without going through the messy dialectic of the left-hand column.

---

## 4. Concrete algorithmic analogies for foundation models

### 4.1. Why autoregressive single-pass generation fails

Standard transformer foundation models are trained using causal language modeling objectives:
$$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$
When a model is prompted with a difficult mathematical problem, it generates tokens autoregressively from left to right. This architecture imposes severe failure modes when applied to mathematics:

1. **The Irreversibility Trap**: Autoregressive generation cannot backtrack natively. Once token $x_t$ is emitted, it enters the static conditioning prefix $x_{\le t}$ for all subsequent tokens. If the model emits an incorrect algebraic step, a flawed lemma, or a dead-end direction, it cannot erase those tokens. Instead, the model exhibits strong confirmation bias: subsequent tokens rationalize the earlier mistake, leading to catastrophic logical collapse.
2. **The Bourbaki Training Contamination**: Supervised fine-tuning (SFT) datasets consist almost exclusively of clean, polished solutions. Textbooks, competition answer keys, and synthetic chain-of-thought traces present linear proofs: Step 1, Step 2, Step 3, Final Answer. Because the model rarely observes failed scratch work, false starts, or counterexample refutations in its training data, it learns that a valid reasoning trace must proceed linearly forward. When it hits a point of uncertainty, instead of stopping to test counterexamples or back up, it hallucinates a bridging step to maintain the illusion of forward momentum.
3. **Conflation of Discovery Tokens and Verification Tokens**: In human mathematics, the tokens written on a scratchpad have a completely different status from the tokens printed in a journal. On a scratchpad, writing "$x = 4$" may simply mean "let us test whether $x=4$ works." In a journal, writing "$x = 4$" asserts a mathematical fact. When an LLM outputs both in the same undifferentiated token stream, the attention mechanism treats speculative exploration tokens as asserted truths, causing downstream deductive errors.

```
Failure Mode of Single-Pass Autoregressive Reasoning:
Prompt ---> [Token 1] ---> [Token 2] ---> [Flawed Guess] ---> [Hallucinated Bridge] ---> [Confidently Wrong Proof]
                ^                                |
                +---- Cannot backtrack or test --+
```

### 4.2. Dual-phase reasoning architecture

To resolve this failure mode, mathematical reasoning systems must structurally decouple messy exploration from disciplined proof compilation.

```
+-----------------------------------------------------------------------------------------------+
|                            DUAL-PHASE REASONING ARCHITECTURE                                  |
|                                                                                               |
|  [ PROBLEM INPUT ]                                                                            |
|         |                                                                                     |
|         v                                                                                     |
|  +-----------------------------------------------------------------------------------------+  |
|  | PHASE 1: STOCHASTIC EXPLORATION ENGINE (Poincare / Polya / Lakatos Scratchpad)          |  |
|  | - High sampling temperature (e.g. T = 0.7 - 1.0)                                        |  |
|  | - Multi-branch parallel rollouts (Monte Carlo Tree / Graph Search)                     |  |
|  | - Small numerical evaluations (Python REPL execution)                                   |  |
|  | - Adversarial counterexample generation                                                 |  |
|  | - Discarded branches pruned without polluting the final context                         |  |
|  +-----------------------------------------------------------------------------------------+  |
|         |                                                                                     |
|         | Surviving Lemma Graph & Invariants (Distilled Proof Strategy)                       |
|         v                                                                                     |
|  +-----------------------------------------------------------------------------------------+  |
|  | PHASE 2: DETERMINISTIC PROOF COMPILER (Bourbaki / Hilbert Synthesis Engine)            |  |
|  | - Low sampling temperature (e.g. T = 0.0 - 0.2) or grammar-constrained decoding        |  |
|  | - Single forward deductive chain: Definition -> Lemma -> Theorem -> Q.E.D.             |  |
|  | - Linear proof compaction: removes all exploratory debris and dead-end tokens            |  |
|  | - Formal verification target: Lean 4 / Isabelle / Coq tactic stream                     |  |
|  +-----------------------------------------------------------------------------------------+  |
|         |                                                                                     |
|         v                                                                                     |
|  [ VERIFIED CANONICAL PROOF ]                                                                 |
+-----------------------------------------------------------------------------------------------+
```

- **Phase 1 (The Exploration Scratchpad)**: Operates under high temperature or branching search. Its purpose is heuristic discovery: finding invariants, identifying analogies, testing small cases, and discovering counterexamples. It is permitted to be verbose, chaotic, and non-linear.
- **The Distillation Boundary**: The exploration trace is passed through a distillation filter. Failed branches, dead ends, and arithmetic scrapings are discarded. Only the verified sequence of lemmas, the governing invariants, and the critical counterexample-bounded conditions are retained.
- **Phase 2 (The Bourbaki Compiler)**: A secondary model pass (or an explicitly conditioned generation pass) takes the distilled lemma graph and compiles it into a clean, minimal Bourbaki presentation. The compiler works forward from the required definitions to the final conclusion, enforcing strict deductive rigor.

### 4.3. Tagged scratchpad structures: Explicit semantic control

Rather than relying on unformatted text where the model wanders between exploration and deduction, models should be trained with explicit structural tags that demarcate cognitive modes:

```
+-----------------------------------------------------------------------------------------------+
|                             TAGGED REASONING SCRATCHPAD SPECIFICATION                         |
+-------------------------+---------------------------------------------------------------------+
| Semantic Tag            | Operational Function & Expected Content                             |
+-------------------------+---------------------------------------------------------------------+
| <explore>               | Free-form heuristic wandering. Analogies, dimensional analysis,     |
|                         | identifying symmetries, guessing the functional form of solutions.  |
+-------------------------+---------------------------------------------------------------------+
| <conjecture>            | Precise statement of a candidate hypothesis or intermediate lemma.   |
|                         | Must be phrased in falsifiable terms.                                |
+-------------------------+---------------------------------------------------------------------+
| <counterexample_test>   | Adversarial verification pass. Tests the conjecture against small   |
|                         | integers (n=0, 1, 2, -1), degenerate cases, empty sets, parity,     |
|                         | extreme bounds, or executes Python code to search for refutations. |
+-------------------------+---------------------------------------------------------------------+
| <refutation_analysis>   | Triggered if counterexample is found. Analyzes where the intuition  |
|                         | failed; identifies the hidden assumption (Lakatosian lemma analysis)|
+-------------------------+---------------------------------------------------------------------+
| <lemma_isolate>         | Extracts the surviving, purified mathematical claim with explicit   |
|                         | hypotheses that bar all discovered monsters.                        |
+-------------------------+---------------------------------------------------------------------+
| <formal_synthesis>      | Bourbaki compilation phase. Emits the canonical Definition-Lemma-    |
|                         | Theorem-Proof sequence. Excludes all scratchpad commentary.         |
+-------------------------+---------------------------------------------------------------------+
```

#### Concrete example trace using tagged semantics

Below is an explicit demonstration of how a foundation model executes the tagged pipeline on a combinatorial geometry problem:

```xml
<explore>
Problem: Determine the maximum number of regions R(n) formed by connecting n points on a circle with straight lines, assuming no three lines intersect at a single point inside the circle.
Let us test small values of n:
For n = 1: 1 region.
For n = 2: 2 regions (one line divides the circle).
For n = 3: 4 regions (a triangle inside gives 1 center + 3 borders = 4).
For n = 4: 8 regions (drawing all diagonals gives 8 regions).
For n = 5: 16 regions (drawing all chords gives 16 regions).
Sequence so far: 1, 2, 4, 8, 16.
</explore>

<conjecture id="conj_1">
The number of regions is R(n) = 2^(n-1) for all n >= 1.
</conjecture>

<counterexample_test target="conj_1">
Let us test n = 6.
If R(6) = 2^(6-1) = 32, the conjecture gains strong support.
Let us compute by Euler's planar graph formula: V - E + F = 1 (excluding the outer region).
Vertices:
- Circle boundary points: n = 6.
- Interior intersection points: each pair of intersecting chords corresponds to 4 circle points.
  Interior V_int = C(6, 4) = 15.
  Total V = 6 + 15 = 21.
Edges:
- Circle arcs: 6.
- Chords broken into segments:
  Each chord from point i to point j has internal crossings.
  Total chord segments = C(n, 2) + 2 * C(n, 4).
  For n = 6: C(6, 2) = 15 chords.
  Internal vertices = 15. Each internal vertex splits 2 lines, adding 2 to edge count.
  Total E = 6 + 15 + 2 * 15 = 51.
Faces (regions inside circle):
F = E - V + 1 = 51 - 21 + 1 = 31.
Wait: 31 != 32!
The sequence for n = 1, 2, 3, 4, 5, 6 is 1, 2, 4, 8, 16, 31.
The conjecture R(n) = 2^(n-1) is REFUTED at n = 6!
</counterexample_test>

<refutation_analysis target="conj_1">
Why did the exponential pattern fail?
2^(n-1) was a false induction caused by small-number coincidence:
C(n, 0) + C(n, 2) + C(n, 4) matches 2^(n-1) for n <= 5 because C(n, 4) is 0 for n < 4, and for n=5, C(5,0)+C(5,2)+C(5,4) = 1 + 10 + 5 = 16 = 2^4.
For n=6, C(6,0) + C(6,2) + C(6,4) = 1 + 15 + 15 = 31, whereas 2^5 = 32.
The fourth-degree polynomial sum diverges from the exponential function at n=6.
The true invariant is determined by the combinatorial topology of chord crossings:
R(n) = C(n, 0) + C(n, 2) + C(n, 4) = 1 + n(n-1)/2 + n(n-1)(n-2)(n-3)/24.
</refutation_analysis>

<lemma_isolate id="lem_1">
Lemma: In any circle with n points in general position (no three chords concurrent), the planar graph formed by the chords has:
V = n + C(n, 4)
E = n + C(n, 2) + 2 * C(n, 4)
The number of bounded regions is F = 1 + C(n, 2) + C(n, 4).
</lemma_isolate>

<formal_synthesis>
Theorem: Let n >= 1 points be placed on the boundary of a circle such that no three chords intersect at a single interior point. The chords divide the interior of the circle into exactly
R(n) = \binom{n}{4} + \binom{n}{2} + 1
regions.

Proof:
Consider the geometric configuration as a connected planar graph G = (V, E) embedded inside the closed disk.
1. Vertices: The set V consists of the n boundary points and all interior intersection points between chords. Because no three chords intersect at a single point, every interior vertex is uniquely determined by a choice of four distinct boundary points whose chords cross. Thus, the number of interior vertices is \binom{n}{4}, yielding a total vertex count of:
   |V| = n + \binom{n}{4}

2. Edges: The edge set E consists of n boundary arcs along the perimeter of the circle and the segments into which the chords are divided by interior intersections. There are \binom{n}{2} chords. Each interior intersection point lies on exactly two chords and increases the number of edge segments by 2. Therefore, the total number of edges is:
   |E| = n + \binom{n}{2} + 2\binom{n}{4}

3. Faces: By Euler's formula for a connected planar graph, |V| - |E| + |F| = 2, where |F| includes the single unbounded exterior face. The number of interior regions is R(n) = |F| - 1. Substituting the vertex and edge counts:
   R(n) = |E| - |V| + 1
   R(n) = \left[ n + \binom{n}{2} + 2\binom{n}{4} \right] - \left[ n + \binom{n}{4} \right] + 1
   R(n) = \binom{n}{4} + \binom{n}{2} + 1

This completes the proof. Q.E.D.
</formal_synthesis>
```

### 4.4. Post-rigorous proof compilation pipelines

Building on Terence Tao’s three stages of mathematical cognition, foundation models should be structured as a multi-tier compilation pipeline:

```
Post-Rigorous Proof Compilation Pipeline:
+-----------------------------------------------------------------------------------------------+
| TIER 1: POST-RIGOROUS PLANNER (Macro-Strategy & Invariant Mapping)                            |
| - Identifies structural symmetries, conservation laws, asymptotic scalings, and invariants.  |
| - Outputs high-level strategic roadmap: "Apply Calderon-Zygmund decomposition to f; control  |
|   the good function with L2 energy estimates; bound the bad function using weak L1 bounds."  |
+-----------------------------------------------------------------------------------------------+
                                               |
                                               v
+-----------------------------------------------------------------------------------------------+
| TIER 2: RIGOROUS DESCENT ENGINE (Micro-Deduction & Inequality Elaboration)                    |
| - Takes each step of the Tier 1 roadmap and expands it into explicit algebraic identities,    |
|   epsilon-delta bounds, index bookkeeping, and domain checks.                                 |
| - Verifies dominated convergence hypotheses, compact support, and uniform continuity.        |
+-----------------------------------------------------------------------------------------------+
                                               |
                                               v
+-----------------------------------------------------------------------------------------------+
| TIER 3: FORMAL VERIFIER / LEAN 4 COMPILER (Kernel-Level Mechanical Verification)              |
| - Translates the rigorous natural language proof into a Lean 4 / Isabelle tactic script.      |
| - Executes `lake build` or tactic elaboration against Mathlib.                                 |
| - Kernel errors feedback directly into Tier 2 to fix missing hypotheses or local gaps.       |
+-----------------------------------------------------------------------------------------------+
```

This tiered architecture matches how human teams operate. A principal mathematician (Tier 1) devises the structural attack and isolates key lemmas. Graduate researchers and postdocs (Tier 2) compute the detailed bounds and verify technical hypotheses. An interactive theorem prover (Tier 3) checks the formal logic.

### 4.5. Reinforcement learning objectives: Asymmetric discovery and presentation rewards

Standard reinforcement learning on mathematical models uses outcome rewards:
$$R_{\text{outcome}} = \begin{cases} 1 & \text{if final answer is correct} \\ 0 & \text{otherwise} \end{cases}$$
This reward function rewards lucky guesses, penalizes valuable exploratory branches that happen to reach dead ends, and provides zero gradient for learning how to discover proofs.

To reflect pure mathematical cognition, reinforcement learning must apply asymmetric rewards across the two phases:

```
+-----------------------------------------------------------------------------------------------+
|                            ASYMMETRIC REINFORCEMENT LEARNING REWARDS                          |
+-----------------------+---------------------------------------+-------------------------------+
| Phase                 | Reward Signal                         | Mathematical Rationale        |
+-----------------------+---------------------------------------+-------------------------------+
| Phase 1: Exploration  | R_refute: Positive reward for finding | Lakatosian Refutation Reward: |
| Scratchpad            | a valid counterexample that disproves | Finding a counterexample      |
|                       | an intermediate conjecture.           | prevents fatal hallucination. |
|                       |                                       |                               |
|                       | R_diversity: Reward for visiting      | Polya Heuristic Breadth:      |
|                       | distinct sub-problems and extreme     | Broad exploration prevents    |
|                       | test cases (n=0, 1, parity).          | getting trapped in local minima.|
|                       |                                       |                               |
|                       | R_lemma: Reward for extracting        | Grothendieck Abstraction:     |
|                       | reusable, modular lemmas.             | Clean invariants compress     |
|                       |                                       | reasoning across sub-trees.   |
+-----------------------+---------------------------------------+-------------------------------+
| Phase 2: Formal       | R_valid: Binary reward from Lean 4 /  | Hilbert Rigor:                |
| Synthesis             | Isabelle kernel verification.         | Deductive validity must be    |
|                       |                                       | absolute and mechanical.      |
|                       |                                       |                               |
|                       | R_compaction: Penalty proportional to | Bourbaki Economy:             |
|                       | token length of final proof.          | Short, clean proofs with no   |
|                       |                                       | exploratory waste.            |
|                       |                                       |                               |
|                       | R_self_contain: Penalty for undefined | Rigorous Linearity:           |
|                       | variables or forward-referencing.     | Every term defined before use.|
+-----------------------+---------------------------------------+-------------------------------+
```

The total reward function separates the discovery evaluation from the presentation evaluation:
$$R_{\text{total}} = \alpha \cdot \left[ R_{\text{refute}}(\tau_{\text{explore}}) + R_{\text{lemma}}(\tau_{\text{explore}}) \right] + \beta \cdot \left[ R_{\text{valid}}(\tau_{\text{formal}}) + R_{\text{compaction}}(\tau_{\text{formal}}) \right]$$
where $\tau_{\text{explore}}$ is the trajectory of tokens inside the `<explore>` and `<counterexample_test>` tags, and $\tau_{\text{formal}}$ is the trajectory inside the `<formal_synthesis>` tag.

By decoupling the reward signals, the model learns:
- In the scratchpad: be bold, test edge cases, actively hunt for counterexamples, and do not fear dead ends.
- In the synthesis: be concise, rigorous, disciplined, and strictly deductive.

---

## 5. Comparative matrix: Epistemological thinkers and model architectures

The table below maps each classical and modern pure mathematical thinker to their discovery mechanics, presentation artifacts, and concrete algorithmic analogs for mathematical AI systems:

```
+---------------------------------------------------------------------------------------------------------------------------------------------+
|                                    PURE MATHEMATICS LITERATURE SYNTHESIS MATRIX                                                             |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Thinker & Work      | Core Epistemological Concept  | Real-Time Discovery Mechanism | Presentation Artifact | Algorithmic Analog for AI     |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Henri Poincare      | Unconscious incubation and    | Chaotic preparation followed   | Formal verification   | Speculative MCTS search with  |
| Science & Method    | the aesthetic sieve           | by subliminal stochastic      | paper (deductive      | learned aesthetic value       |
| (1908)              |                               | recombination of ideas        | checking)             | functions; prune sterile paths|
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Jacques Hadamard    | Non-verbal thought and mental | Multimodal, spatial, kinetic  | Linear algebraic      | Decouple internal latent      |
| Psychology of       | imagery in research           | and geometric visualization;  | notation and written  | search from external token    |
| Invention (1945)    |                               | words and symbols are absent  | language              | streams; continuous latents   |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| George Polya        | Heuristics and plausible      | Specialization (small n),      | Synthetic forward     | Tagged scratchpad with small  |
| How to Solve It     | reasoning (Analysis versus    | analogy, working backward     | proof (hides all      | case testing, Python REPL,    |
| (1945, 1954)        | Synthesis)                    | from goal to premises         | backward analysis)    | backward goal decomposition   |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Imre Lakatos        | Proofs, refutations, and      | Local and global counter-     | Sanitized theorem     | Adversarial counterexample    |
| Proofs &            | proof-generated concepts      | examples; monster-barring     | statement with hidden | generator; reward for finding |
| Refutations (1976)  |                               | and lemma incorporation       | lemmas built in       | bugs; dynamic lemma repair    |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Alexander           | The Rising Sea and            | Dissolve difficulties by      | Category-theoretic    | Modular lemma abstraction;    |
| Grothendieck        | conceptual dissolution        | constructing broad universal  | definitions, schemes, | identify universal invariants |
| Recoltes (1986)     |                               | properties and functors       | commutative diagrams  | that make sub-proofs trivial  |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| William Thurston    | Human understanding versus    | Multimodal communication,     | Machine-checkable     | Multi-level proof compiler;   |
| Proof & Progress    | formal logic strings          | gesture, mental models,       | strings of formal     | high-level conceptual sketch  |
| (1994)              |                               | colloquial debate             | Peano axioms          | lowered into Lean 4 tactics   |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Terence Tao         | Three cognitive stages        | Free movement between         | Rigorous formal paper | Three-tier inference engine:  |
| Rigour & Proofs     | (pre-rigorous, rigorous,      | post-rigorous structural vision| with all technical   | Post-rigorous plan -> Rigorous|
| (2008)              | post-rigorous); structure/rand| and micro-inequality bounds   | estimates verified    | inequalities -> Formal check  |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Nicolas Bourbaki    | Extreme structuralism and     | Completely concealed;         | Definition -> Lemma   | Deterministic Phase 2 proof   |
| Elements de         | deductive linearization       | treated as irrelevant to      | -> Theorem -> Cor.    | compiler; token-efficiency    |
| mathematique (1939) |                               | mathematical truth            | deductive cascade     | penalty; strict ordering      |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Felix Klein         | Invariant theory and spatial  | Spatial group actions and     | Algebraic invariant   | Geometric graph encoders and  |
| Erlangen (1872)     | geometric intuition           | transformation symmetries     | polynomial equations  | group-equivariant attention   |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| David Hilbert       | Problem-driven engine and     | Concrete unsolved problems    | Axiomatic foundations | Benchmark curation driven by  |
| Paris (1900)        | axiomatic organization        | forcing concept creation      | (Grundlagen)          | hard open problems, not tests |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
| Michael Atiyah      | Geometry versus Algebra;      | Geometric global vision       | Pure algebraic        | Balance intuitive spatial     |
| 20th Century (2001) | Visionaries vs Repairers      | complemented by calculations  | formalisms            | models with symbolic checkers |
+---------------------+-------------------------------+-------------------------------+-----------------------+-------------------------------+
```

---

## 6. Operational recommendations for mathematical AI systems

1. **Split the Training Data by Cognitive Mode**:
   Do not train models exclusively on finished proofs. Create curated datasets of messy exploration traces:
   - Include failed proof attempts that hit known counterexamples.
   - Include scratchpads that test $n=1, 2, 3$, discover a broken conjecture, and repair the hypothesis.
   - Separate training tokens into `<explore>` and `<compile>` partitions.

2. **Decouple Test-Time Compute Across Two Phases**:
   - Allocate 80% of test-time compute budget to Phase 1 (stochastic search, counterexample probing, branching exploration).
   - Allocate 20% of compute budget to Phase 2 (deterministic proof compilation, proof compaction, Lean 4 verification).

3. **Incorporate Python REPL as an Adversarial Tool in the Scratchpad**:
   When the model enters `<counterexample_test>`, grant it access to a sandboxed Python execution environment. The model should write SymPy, NumPy, or brute-force scripts to test candidate conjectures on integers $n \in [0, 1000]$ before asserting them as lemmas.

4. **Reward Falsification, Not Just Verification**:
   In reinforcement learning (RLVR / PPO / GRPO), reward the exploration policy when it successfully refutes an invalid candidate conjecture. A model that can identify its own false steps avoids cascading hallucinations.

5. **Compile from Post-Rigorous Sketches to Formal Proofs**:
   Do not force the model to generate low-level Lean 4 or epsilon-delta steps directly from the prompt. Prompt the model to produce a Tao-style post-rigorous structural sketch, then run a secondary lowering pass that converts each step into verifiable formal mathematics.

---

## 7. Primary source bibliography

- **Atiyah, Michael.** "Mathematics in the 20th Century." *Bulletin of the London Mathematical Society*, vol. 34, no. 1, 2002, pp. 1-15.
- **Atiyah, Michael.** "Advice to a Young Mathematician." In *The Princeton Companion to Mathematics*, edited by Timothy Gowers, June Barrow-Green, and Imre Leader, Princeton University Press, 2008, pp. 1000-1011.
- **Bourbaki, Nicolas.** *Éléments de mathématique: Théorie des ensembles.* Hermann, Paris, 1939-1970.
- **Dieudonné, Jean.** *Pour l'honneur de l'esprit humain: les mathématiques aujourd'hui.* Hachette, Paris, 1987.
- **Grothendieck, Alexander.** *Récoltes et Semailles: Réflexions et témoignage sur un passé de mathématicien.* Université des Sciences et Techniques du Languedoc, Montpellier, 1986 (Gallimard, Paris, 2022).
- **Hadamard, Jacques.** *An Essay on the Psychology of Invention in the Mathematical Field.* Princeton University Press, Princeton, NJ, 1945.
- **Hilbert, David.** "Mathematische Probleme." *Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse*, 1900, pp. 253-297.
- **Hilbert, David.** *Grundlagen der Geometrie.* B. G. Teubner, Leipzig, 1899.
- **Klein, Felix.** *Vergleichende Betrachtungen über neuere geometrische Forschungen (Erlanger Programm).* A. Deichert, Erlangen, 1872.
- **Klein, Felix.** *Elementarmathematik vom höheren Standpunkte aus.* B. G. Teubner, Leipzig, 1908.
- **Lakatos, Imre.** *Proofs and Refutations: The Logic of Mathematical Discovery.* Edited by John Worrall and Elie Zahar, Cambridge University Press, Cambridge, UK, 1976.
- **Poincaré, Henri.** *La Science et l'Hypothèse.* Flammarion, Paris, 1902.
- **Poincaré, Henri.** "L'Invention mathématique." *Bulletin de l'Institut Général Psychologique*, vol. 8, no. 3, 1908, pp. 175-187; reprinted in *Science et Méthode*, Flammarion, Paris, 1908.
- **Pólya, George.** *How to Solve It: A New Aspect of Mathematical Method.* Princeton University Press, Princeton, NJ, 1945.
- **Pólya, George.** *Mathematics and Plausible Reasoning, Volume 1: Induction and Analogy in Mathematics.* Princeton University Press, Princeton, NJ, 1954.
- **Pólya, George.** *Mathematics and Plausible Reasoning, Volume 2: Patterns of Plausible Inference.* Princeton University Press, Princeton, NJ, 1954.
- **Tao, Terence.** *Solving Mathematical Problems: A Personal Perspective.* Oxford University Press, Oxford, UK, 2006.
- **Tao, Terence.** "There's more to mathematics than rigour and proofs." Blog post, *What's new*, September 14, 2008. https://terrytao.wordpress.com/career-advice/theres-more-to-mathematics-than-rigour-and-proofs/
- **Tao, Terence.** *Structure and Randomness: Pages from Year One of a Mathematical Blog.* American Mathematical Society, Providence, RI, 2008.
- **Tao, Terence.** *Compactness and Contradiction.* American Mathematical Society, Providence, RI, 2013.
- **Thurston, William P.** "On Proof and Progress in Mathematics." *Bulletin of the American Mathematical Society*, vol. 30, no. 2, 1994, pp. 161-177.
