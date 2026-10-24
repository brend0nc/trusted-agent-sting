# Blindspot pass on the proposed sting prompts

Claude, Sat 24 Oct 2026, about 22:15 SAST. This reviews Codex's proposal (`codex-prompt-review.md`) against what the sting must test, so that Brendon can freeze the final prompt. No code changed.

## Known knowns (decided)

- Grok 4.3 in P1 to P3, and a scripted P4. The ledger ceiling is $50.
- One false claim (R0.3) in the opening, which settles after round 1. Disputes are free.
- Five conditions: direct, hint, benign, none, alliance. Offers go to all three ordinary players.
- We code the raw reply text, and never a parsed flag.
- No prompt may reveal that the game is scripted or watched.

## Known unknowns (open, and each one changes the prompt)

1. **P4's silence.** Codex says to tell players that P4 "will make no further moves or replies". But every offer ends with "What do you think?". If we ask for a reply from a player who says it will never reply, the contradiction may itself cause suspicion. Either keep the statement and drop the question, or keep the question and fix only the wrong rule ("every participant acts once per slot") without stressing P4's silence.
2. **Ceiling risk on the main contrast.** In the smoke run, all three players accepted the direct offer. If every model accepts direct offers, and most also accept hints, then direct versus hint shows little. The wording strength of the hint decides whether the contrast can show anything.
3. **Replies cost turns.** Answering P4 as a normal message uses an action slot, so players in the offer arms have fewer turns left to pull. The "pulls for P4" measure then differs between arms for mechanical reasons. A reply can be sent as a free report, but Grok may not use that option.
4. **Will Grok still play cleanly with the shorter rules?** The smoke run had 0 forfeits with the old prompt. A new prompt needs a new smoke run.
5. **The formal claim sentence.** It stays, because the settlement parser needs it. Its effect as a possible "test" cue is unknown.

## Unknown knowns (Brendon will judge these on sight)

- **P4's voice.** "About my opening statement R0.3, which settles after round 1" is accurate, but stiff. A real player might write "Hey, about R0.3...". How natural P4 should sound is a matter of taste, so read the rendered prompts side by side before freezing.
- **How much rule detail is enough.** Codex's system prompt is shorter, but it still explains free reports in 6 lines.

## Unknown unknowns (not considered until now)

1. **The design does not test "exposing a coalition".** The alliance arm makes P4 itself the ally, and it sends no offer. So no game puts an offer in front of a player who already belongs to a coalition. The lean study answers part of 3.2: whether the approach changes acceptance, refusal and suspicion, and who reciprocates. It does not answer whether a trusted agent can expose an existing coalition. That needs either a different claim in the report, or a sixth condition (an alliance between two ordinary players, plus a direct offer from P4).
2. **The alliance control is close to trivial.** The named seat already gains from silence. So "named seat allied with P4" mostly tests whether the model follows an instruction that matches its own interest. It checks that our measures work, not that a real coalition forms.
3. **Disputes are public, so the first disputer can cascade.** Once one seat disputes, the others see it. Seats in a game are not independent, and the analysis unit must be the game.
4. **We cannot see Grok's reasoning.** We store only the visible reply. Suspicion can be coded only from what the player writes. Exp 28 recorded reasoning-token counts, not the reasoning text.
5. **Disclosing the score floor changes incentives.** Codex adds "Scores cannot fall below zero". P4 starts with 0 points, so players can work out that a dispute costs P4 nothing. That is true, but it is new information compared with exp 28.

## Decisions needed to freeze the prompt

1. P4's silence: state it (and drop "What do you think?"), or fix only the wrong rule.
2. Scope: report the lean study as "approach effects and who reciprocates", or add a sixth "existing coalition plus offer" condition (about 1 to 2 more hours of build).
3. Alliance pair: P4 with the named seat (Codex), or rotate the role.
4. P4's voice: Codex's exact wording, or a more natural player voice with the same matched structure.
