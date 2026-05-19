# Run Log
| Date (UTC) | Commit | Config | Seed | Artifacts (path or Drive link) | Notes |
|------------|--------|--------|------|---------------------------------|-------|
| 2026-05-19 | claude/init-project-setup | smoke.yaml | 0 | local /tmp (not uploaded) | Phase 1 baseline wired up; 20-episode CPU smoke runs end-to-end, recorder/replay verified. Predator reward stays at 0 in tiny-config smoke (no catches in 20 steps with random-ish policies) — expected; revisit with base.yaml on Colab. |
| 2026-05-19 | claude/init-project-setup | preview.yaml | 7 | local /tmp (not uploaded) | Phase 2 — 80-ep baseline (no novelty): greedy eval over 3 ep yielded predator_return=20.0, prey=-112.8. |
| 2026-05-19 | claude/init-project-setup | preview_novelty.yaml | 7 | local /tmp (not uploaded) | Phase 2 — 80-ep novelty-on (bonus_coef=0.05): greedy eval over 3 ep yielded predator_return=63.3 (3.2× baseline), prey=-61.5. Novelty score climbs steadily for both teams — bonus is doing useful work at this scale. Worth re-running with longer schedules to see if the gap persists. |
