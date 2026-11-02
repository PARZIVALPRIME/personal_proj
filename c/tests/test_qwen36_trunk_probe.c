/* q36_trunk_probe_verdict(): whether the dense trunk goes to VRAM. The probe
 * times one synchronous GEMV on the device; the CUDA chain pays that call's
 * round trip once per several GEMVs, so with the chain on the round trip is
 * spread over COLI_TRUNK_PROBE_PER_SYNC of them. Without the chain, or with
 * per_sync 1, the verdict is the old one: GPU time against CPU time. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

int main(void) {
    double eff = 0;

    /* no chain: the synchronous time decides, as before */
    CHECK(q36_trunk_probe_verdict(0.40e-3, 0.30e-3, 0.35e-3, 0, 4, &eff) == 0);
    CHECK(eff == 0.40e-3);
    CHECK(q36_trunk_probe_verdict(0.30e-3, 0.20e-3, 0.35e-3, 0, 4, NULL) == 1);

    /* the laptop case: a GEMV that loses only by its round trip wins in the chain
     * (0.10 ms kernel + 0.30/4 ms of round trip = 0.175 ms < 0.35 ms) */
    CHECK(q36_trunk_probe_verdict(0.40e-3, 0.30e-3, 0.35e-3, 1, 4, &eff) == 1);
    CHECK(eff > 0.174e-3 && eff < 0.176e-3);

    /* per_sync 1 (or less) is the old verdict even with the chain */
    CHECK(q36_trunk_probe_verdict(0.40e-3, 0.30e-3, 0.35e-3, 1, 1, &eff) == 0 && eff == 0.40e-3);
    CHECK(q36_trunk_probe_verdict(0.40e-3, 0.30e-3, 0.35e-3, 1, 0, NULL) == 0);

    /* a slow kernel still loses: the M10 of #1652 stays on the CPU */
    CHECK(q36_trunk_probe_verdict(68.8e-3, 0.05e-3, 41.7e-3, 1, 4, NULL) == 0);

    /* no round-trip measurement (0) leaves the time as it is */
    CHECK(q36_trunk_probe_verdict(0.40e-3, 0.0, 0.35e-3, 1, 4, &eff) == 0 && eff == 0.40e-3);

    /* a round trip measured above the whole GEMV (noise) never makes it negative */
    CHECK(q36_trunk_probe_verdict(0.40e-3, 0.90e-3, 0.05e-3, 1, 4, &eff) == 0);
    CHECK(eff > 0.0 && eff <= 0.40e-3);

    printf(fails ? "test_qwen36_trunk_probe: FAIL (%d)\n" : "test_qwen36_trunk_probe: PASS\n", fails);
    return fails != 0;
}
