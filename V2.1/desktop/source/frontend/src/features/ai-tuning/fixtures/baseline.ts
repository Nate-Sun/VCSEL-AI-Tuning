export const FIR_TAP_LABELS = [
  "Pre-3",
  "Pre-2",
  "Pre-1",
  "Main",
  "Post-1",
  "Post-2",
  "Post-3",
] as const;

export const B0 = Object.freeze({
  pattern: "SSPRQ" as const,
  fir: Object.freeze({
    taps: Object.freeze([0, 9, -28, 104, -8, 1, -3] as const),
    upper_nlc: 6,
    lower_nlc: 6,
  }),
  i2c: Object.freeze({
    ibias_dac: 70,
    vref: 15,
    vbias_ref: 15,
    vga_cur: 13,
    res_gain: 6,
    res_bw: 0,
    gain_ctrl: 5,
    efin_cur: 15,
    eq: 22,
  }),
});
