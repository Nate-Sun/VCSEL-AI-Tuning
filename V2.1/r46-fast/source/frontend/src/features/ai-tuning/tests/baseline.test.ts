import { B0, FIR_TAP_LABELS } from "../fixtures/baseline";

describe("commissioned B0", () => {
  it("keeps the exact labeled FIR order", () => {
    expect(FIR_TAP_LABELS).toEqual([
      "Pre-3",
      "Pre-2",
      "Pre-1",
      "Main",
      "Post-1",
      "Post-2",
      "Post-3",
    ]);
    expect(B0.pattern).toBe("SSPRQ");
    expect(B0.fir.taps).toEqual([0, 9, -28, 104, -8, 1, -3]);
    expect(B0.fir.upper_nlc).toBe(6);
    expect(B0.fir.lower_nlc).toBe(6);
  });

  it("keeps the exact I2C baseline values", () => {
    expect(B0.i2c).toEqual({
      ibias_dac: 70,
      vref: 15,
      vbias_ref: 15,
      vga_cur: 13,
      res_gain: 6,
      res_bw: 0,
      gain_ctrl: 5,
      efin_cur: 15,
      eq: 22,
    });
  });
});
