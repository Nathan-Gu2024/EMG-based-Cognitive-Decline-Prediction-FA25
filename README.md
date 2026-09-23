// ---------------------------------------------------------------------------
// Banked RMW for RM / CMM release bitmaps
//   * per-word clock enable built from word-level strobes (no 32-in OR of data)
//   * per-word index decode: one shared 5->32 decoder + 8-bit const compare/bank
//   * clear-dominant next state:  q <= (q | set) & ~clr
//     (priority: rel_clr, cpu_clr  >  cpu_set  >  index set)
//
// Assumes:
//   o_cet_release_id, r_cet_release_pending : [$clog2(NIE)-1:0]
//   r_rm_release_bitmap, r_cmm_release_bitmap, w_cet_release_bitmap : logic [NIE-1:0]
//   w_rm_input_set_bm, w_cmm_input_set_bm driven elsewhere
// ---------------------------------------------------------------------------
localparam int W  = 32;              // bank width == SFR word width
localparam int NW = NIE / W;         // 144 banks for NIE = 4608
localparam int WB = $clog2(W);       // 5  : bit-in-word index width
localparam int IW = $clog2(NIE);     // 13 : full index width

// elaboration-time sanity checks
if (NIE % W != 0)    begin : g_chk_nie  $error("NIE must be a multiple of 32");           end
if (AHB_HST_DW != W) begin : g_chk_dw   $error("banking assumes 32-bit SFR words");       end

// shared bit-in-word decoders (one of each for the whole array)
logic [W-1:0] rel_bit_oh, pend_bit_oh;
assign rel_bit_oh  = W'(1) << o_cet_release_id     [WB-1:0];
assign pend_bit_oh = W'(1) << r_cet_release_pending[WB-1:0];

for (genvar bw = 0; bw < NW; bw++) begin : g_bank
    localparam logic [IW-WB-1:0] WORD = bw[IW-WB-1:0];

    // ---- word-level index hits (8-bit compare against a constant) --------
    wire rel_hit  = o_cet_release_req         & (o_cet_release_id     [IW-1:WB] == WORD);
    wire pend_hit = r_cet_release_pending_vld & (r_cet_release_pending[IW-1:WB] == WORD);

    wire [W-1:0] rel_clr = {W{rel_hit}}  & rel_bit_oh;
    wire [W-1:0] pend_oh = {W{pend_hit}} & pend_bit_oh;

    // ---- index-set inputs --------------------------------------------------
    // If these come from an index upstream, pass a word-hit in instead and use
    // it in ce_* below rather than the 32-input OR-reduce.
    wire [W-1:0] rm_in  = w_rm_input_set_bm [bw*W +: W];
    wire [W-1:0] cmm_in = w_cmm_input_set_bm[bw*W +: W];

    // ---- per-bit set / clear masks -----------------------------------------
    wire [W-1:0] rm_clr  = ({W{i_BitMapCtrl_RegIsRMClr_write_access[bw]}}
                            & i_BitMapCtrl_RegIsRMClr_write_data [bw*W +: W]) | rel_clr;
    wire [W-1:0] rm_set  = ({W{i_BitMapCtrl_RegIsRMSet_write_access[bw]}}
                            & i_BitMapCtrl_RegIsRMSet_write_data [bw*W +: W]) | rm_in;

    wire [W-1:0] cmm_clr = ({W{i_BitMapCtrl_RegIsCMMClr_write_access[bw]}}
                            & i_BitMapCtrl_RegIsCMMClr_write_data[bw*W +: W]) | rel_clr;
    wire [W-1:0] cmm_set = ({W{i_BitMapCtrl_RegIsCMMSet_write_access[bw]}}
                            & i_BitMapCtrl_RegIsCMMSet_write_data[bw*W +: W]) | cmm_in;

    // ---- per-word clock enables (word strobes, not data OR-reduce) ---------
    // Enabling on an access with zero data is harmless: (q|0)&~0 == q.
    wire ce_rm  = i_BitMapCtrl_RegIsRMClr_write_access[bw]
                | i_BitMapCtrl_RegIsRMSet_write_access[bw]
                | rel_hit | (|rm_in);
    wire ce_cmm = i_BitMapCtrl_RegIsCMMClr_write_access[bw]
                | i_BitMapCtrl_RegIsCMMSet_write_access[bw]
                | rel_hit | (|cmm_in);

    // ---- storage: one local register per bank (single always_ff writer) ---
    logic [W-1:0] rm_q, cmm_q;

    always_ff @(posedge i_clk or negedge i_rst_b) begin
        if (!i_rst_b)   rm_q <= '0;
        else if (ce_rm) rm_q <= (rm_q | rm_set) & ~rm_clr;
    end

    always_ff @(posedge i_clk or negedge i_rst_b) begin
        if (!i_rst_b)    cmm_q <= '0;
        else if (ce_cmm) cmm_q <= (cmm_q | cmm_set) & ~cmm_clr;
    end

    // ---- outputs / read plane ----------------------------------------------
    assign r_rm_release_bitmap [bw*W +: W] = rm_q;
    assign r_cmm_release_bitmap[bw*W +: W] = cmm_q;
    assign w_cet_release_bitmap[bw*W +: W] = rm_q & cmm_q & ~pend_oh;
end
